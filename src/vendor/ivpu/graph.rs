//! Level Zero graph-extension execution: load an OpenVINO-compiled native blob, bind host
//! buffers, submit, and synchronize. This is the exact sequence validated in the C++ spike
//! (blob loads as `ZE_GRAPH_FORMAT_NATIVE`; graph args are FP32 on-device). `no_std`; uses
//! the loader in [`super::ze`]. Errors surface as the raw `ze_result_t` (or a small negative
//! sentinel); the function is panic-free.

use alloc::vec::Vec;
use core::ffi::c_void;
use core::mem;
use core::ptr;

use super::ze;
use super::ze::raw::*;

fn chk(r: ze_result_t) -> Result<(), i32> {
    if r == ZE_RESULT_SUCCESS {
        Ok(())
    } else {
        Err(r as i32)
    }
}

/// Run `blob` on `dev`, feeding its INPUT arguments in order from `inputs` (each f32) and
/// copying the first OUTPUT argument into `output` (f32). Returns the number of f32 outputs
/// written. `inputs` must be in the graph's input-argument order (for our MatMul blob:
/// `[activation, weight]`); each is only written if its length matches the arg.
///
/// Assumes FP32 device I/O (true for the OpenVINO f32 MatMul); other precisions are left
/// zero-filled for now. All Level Zero objects are cleaned up before returning.
pub(super) fn run_native_blob(
    dev: &ze::Device,
    blob: &[u8],
    inputs: &[&[f32]],
    output: &mut [f32],
) -> Result<usize, i32> {
    let l = ze::lib().ok_or(-1)?;
    unsafe {
        // Context.
        let mut ctx_desc: ze_context_desc_t = mem::zeroed();
        ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        let mut ctx: ze_context_handle_t = ptr::null_mut();
        chk((l.context_create)(dev.driver, &ctx_desc, &mut ctx))?;

        // Run the body; always tear the context down afterwards.
        let r = run_inner(l, dev, ctx, blob, inputs, output);
        (l.context_destroy)(ctx);
        r
    }
}

unsafe fn run_inner(
    l: &ze::ZeLib,
    dev: &ze::Device,
    ctx: ze_context_handle_t,
    blob: &[u8],
    inputs: &[&[f32]],
    output: &mut [f32],
) -> Result<usize, i32> {
    // Graph dispatch table (v1.0 fields only; base name gives a >=1.0 table).
    let mut table: *mut c_void = ptr::null_mut();
    chk((l.get_ext_fn_addr)(
        dev.driver,
        c"ZE_extension_graph".as_ptr(),
        &mut table,
    ))?;
    let g = table as *mut ze_graph_dditable_ext_t;
    if g.is_null() {
        return Err(-2);
    }
    let pfn_create = (*g).pfnCreate.ok_or(-3)?;
    let pfn_props = (*g).pfnGetProperties.ok_or(-3)?;
    let pfn_arg = (*g).pfnGetArgumentProperties.ok_or(-3)?;
    let pfn_set = (*g).pfnSetArgumentValue.ok_or(-3)?;
    let pfn_init = (*g).pfnAppendGraphInitialize.ok_or(-3)?;
    let pfn_exec = (*g).pfnAppendGraphExecute.ok_or(-3)?;

    // Create the graph from the native blob.
    let mut gd: ze_graph_desc_t = mem::zeroed();
    gd.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC;
    gd.format = ZE_GRAPH_FORMAT_NATIVE;
    gd.inputSize = blob.len();
    gd.pInput = blob.as_ptr();
    let mut graph: ze_graph_handle_t = ptr::null_mut();
    chk(pfn_create(ctx, dev.handle, &gd, &mut graph))?;

    let mut gp: ze_graph_properties_t = mem::zeroed();
    gp.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    chk(pfn_props(graph, &mut gp))?;

    // Allocate a host buffer per argument, bind it, fill inputs.
    let mut host_desc: ze_host_mem_alloc_desc_t = mem::zeroed();
    host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    let mut mems: Vec<*mut c_void> = Vec::new();
    let mut out_ptr: *mut c_void = ptr::null_mut();
    let mut out_elems: usize = 0;
    let mut input_idx: usize = 0;

    for i in 0..gp.numGraphArgs {
        let mut ap: ze_graph_argument_properties_t = mem::zeroed();
        ap.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        chk(pfn_arg(graph, i, &mut ap))?;

        let mut elems: usize = 1;
        for k in 0..(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE as usize) {
            let d = ap.dims[k];
            if d > 0 {
                elems *= d as usize;
            }
        }
        let psz: usize = match ap.devicePrecision {
            ZE_GRAPH_ARGUMENT_PRECISION_FP32 => 4,
            ZE_GRAPH_ARGUMENT_PRECISION_FP16 => 2,
            _ => 0,
        };
        let bytes = elems * psz;

        let mut mem_ptr: *mut c_void = ptr::null_mut();
        if chk((l.mem_alloc_host)(
            ctx,
            &host_desc,
            bytes,
            4096,
            &mut mem_ptr,
        ))
        .is_err()
        {
            free_all(l, ctx, &mems);
            return Err(-4);
        }
        mems.push(mem_ptr);
        ptr::write_bytes(mem_ptr as *mut u8, 0, bytes);

        if ap.type_ == ZE_GRAPH_ARGUMENT_TYPE_INPUT {
            if psz == 4 {
                if let Some(src) = inputs.get(input_idx) {
                    if src.len() == elems {
                        let dst = mem_ptr as *mut f32;
                        for j in 0..elems {
                            *dst.add(j) = src[j];
                        }
                    }
                }
            }
            input_idx += 1;
        } else {
            out_ptr = mem_ptr;
            out_elems = elems;
        }

        if chk(pfn_set(graph, i, mem_ptr)).is_err() {
            free_all(l, ctx, &mems);
            return Err(-5);
        }
    }

    // Command queue + list; append graph init then execute; run and wait.
    let mut qd: ze_command_queue_desc_t = mem::zeroed();
    qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.ordinal = 0;
    qd.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    let mut queue: ze_command_queue_handle_t = ptr::null_mut();
    chk((l.cmd_queue_create)(ctx, dev.handle, &qd, &mut queue))?;

    let mut ld: ze_command_list_desc_t = mem::zeroed();
    ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    ld.commandQueueGroupOrdinal = 0;
    let mut cl: ze_command_list_handle_t = ptr::null_mut();
    chk((l.cmd_list_create)(ctx, dev.handle, &ld, &mut cl))?;

    let mut result: Result<usize, i32> = (|| {
        chk(pfn_init(cl, graph, ptr::null_mut(), 0, ptr::null_mut()))?;
        chk(pfn_exec(
            cl,
            graph,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        ))?;
        chk((l.cmd_list_close)(cl))?;
        chk((l.cmd_queue_exec)(queue, 1, &mut cl, ptr::null_mut()))?;
        chk((l.cmd_queue_sync)(queue, u64::MAX))?;
        Ok(0)
    })();

    // Copy the output out.
    if result.is_ok() && !out_ptr.is_null() {
        let n = core::cmp::min(output.len(), out_elems);
        let src = out_ptr as *const f32;
        for j in 0..n {
            output[j] = *src.add(j);
        }
        result = Ok(n);
    }

    (l.cmd_list_destroy)(cl);
    (l.cmd_queue_destroy)(queue);
    free_all(l, ctx, &mems);
    if let Some(destroy) = (*g).pfnDestroy {
        destroy(graph);
    }
    result
}

unsafe fn free_all(l: &ze::ZeLib, ctx: ze_context_handle_t, mems: &[*mut c_void]) {
    for &p in mems {
        (l.mem_free)(ctx, p);
    }
}
