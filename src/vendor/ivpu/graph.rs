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
    let mut fixture = PersistentNativeGraph::new(dev, blob, inputs)?;
    fixture.execute(2_000_000_000)?;
    fixture.copy_output(output)
}

/// Query Level Zero graph argument byte sizes without executing (FP32/compatible widths only).
pub(super) fn native_blob_binding_bytes(
    dev: &ze::Device,
    blob: &[u8],
) -> Result<(Vec<u64>, Vec<u64>), i32> {
    let l = ze::lib().ok_or(-1)?;
    // SAFETY: Level Zero handles are initialized before use and destroyed before return.
    unsafe {
        let mut table: *mut c_void = ptr::null_mut();
        chk((l.get_ext_fn_addr)(
            dev.driver,
            c"ZE_extension_graph".as_ptr(),
            &mut table,
        ))?;
        let graph_table = table.cast::<ze_graph_dditable_ext_t>();
        if graph_table.is_null() {
            return Err(-2);
        }
        let pfn_create = (*graph_table).pfnCreate.ok_or(-3)?;
        let pfn_props = (*graph_table).pfnGetProperties.ok_or(-3)?;
        let pfn_arg = (*graph_table).pfnGetArgumentProperties.ok_or(-3)?;
        let pfn_destroy = (*graph_table).pfnDestroy.ok_or(-3)?;

        let mut context_desc: ze_context_desc_t = mem::zeroed();
        context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        let mut context = ptr::null_mut();
        chk((l.context_create)(dev.driver, &context_desc, &mut context))?;
        let mut graph = ptr::null_mut();
        let graph_desc = ze_graph_desc_t {
            stype: ZE_STRUCTURE_TYPE_GRAPH_DESC,
            pNext: ptr::null_mut(),
            format: ZE_GRAPH_FORMAT_NATIVE,
            inputSize: blob.len(),
            pInput: blob.as_ptr(),
            pBuildFlags: ptr::null(),
        };
        let create = pfn_create(context, dev.handle, &graph_desc, &mut graph);
        if create != ZE_RESULT_SUCCESS {
            (l.context_destroy)(context);
            return Err(create as i32);
        }
        let mut properties: ze_graph_properties_t = mem::zeroed();
        properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        let props = pfn_props(graph, &mut properties);
        if props != ZE_RESULT_SUCCESS {
            pfn_destroy(graph);
            (l.context_destroy)(context);
            return Err(props as i32);
        }
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for argument in 0..properties.numGraphArgs {
            let mut argument_properties: ze_graph_argument_properties_t = mem::zeroed();
            argument_properties.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
            let arg = pfn_arg(graph, argument, &mut argument_properties);
            if arg != ZE_RESULT_SUCCESS {
                pfn_destroy(graph);
                (l.context_destroy)(context);
                return Err(arg as i32);
            }
            let elements = argument_properties
                .dims
                .iter()
                .copied()
                .filter(|dimension| *dimension != 0)
                .try_fold(1usize, |product, dimension| {
                    product.checked_mul(dimension as usize).ok_or(-4)
                })?;
            let element_bytes = match argument_properties.devicePrecision {
                ZE_GRAPH_ARGUMENT_PRECISION_FP32
                | ZE_GRAPH_ARGUMENT_PRECISION_INT32
                | ZE_GRAPH_ARGUMENT_PRECISION_UINT32 => 4usize,
                ZE_GRAPH_ARGUMENT_PRECISION_FP16 => 2,
                ZE_GRAPH_ARGUMENT_PRECISION_INT64 | ZE_GRAPH_ARGUMENT_PRECISION_UINT64 => 8,
                _ => {
                    pfn_destroy(graph);
                    (l.context_destroy)(context);
                    return Err(-4);
                }
            };
            let bytes = (elements.checked_mul(element_bytes).ok_or(-4)?) as u64;
            if argument_properties.type_ == ZE_GRAPH_ARGUMENT_TYPE_INPUT {
                inputs.push(bytes);
            } else {
                outputs.push(bytes);
            }
        }
        pfn_destroy(graph);
        (l.context_destroy)(context);
        Ok((inputs, outputs))
    }
}

/// Reusable Level Zero fixture used by the one-shot self-test and guarded hardware benchmark.
pub(super) struct PersistentNativeGraph {
    l: &'static ze::ZeLib,
    graph_table: *mut ze_graph_dditable_ext_t,
    context: ze_context_handle_t,
    graph: ze_graph_handle_t,
    queue: ze_command_queue_handle_t,
    execute_list: ze_command_list_handle_t,
    memories: Vec<*mut c_void>,
    output: *mut f32,
    output_elements: usize,
    failed: bool,
}

impl PersistentNativeGraph {
    pub(super) fn new(dev: &ze::Device, blob: &[u8], inputs: &[&[f32]]) -> Result<Self, i32> {
        let l = ze::lib().ok_or(-1)?;
        // SAFETY: every Level Zero handle is initialized before use and owned by the fixture.
        unsafe {
            let mut table: *mut c_void = ptr::null_mut();
            chk((l.get_ext_fn_addr)(
                dev.driver,
                c"ZE_extension_graph".as_ptr(),
                &mut table,
            ))?;
            let graph_table = table.cast::<ze_graph_dditable_ext_t>();
            if graph_table.is_null() {
                return Err(-2);
            }
            let pfn_create = (*graph_table).pfnCreate.ok_or(-3)?;
            let pfn_props = (*graph_table).pfnGetProperties.ok_or(-3)?;
            let pfn_arg = (*graph_table).pfnGetArgumentProperties.ok_or(-3)?;
            let pfn_set = (*graph_table).pfnSetArgumentValue.ok_or(-3)?;
            let pfn_init = (*graph_table).pfnAppendGraphInitialize.ok_or(-3)?;
            let pfn_exec = (*graph_table).pfnAppendGraphExecute.ok_or(-3)?;

            let mut context_desc: ze_context_desc_t = mem::zeroed();
            context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
            let mut context = ptr::null_mut();
            chk((l.context_create)(dev.driver, &context_desc, &mut context))?;
            let mut fixture = Self {
                l,
                graph_table,
                context,
                graph: ptr::null_mut(),
                queue: ptr::null_mut(),
                execute_list: ptr::null_mut(),
                memories: Vec::new(),
                output: ptr::null_mut(),
                output_elements: 0,
                failed: false,
            };

            let graph_desc = ze_graph_desc_t {
                stype: ZE_STRUCTURE_TYPE_GRAPH_DESC,
                pNext: ptr::null_mut(),
                format: ZE_GRAPH_FORMAT_NATIVE,
                inputSize: blob.len(),
                pInput: blob.as_ptr(),
                pBuildFlags: ptr::null(),
            };
            chk(pfn_create(
                fixture.context,
                dev.handle,
                &graph_desc,
                &mut fixture.graph,
            ))?;

            let mut properties: ze_graph_properties_t = mem::zeroed();
            properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
            chk(pfn_props(fixture.graph, &mut properties))?;

            let mut host_desc: ze_host_mem_alloc_desc_t = mem::zeroed();
            host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
            let mut input_index = 0usize;
            for argument in 0..properties.numGraphArgs {
                let mut argument_properties: ze_graph_argument_properties_t = mem::zeroed();
                argument_properties.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
                chk(pfn_arg(fixture.graph, argument, &mut argument_properties))?;
                let elements = argument_properties
                    .dims
                    .iter()
                    .copied()
                    .filter(|dimension| *dimension != 0)
                    .try_fold(1usize, |product, dimension| {
                        product.checked_mul(dimension as usize).ok_or(-4)
                    })?;
                let element_bytes = match argument_properties.devicePrecision {
                    ZE_GRAPH_ARGUMENT_PRECISION_FP32
                    | ZE_GRAPH_ARGUMENT_PRECISION_INT32
                    | ZE_GRAPH_ARGUMENT_PRECISION_UINT32 => 4,
                    ZE_GRAPH_ARGUMENT_PRECISION_FP16 => 2,
                    ZE_GRAPH_ARGUMENT_PRECISION_INT64 | ZE_GRAPH_ARGUMENT_PRECISION_UINT64 => 8,
                    _ => return Err(-4),
                };
                let bytes = elements.checked_mul(element_bytes).ok_or(-4)?;
                let mut memory = ptr::null_mut();
                chk((l.mem_alloc_host)(
                    fixture.context,
                    &host_desc,
                    bytes,
                    4096,
                    &mut memory,
                ))?;
                fixture.memories.push(memory);
                ptr::write_bytes(memory.cast::<u8>(), 0, bytes);

                if argument_properties.type_ == ZE_GRAPH_ARGUMENT_TYPE_INPUT {
                    let input = inputs.get(input_index).ok_or(-4)?;
                    if input.len().checked_mul(mem::size_of::<f32>()) != Some(bytes) {
                        return Err(-4);
                    }
                    ptr::copy_nonoverlapping(input.as_ptr().cast::<u8>(), memory.cast(), bytes);
                    input_index += 1;
                } else if fixture.output.is_null() {
                    fixture.output = memory.cast::<f32>();
                    fixture.output_elements = elements;
                }
                chk(pfn_set(fixture.graph, argument, memory))?;
            }
            if input_index != inputs.len() || fixture.output.is_null() {
                return Err(-4);
            }

            let mut queue_desc: ze_command_queue_desc_t = mem::zeroed();
            queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
            queue_desc.ordinal = 0;
            queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
            chk((l.cmd_queue_create)(
                fixture.context,
                dev.handle,
                &queue_desc,
                &mut fixture.queue,
            ))?;

            let mut list_desc: ze_command_list_desc_t = mem::zeroed();
            list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
            list_desc.commandQueueGroupOrdinal = 0;
            let mut initialize_list = ptr::null_mut();
            chk((l.cmd_list_create)(
                fixture.context,
                dev.handle,
                &list_desc,
                &mut initialize_list,
            ))?;
            let initialize_result = (|| {
                chk(pfn_init(
                    initialize_list,
                    fixture.graph,
                    ptr::null_mut(),
                    0,
                    ptr::null_mut(),
                ))?;
                chk((l.cmd_list_close)(initialize_list))?;
                chk((l.cmd_queue_exec)(
                    fixture.queue,
                    1,
                    &mut initialize_list,
                    ptr::null_mut(),
                ))?;
                chk((l.cmd_queue_sync)(fixture.queue, 2_000_000_000))
            })();
            (l.cmd_list_destroy)(initialize_list);
            initialize_result?;

            chk((l.cmd_list_create)(
                fixture.context,
                dev.handle,
                &list_desc,
                &mut fixture.execute_list,
            ))?;
            chk(pfn_exec(
                fixture.execute_list,
                fixture.graph,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
            ))?;
            chk((l.cmd_list_close)(fixture.execute_list))?;
            Ok(fixture)
        }
    }

    pub(super) fn execute(&mut self, timeout_ns: u64) -> Result<(), i32> {
        if self.failed {
            return Err(-6);
        }
        // SAFETY: the closed list and queue remain owned and idle after each successful sync.
        let result = unsafe {
            chk((self.l.cmd_queue_exec)(
                self.queue,
                1,
                &mut self.execute_list,
                ptr::null_mut(),
            ))
            .and_then(|()| chk((self.l.cmd_queue_sync)(self.queue, timeout_ns)))
        };
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    pub(super) fn copy_output(&self, output: &mut [f32]) -> Result<usize, i32> {
        if self.failed || self.output.is_null() || output.len() < self.output_elements {
            return Err(-7);
        }
        // SAFETY: synchronization precedes this read and the output allocation remains live.
        unsafe {
            ptr::copy_nonoverlapping(self.output, output.as_mut_ptr(), self.output_elements);
        }
        Ok(self.output_elements)
    }
}

impl Drop for PersistentNativeGraph {
    fn drop(&mut self) {
        // SAFETY: each non-null handle is uniquely owned and destroyed in dependency order.
        unsafe {
            if !self.execute_list.is_null() {
                (self.l.cmd_list_destroy)(self.execute_list);
            }
            if !self.queue.is_null() {
                (self.l.cmd_queue_destroy)(self.queue);
            }
            for &memory in &self.memories {
                (self.l.mem_free)(self.context, memory);
            }
            if !self.graph.is_null() && !self.graph_table.is_null() {
                if let Some(destroy) = (*self.graph_table).pfnDestroy {
                    destroy(self.graph);
                }
            }
            if !self.context.is_null() {
                (self.l.context_destroy)(self.context);
            }
        }
    }
}
