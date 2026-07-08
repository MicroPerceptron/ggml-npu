//! Zero-copy dispatch for a compiled NPU program (one matmul, or a fused subgraph): a per-blob
//! graph plus a per-node recorded execute list. The program has **N inputs and one output**; all
//! weights are baked into the blob as compile-time `Constant`s, so they are neither bound nor
//! copied. A single matmul is just the `N == 1` case (its one input is the activation); a fused
//! subgraph (e.g. an FFN) has one input per external boundary tensor.
//!
//! Two caches, both lock-free:
//! - **[`GRAPHS`]** — one Level Zero graph per distinct blob (keyed by the blob's leaked address).
//!   `pfnCreate` re-parses the blob and `pfnAppendGraphInitialize` runs once per blob.
//! - **[`LISTS`]** — one recorded execute command list per program *instance*, keyed by
//!   `(output pointer, blob)`. The output tensor's address uniquely identifies the instance (each
//!   node writes its own dst), and its input/output addresses are stable across tokens, so the
//!   recorded list is replayed verbatim; a dispatch is just execute + sync.
//!
//! Bindings are the operands' own `tensor->data` pointers when those live in NPU memory (our
//! buffer type) — bound **zero copy**. A foreign operand (a CPU-produced activation, or an output
//! consumed on CPU) is staged through a small bounce buffer. Single-threaded compute per backend;
//! FP32 device I/O at the boundary.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cmp::min;
use core::ffi::c_void;
use core::mem;
use core::ptr;

use super::cache::Cache;
use super::ze;
use super::ze::raw::*;
use crate::ops::Operand;

/// One graph argument: its index in the blob and element count (FP32 device I/O).
#[derive(Clone, Copy)]
struct ArgInfo {
    index: u32,
    elems: usize,
}

/// A per-blob compiled graph: created + initialized once, shared by every instance using this blob.
struct Graph {
    handle: ze_graph_handle_t,
    inputs: Vec<ArgInfo>, // the N inputs, in blob argument order; weights are baked, not args
    out: ArgInfo,         // the single output
}
// SAFETY: immutable after build; the Level Zero handle is shared read-only.
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

/// How one operand is fed. `bounce` null → the operand's own NPU pointer was bound directly
/// (zero copy); otherwise `bounce` is a staging buffer through which `elems` f32 are copied each
/// dispatch (the operand is in foreign, non-NPU memory).
#[derive(Clone, Copy)]
struct Bind {
    bounce: *mut c_void,
    elems: usize,
}

/// A per-instance recorded execute list. Operand pointers are baked in at record time (binding
/// snapshot), so it is replayed verbatim across tokens.
struct NodeList {
    exec: ze_command_list_handle_t,
    inputs: Vec<Bind>,
    out: Bind,
}
// SAFETY: immutable after build; compute is single-threaded per backend.
unsafe impl Send for NodeList {}
unsafe impl Sync for NodeList {}

static GRAPHS: Cache<usize, Option<&'static Graph>> = Cache::new();
static LISTS: Cache<(usize, usize), Option<&'static NodeList>> = Cache::new();

fn graph(blob: &'static [u8]) -> Option<&'static Graph> {
    GRAPHS.get_or_insert_with(blob.as_ptr() as usize, || unsafe { build_graph(blob) })
}

unsafe fn build_graph(blob: &[u8]) -> Option<&'static Graph> {
    let l = ze::lib()?;
    let rt = ze::runtime()?;
    let g = rt.ddi;
    let pfn_create = (*g).pfnCreate?;
    let pfn_props = (*g).pfnGetProperties?;
    let pfn_arg = (*g).pfnGetArgumentProperties?;
    let pfn_init = (*g).pfnAppendGraphInitialize?;

    let mut gd: ze_graph_desc_t = mem::zeroed();
    gd.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC;
    gd.format = ZE_GRAPH_FORMAT_NATIVE;
    gd.inputSize = blob.len();
    gd.pInput = blob.as_ptr();
    let mut handle: ze_graph_handle_t = ptr::null_mut();
    if pfn_create(rt.context, rt.device, &gd, &mut handle) != ZE_RESULT_SUCCESS {
        return None;
    }

    let mut gp: ze_graph_properties_t = mem::zeroed();
    gp.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    if pfn_props(handle, &mut gp) != ZE_RESULT_SUCCESS {
        return None;
    }

    // Classify args: inputs in blob order, exactly one output. FP32 device I/O.
    let mut inputs: Vec<ArgInfo> = Vec::new();
    let mut out: Option<ArgInfo> = None;
    for i in 0..gp.numGraphArgs {
        let mut ap: ze_graph_argument_properties_t = mem::zeroed();
        ap.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        if pfn_arg(handle, i, &mut ap) != ZE_RESULT_SUCCESS {
            return None;
        }
        let mut elems = 1usize;
        for d in 0..(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE as usize) {
            if ap.dims[d] > 0 {
                elems *= ap.dims[d] as usize;
            }
        }
        let info = ArgInfo { index: i, elems };
        if ap.type_ == ZE_GRAPH_ARGUMENT_TYPE_INPUT {
            inputs.push(info);
        } else if out.is_some() {
            return None; // exactly one output expected
        } else {
            out = Some(info);
        }
    }
    let out = out?;
    if inputs.is_empty() {
        return None;
    }

    // Initialize the graph once — independent of argument binding (validated).
    let mut ld: ze_command_list_desc_t = mem::zeroed();
    ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    ld.commandQueueGroupOrdinal = 0;
    let mut init_list: ze_command_list_handle_t = ptr::null_mut();
    if (l.cmd_list_create)(rt.context, rt.device, &ld, &mut init_list) != ZE_RESULT_SUCCESS {
        return None;
    }
    let ok = pfn_init(init_list, handle, ptr::null_mut(), 0, ptr::null_mut()) == ZE_RESULT_SUCCESS
        && (l.cmd_list_close)(init_list) == ZE_RESULT_SUCCESS
        && (l.cmd_queue_exec)(rt.queue, 1, &mut init_list, ptr::null_mut()) == ZE_RESULT_SUCCESS
        && (l.cmd_queue_sync)(rt.queue, u64::MAX) == ZE_RESULT_SUCCESS;
    (l.cmd_list_destroy)(init_list);
    if !ok {
        return None;
    }

    Some(Box::leak(Box::new(Graph { handle, inputs, out })))
}

fn node_list(
    g: &'static Graph,
    blob_addr: usize,
    inputs: &[&Operand],
    out: &Operand,
) -> Option<&'static NodeList> {
    if inputs.len() != g.inputs.len() {
        return None;
    }
    // (output pointer, blob) uniquely identifies this instance: each node writes its own dst, and a
    // given output is always produced by the same node with the same inputs.
    let key = (out.ptr as usize, blob_addr);
    LISTS.get_or_insert_with(key, || unsafe { build_node_list(g, inputs, out) })
}

/// Bind one operand: its own NPU pointer directly when owned (zero copy), else a fresh bounce
/// buffer. Returns the [`Bind`] describing how to feed it at dispatch time.
unsafe fn bind_arg(
    pfn_set: unsafe extern "C" fn(ze_graph_handle_t, u32, *const c_void) -> ze_result_t,
    handle: ze_graph_handle_t,
    arg: ArgInfo,
    op: &Operand,
) -> Option<Bind> {
    if op.owned {
        if pfn_set(handle, arg.index, op.ptr) != ZE_RESULT_SUCCESS {
            return None;
        }
        Some(Bind { bounce: ptr::null_mut(), elems: 0 })
    } else {
        let bytes = arg.elems * 4;
        let bounce = ze::alloc_host(bytes);
        if bounce.is_null() {
            return None;
        }
        ptr::write_bytes(bounce as *mut u8, 0, bytes);
        if pfn_set(handle, arg.index, bounce) != ZE_RESULT_SUCCESS {
            return None;
        }
        Some(Bind { bounce, elems: min(op.elems, arg.elems) })
    }
}

unsafe fn build_node_list(g: &Graph, inputs: &[&Operand], out: &Operand) -> Option<&'static NodeList> {
    let l = ze::lib()?;
    let rt = ze::runtime()?;
    let ddi = rt.ddi;
    let pfn_set = (*ddi).pfnSetArgumentValue?;
    let pfn_exec = (*ddi).pfnAppendGraphExecute?;

    let mut binds: Vec<Bind> = Vec::with_capacity(inputs.len());
    for (i, op) in inputs.iter().enumerate() {
        binds.push(bind_arg(pfn_set, g.handle, g.inputs[i], op)?);
    }
    let out_bind = bind_arg(pfn_set, g.handle, g.out, out)?;

    let mut ld: ze_command_list_desc_t = mem::zeroed();
    ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    ld.commandQueueGroupOrdinal = 0;
    let mut exec: ze_command_list_handle_t = ptr::null_mut();
    if (l.cmd_list_create)(rt.context, rt.device, &ld, &mut exec) != ZE_RESULT_SUCCESS {
        return None;
    }
    if pfn_exec(exec, g.handle, ptr::null_mut(), ptr::null_mut(), 0, ptr::null_mut()) != ZE_RESULT_SUCCESS
        || (l.cmd_list_close)(exec) != ZE_RESULT_SUCCESS
    {
        return None;
    }
    Some(Box::leak(Box::new(NodeList { exec, inputs: binds, out: out_bind })))
}

/// Run the program baked into `blob` (a matmul or a fused subgraph): reuse (or build) its graph +
/// this instance's recorded list, stage any foreign inputs, replay the list, wait, and copy back a
/// foreign output. `inputs` must match the blob's input arg order; NPU-resident operands are bound
/// directly (no copy), foreign ones staged through bounces.
pub(super) fn run(blob: &'static [u8], inputs: &[&Operand], out: &Operand) -> bool {
    let Some(g) = graph(blob) else {
        return false;
    };
    let Some(node) = node_list(g, blob.as_ptr() as usize, inputs, out) else {
        return false;
    };
    let (Some(l), Some(rt)) = (ze::lib(), ze::runtime()) else {
        return false;
    };
    unsafe {
        // Stage foreign inputs (owned inputs are already bound to their own memory).
        for (bind, op) in node.inputs.iter().zip(inputs.iter()) {
            stage_in(bind, op);
        }
        // Replay the pre-recorded execute list, then wait for completion.
        let mut list = node.exec;
        if (l.cmd_queue_exec)(rt.queue, 1, &mut list, ptr::null_mut()) != ZE_RESULT_SUCCESS
            || (l.cmd_queue_sync)(rt.queue, u64::MAX) != ZE_RESULT_SUCCESS
        {
            return false;
        }
        // Copy a foreign output back (an owned output was written in place — zero copy).
        if !node.out.bounce.is_null() {
            let nout = min(out.elems, node.out.elems);
            ptr::copy_nonoverlapping(node.out.bounce as *const f32, out.ptr as *mut f32, nout);
        }
    }
    true
}

/// Copy a foreign input's current data into its bounce buffer. No-op for a directly-bound operand.
unsafe fn stage_in(bind: &Bind, op: &Operand) {
    if !bind.bounce.is_null() {
        ptr::copy_nonoverlapping(op.ptr as *const f32, bind.bounce as *mut f32, bind.elems);
    }
}
