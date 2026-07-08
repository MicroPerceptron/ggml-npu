//! Zero-copy dispatch: a per-shape graph plus a per-node recorded execute list.
//!
//! Two caches, both lock-free:
//! - **[`GRAPHS`]** — one Level Zero graph per `(M,K,N)` shape (`pfnCreate` re-parses the blob and
//!   `pfnAppendGraphInitialize` runs once; both are expensive, so a shape is built once and shared
//!   by every node that has it).
//! - **[`LISTS`]** — one recorded execute command list per graph *node*, keyed by its operand
//!   pointer-tuple `(activation, weight, output)`. On first sight of a node we bind its operands
//!   and record the list; thereafter a dispatch is just execute + sync.
//!
//! **The bindings are the operands' own `tensor->data` pointers** whenever those live in NPU
//! memory (our buffer type) — so the large weight, and usually the output, are bound with **zero
//! copy**, once, and reused for every token (a node's operand addresses are stable across tokens;
//! the recorded list snapshots them — validated by the `ze_{latebind,multilist,subrange}` spikes).
//! A foreign operand (e.g. a CPU-produced activation) is staged through a small bounce buffer.
//!
//! Assumes single-threaded compute per backend (ggml calls `graph_compute` serially) and FP32
//! device I/O (the OpenVINO f32 MatMul).

use alloc::boxed::Box;
use core::cmp::min;
use core::ffi::c_void;
use core::mem;
use core::ptr;

use super::cache::{Cache, PtrKey, ShapeCache};
use super::ze;
use super::ze::raw::*;
use crate::ops::Operand;

/// One graph argument: its index in the blob and element count (FP32 device I/O).
#[derive(Clone, Copy)]
struct ArgInfo {
    index: u32,
    elems: usize,
}

/// A per-shape compiled graph: created + initialized once, shared by every node of this shape.
struct Graph {
    handle: ze_graph_handle_t,
    act: ArgInfo, // first input  (activation)
    wgt: ArgInfo, // second input (weight)
    out: ArgInfo, // output
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

/// A per-node recorded execute list. Operand pointers are baked in at record time (binding
/// snapshot), so it is replayed verbatim across tokens.
struct NodeList {
    exec: ze_command_list_handle_t,
    act: Bind,
    wgt: Bind,
    out: Bind,
}
// SAFETY: immutable after build; compute is single-threaded per backend.
unsafe impl Send for NodeList {}
unsafe impl Sync for NodeList {}

static GRAPHS: ShapeCache<Option<&'static Graph>> = ShapeCache::new();
static LISTS: Cache<PtrKey, Option<&'static NodeList>> = Cache::new();

fn graph(m: i64, k: i64, n: i64) -> Option<&'static Graph> {
    GRAPHS.get_or_insert_with((m, k, n), || {
        let blob = super::compile::get_or_compile_matmul(m, k, n)?;
        unsafe { build_graph(blob) }
    })
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

    // Classify args: inputs in blob order [activation, weight]; one output. FP32 device I/O only.
    let mut inputs = [ArgInfo { index: 0, elems: 0 }; 2];
    let mut n_in = 0usize;
    let mut out: Option<ArgInfo> = None;
    for i in 0..gp.numGraphArgs {
        let mut ap: ze_graph_argument_properties_t = mem::zeroed();
        ap.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        if pfn_arg(handle, i, &mut ap) != ZE_RESULT_SUCCESS {
            return None;
        }
        if ap.devicePrecision != ZE_GRAPH_ARGUMENT_PRECISION_FP32 {
            return None; // we only compile + copy f32
        }
        let mut elems = 1usize;
        for d in 0..(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE as usize) {
            if ap.dims[d] > 0 {
                elems *= ap.dims[d] as usize;
            }
        }
        let info = ArgInfo { index: i, elems };
        if ap.type_ == ZE_GRAPH_ARGUMENT_TYPE_INPUT {
            if n_in >= 2 {
                return None;
            }
            inputs[n_in] = info;
            n_in += 1;
        } else {
            if out.is_some() {
                return None;
            }
            out = Some(info);
        }
    }
    if n_in != 2 {
        return None;
    }
    let out = out?;

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

    Some(Box::leak(Box::new(Graph { handle, act: inputs[0], wgt: inputs[1], out })))
}

fn node_list(g: &'static Graph, act: &Operand, wgt: &Operand, out: &Operand) -> Option<&'static NodeList> {
    let key: PtrKey = (act.ptr as usize, wgt.ptr as usize, out.ptr as usize);
    LISTS.get_or_insert_with(key, || unsafe { build_node_list(g, act, wgt, out) })
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

unsafe fn build_node_list(g: &Graph, act: &Operand, wgt: &Operand, out: &Operand) -> Option<&'static NodeList> {
    let l = ze::lib()?;
    let rt = ze::runtime()?;
    let ddi = rt.ddi;
    let pfn_set = (*ddi).pfnSetArgumentValue?;
    let pfn_exec = (*ddi).pfnAppendGraphExecute?;

    let act = bind_arg(pfn_set, g.handle, g.act, act)?;
    let wgt = bind_arg(pfn_set, g.handle, g.wgt, wgt)?;
    let out = bind_arg(pfn_set, g.handle, g.out, out)?;

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
    Some(Box::leak(Box::new(NodeList { exec, act, wgt, out })))
}

/// Run a MatMul of shape `(M,K,N)`, reusing (or building) its graph + this node's recorded list.
/// Operands in NPU memory are bound directly (no copy); foreign ones are staged through bounces.
pub(super) fn run_matmul(m: i64, k: i64, n: i64, act: &Operand, wgt: &Operand, out: &Operand) -> bool {
    let Some(g) = graph(m, k, n) else {
        return false;
    };
    let Some(node) = node_list(g, act, wgt, out) else {
        return false;
    };
    let (Some(l), Some(rt)) = (ze::lib(), ze::runtime()) else {
        return false;
    };
    unsafe {
        // Stage foreign inputs (owned inputs are already bound to their own memory).
        stage_in(&node.act, act);
        stage_in(&node.wgt, wgt);
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
