//! Zero-copy dispatch for a compiled NPU program (one matmul, or a fused subgraph): a per-blob
//! graph plus a per-node recorded execute list. The program has **N inputs and N outputs**; all
//! weights are baked into the blob as compile-time `Constant`s, so they are neither bound nor
//! copied. A single matmul is just the `1 input, 1 output` case (its input is the activation); a
//! fused subgraph has one input per external boundary tensor and one output per node whose value
//! leaves the subgraph (usually one — e.g. an FFN `down` — but several for a fan-out like Q/K/V).
//!
//! Two caches, both lock-free:
//! - **[`GRAPHS`]** — one Level Zero graph per distinct blob (keyed by the blob's leaked address).
//!   `pfnCreate` re-parses the blob and `pfnAppendGraphInitialize` runs once per blob.
//! - **[`LISTS`]** — one recorded execute command list per program *instance*, keyed by a hash of
//!   its **full binding set** (the blob plus every operand address). The list bakes those pointers
//!   in at record time and is replayed verbatim, so the key must be the exact set it was recorded
//!   with — keying on only the output address aliases distinct instances that ggml's gallocr landed
//!   on a recycled address (a stale binding → NPU MMU fault). Addresses are stable across tokens, so
//!   the common case is still a hit + verbatim replay; a shifted operand simply re-records.
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
use super::profile;
use super::ze;
use super::ze::raw::*;
use super::NpuHostAlloc;
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
    outs: Vec<ArgInfo>,   // the N outputs, in blob argument order
}
// SAFETY: immutable after build; the Level Zero handle is shared read-only.
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

/// How one operand is fed. `bounce == None` → the operand's own NPU pointer was bound directly
/// (zero copy). `Some(buf)` → a staging buffer, NPU-mapped via [`NpuHostAlloc`] and **owned** (freed
/// with the [`NodeList`] rather than leaked as a raw pointer), through which `elems` f32 are copied
/// each dispatch (the operand is in foreign, non-NPU memory). Its address is bound into the recorded
/// execute list, so the allocation must outlive the list — which it does, being owned by it.
struct Bind {
    bounce: Option<Vec<u8, NpuHostAlloc>>,
    elems: usize,
}

/// A per-instance recorded execute list. Operand pointers are baked in at record time (binding
/// snapshot), so it is replayed verbatim across tokens.
struct NodeList {
    exec: ze_command_list_handle_t,
    inputs: Vec<Bind>,
    outs: Vec<Bind>,
}
// SAFETY: immutable after build; compute is single-threaded per backend.
unsafe impl Send for NodeList {}
unsafe impl Sync for NodeList {}

static GRAPHS: Cache<usize, Option<&'static Graph>> = Cache::new();
static LISTS: Cache<u64, Option<&'static NodeList>> = Cache::new();

/// FNV-1a fold of one 64-bit word (for the binding-set key below).
fn mix(h: u64, v: u64) -> u64 {
    let mut h = h ^ v;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    h
}

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

    // Classify args: inputs then outputs, each in blob argument order. FP32 device I/O.
    let mut inputs: Vec<ArgInfo> = Vec::new();
    let mut outs: Vec<ArgInfo> = Vec::new();
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
        } else {
            outs.push(info);
        }
    }
    if inputs.is_empty() || outs.is_empty() {
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

    Some(Box::leak(Box::new(Graph { handle, inputs, outs })))
}

fn node_list(
    g: &'static Graph,
    blob_addr: usize,
    inputs: &[&Operand],
    outs: &[&Operand],
) -> Option<&'static NodeList> {
    if inputs.len() != g.inputs.len() || outs.len() != g.outs.len() || outs.is_empty() {
        return None;
    }
    // Key by the FULL binding set (blob + every operand address), NOT just (first output, blob).
    // The recorded execute list bakes each operand pointer in at record time and is replayed
    // verbatim, so the key must identify the exact set of pointers it was recorded with. Keying on
    // only the output aliases distinct instances: ggml's gallocr recycles one address across
    // non-overlapping tensors, and weightless glue subgraphs (norm×weight, residual-add, silu) share
    // ONE blob across every layer (no baked weight to distinguish them) — so two different instances
    // can collide on (output_ptr, blob) and the second would replay the first's now-stale/foreign
    // input bindings. The NPU then reads an unmapped address → MMU translation fault → device hang.
    // Folding all operand pointers makes each distinct binding its own list, and self-heals if an
    // operand's address shifts across tokens (new key → re-record against the live pointer).
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h = mix(h, blob_addr as u64);
    for op in inputs {
        h = mix(h, op.ptr as u64);
    }
    for op in outs {
        h = mix(h, op.ptr as u64);
    }
    LISTS.get_or_insert_with(h, || unsafe { build_node_list(g, inputs, outs) })
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
        Some(Bind { bounce: None, elems: 0 })
    } else {
        let bytes = arg.elems * 4;
        // NPU-mapped staging buffer, owned via `NpuHostAlloc` (no leaked raw pointer). `try_reserve`
        // so a missing runtime / OOM returns None rather than aborting.
        let mut buf = Vec::<u8, NpuHostAlloc>::new_in(NpuHostAlloc);
        buf.try_reserve_exact(bytes).ok()?;
        ptr::write_bytes(buf.as_mut_ptr(), 0, bytes);
        buf.set_len(bytes);
        if pfn_set(handle, arg.index, buf.as_ptr() as *const c_void) != ZE_RESULT_SUCCESS {
            return None;
        }
        Some(Bind { bounce: Some(buf), elems: min(op.elems, arg.elems) })
    }
}

unsafe fn build_node_list(g: &Graph, inputs: &[&Operand], outs: &[&Operand]) -> Option<&'static NodeList> {
    let l = ze::lib()?;
    let rt = ze::runtime()?;
    let ddi = rt.ddi;
    let pfn_set = (*ddi).pfnSetArgumentValue?;
    let pfn_exec = (*ddi).pfnAppendGraphExecute?;

    let mut binds: Vec<Bind> = Vec::with_capacity(inputs.len());
    for (i, op) in inputs.iter().enumerate() {
        binds.push(bind_arg(pfn_set, g.handle, g.inputs[i], op)?);
    }
    let mut out_binds: Vec<Bind> = Vec::with_capacity(outs.len());
    for (i, op) in outs.iter().enumerate() {
        out_binds.push(bind_arg(pfn_set, g.handle, g.outs[i], op)?);
    }

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
    Some(Box::leak(Box::new(NodeList { exec, inputs: binds, outs: out_binds })))
}

/// Run the program baked into `blob` (a matmul or a fused subgraph): reuse (or build) its graph +
/// this instance's recorded list, stage any foreign inputs, replay the list, wait, and copy back a
/// foreign output. `inputs` must match the blob's input arg order; NPU-resident operands are bound
/// directly (no copy), foreign ones staged through bounces.
pub(super) fn run(blob: &'static [u8], inputs: &[&Operand], outs: &[&Operand]) -> bool {
    let Some(g) = graph(blob) else {
        profile::note("[npu] run FAIL: graph(blob) None\n");
        return false;
    };
    let Some(node) = node_list(g, blob.as_ptr() as usize, inputs, outs) else {
        profile::note(&alloc::format!(
            "[npu] run FAIL: node_list None — inputs {} (g.inputs {}) outs {} (g.outs {})\n",
            inputs.len(),
            g.inputs.len(),
            outs.len(),
            g.outs.len(),
        ));
        return false;
    };
    let (Some(l), Some(rt)) = (ze::lib(), ze::runtime()) else {
        return false;
    };
    let prof = profile::enabled();
    unsafe {
        let t0 = if prof { profile::now_ns() } else { 0 };
        // Stage foreign inputs (owned inputs are already bound to their own memory).
        let mut bytes_in = 0u64;
        for (bind, op) in node.inputs.iter().zip(inputs.iter()) {
            stage_in(bind, op);
            if prof && bind.bounce.is_some() {
                bytes_in += bind.elems as u64 * 4;
            }
        }
        let t1 = if prof { profile::now_ns() } else { 0 };
        // Replay the pre-recorded execute list (async submit), then wait for completion. Split so
        // the profiler can separate host submit cost from the NPU round-trip (the ~fixed floor).
        let mut list = node.exec;
        if (l.cmd_queue_exec)(rt.queue, 1, &mut list, ptr::null_mut()) != ZE_RESULT_SUCCESS {
            profile::note("[npu] run FAIL: cmd_queue_exec\n");
            return false;
        }
        let tmid = if prof { profile::now_ns() } else { 0 };
        if (l.cmd_queue_sync)(rt.queue, u64::MAX) != ZE_RESULT_SUCCESS {
            profile::note("[npu] run FAIL: cmd_queue_sync\n");
            return false;
        }
        let t2 = if prof { profile::now_ns() } else { 0 };
        // Copy foreign outputs back (owned outputs were written in place — zero copy).
        let mut bytes_out = 0u64;
        for (bind, op) in node.outs.iter().zip(outs.iter()) {
            if let Some(buf) = &bind.bounce {
                let nout = min(op.elems, bind.elems);
                ptr::copy_nonoverlapping(buf.as_ptr() as *const f32, op.ptr as *mut f32, nout);
                if prof {
                    bytes_out += nout as u64 * 4;
                }
            }
        }
        if prof {
            let t3 = profile::now_ns();
            let owned =
                node.inputs.iter().chain(node.outs.iter()).filter(|b| b.bounce.is_none()).count() as u64;
            let bounced = (node.inputs.len() + node.outs.len()) as u64 - owned;
            profile::record(t0, t1, tmid, t2, t3, owned, bounced, bytes_in, bytes_out);
        }
    }
    true
}

/// Copy a foreign input's current data into its bounce buffer. No-op for a directly-bound operand.
/// The bounce is NPU-mapped memory whose address is bound into the graph; each dispatch overwrites
/// its contents in place (mutated through the bound pointer, hence the `as *mut`).
unsafe fn stage_in(bind: &Bind, op: &Operand) {
    if let Some(buf) = &bind.bounce {
        ptr::copy_nonoverlapping(op.ptr as *const f32, buf.as_ptr() as *mut f32, bind.elems);
    }
}
