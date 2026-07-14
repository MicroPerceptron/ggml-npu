//! Opt-in Level Zero compatibility / differential-oracle session (`level-zero-compat` only).
//!
//! This is **not** the default runtime — that lives in [`super::session`]. Keep this module only
//! until the retire-Level-Zero milestone closes; do not grow new production paths here.
//!
//! Zero-copy dispatch for a compiled NPU program (one matmul, or a fused subgraph): a per-blob
//! graph plus a per-node recorded execute list. The program has **N inputs and N outputs**; all
//! weights are baked into the blob as compile-time `Constant`s, so they are neither bound nor
//! copied. A single matmul is just the `1 input, 1 output` case (its input is the activation); a
//! fused subgraph has one input per external boundary tensor and one output per node whose value
//! leaves the subgraph (usually one — e.g. an FFN `down` — but several for a fan-out like Q/K/V).
//!
//! Two caches, both lock-free:
//! - **[`CONST_GRAPHS`] / [`SG_GRAPHS`]** — one Level Zero graph per distinct *program*, keyed by
//!   the same semantic identity that names the compiled artifact: `(weight ptr, M)` for a single
//!   baked matmul, the exact [`sg_semantic_key`](super::compile::sg_semantic_key) for a fused
//!   subgraph. The native blob is a **transient** compile artifact: [`build_graph`] parses it into
//!   the driver-owned graph (`pfnCreate` then `pfnAppendGraphInitialize`, once), after which the
//!   blob bytes are **freed** — only the graph is kept resident, so a program's weights are never
//!   also held as a second, redundant copy of blob bytes in RAM.
//! - **[`LISTS`]** — one recorded execute command list per program *instance*, keyed by a hash of
//!   its **full binding set** (the program's graph identity plus every operand address). The list
//!   bakes those pointers in at record time and is replayed verbatim, so the key must be the exact
//!   set it was recorded with — keying on only the output address aliases distinct instances that
//!   ggml's gallocr landed on a recycled address (a stale binding → NPU MMU fault). Addresses are
//!   stable across tokens, so the common case is still a hit + verbatim replay; a shifted operand
//!   simply re-records.
//!
//! Bindings are the operands' own `tensor->data` pointers when those live in NPU memory (our
//! buffer type) — bound **zero copy**. A foreign operand (a CPU-produced activation, or an output
//! consumed on CPU) is staged through a small bounce buffer. Single-threaded compute per backend;
//! FP32 device I/O at the boundary.
#![allow(dead_code)] // unused when Direct `session` is also compiled; retained as L0 oracle

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cmp::min;
use core::ffi::c_void;
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicBool, Ordering};

use super::cache::Cache;
use super::profile;
use super::ze;
use super::ze::raw::*;
use super::NpuHostAlloc;
use crate::host::{ConfigService, HostEnvironment};
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

/// One built graph per distinct program, keyed by the artifact's semantic identity — so the
/// transient compile blob is dropped after [`build_graph`] rather than kept resident. `(weight
/// ptr, M)` for a single baked matmul; the structural signature for a fused subgraph. Model-leaf
/// weight pointers are stable (never recycled by gallocr, unlike activation buffers), so these keys
/// are safe from the address-reuse aliasing that [`LISTS`] guards against.
static CONST_GRAPHS: Cache<(u64, usize, i64), Option<&'static Graph>> = Cache::new(256);
static SG_GRAPHS: Cache<(u64, crate::identity::ExactSemanticKey), Option<&'static Graph>> =
    Cache::new(256);

const MAX_INPUT_BINDINGS: usize = 64;
const MAX_OUTPUT_BINDINGS: usize = 8;

/// Exact recorded-command-list identity. A hash alone is unsafe here: a collision can replay a
/// command list containing unrelated pointers and fault the NPU MMU. Fixed arrays keep the key
/// `Copy` for the lock-free cache while covering the graph recognizer's explicit limits.
#[derive(Clone, Copy, PartialEq, Eq)]
struct BindingKey {
    program: usize,
    generation: u64,
    n_inputs: u8,
    n_outputs: u8,
    inputs: [usize; MAX_INPUT_BINDINGS],
    outputs: [usize; MAX_OUTPUT_BINDINGS],
}

static LISTS: Cache<BindingKey, Option<&'static NodeList>> = Cache::new(2048);
static PENDING: AtomicBool = AtomicBool::new(false);

pub(super) fn async_enabled() -> bool {
    crate::host::builtin().config().is_set(c"GGML_NPU_ASYNC")
}

pub(super) fn synchronize() -> bool {
    if !PENDING.swap(false, Ordering::AcqRel) {
        return true;
    }
    let (Some(l), Some(rt)) = (ze::lib(), ze::runtime()) else {
        return false;
    };
    unsafe { (l.cmd_queue_sync)(rt.queue, u64::MAX) == ZE_RESULT_SUCCESS }
}

/// Build a program's graph on a cache miss: run the caller's `compile` to get a **transient** native
/// blob, parse it into the driver-owned graph via [`build_graph`], and drop the blob (freed at the
/// end of this call — only the graph is kept). `None` if compilation or graph creation failed
/// (cached by the caller as `None`, not retried).
fn build_from(compile: impl FnOnce() -> Option<Vec<u8>>) -> Option<&'static Graph> {
    let blob = compile()?;
    unsafe { build_graph(&blob) }
}

unsafe fn build_graph(blob: &[u8]) -> Option<&'static Graph> {
    let l = ze::lib()?;
    let rt = ze::runtime()?;
    let g = rt.ddi;
    let pfn_create = (*g).pfnCreate?;
    let pfn_props = (*g).pfnGetProperties?;
    let pfn_arg = (*g).pfnGetArgumentProperties?;
    let pfn_init = (*g).pfnAppendGraphInitialize?;
    let pfn_destroy = (*g).pfnDestroy;

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
        if let Some(destroy) = pfn_destroy {
            destroy(handle);
        }
        return None;
    }

    // Classify args: inputs then outputs, each in blob argument order. FP32 device I/O.
    let mut inputs: Vec<ArgInfo> = Vec::new();
    let mut outs: Vec<ArgInfo> = Vec::new();
    for i in 0..gp.numGraphArgs {
        let mut ap: ze_graph_argument_properties_t = mem::zeroed();
        ap.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        if pfn_arg(handle, i, &mut ap) != ZE_RESULT_SUCCESS {
            if let Some(destroy) = pfn_destroy {
                destroy(handle);
            }
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
        if let Some(destroy) = pfn_destroy {
            destroy(handle);
        }
        return None;
    }

    // Initialize the graph once — independent of argument binding (validated).
    let mut ld: ze_command_list_desc_t = mem::zeroed();
    ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    ld.commandQueueGroupOrdinal = 0;
    let mut init_list: ze_command_list_handle_t = ptr::null_mut();
    if (l.cmd_list_create)(rt.context, rt.device, &ld, &mut init_list) != ZE_RESULT_SUCCESS {
        if let Some(destroy) = pfn_destroy {
            destroy(handle);
        }
        return None;
    }
    let ok = pfn_init(init_list, handle, ptr::null_mut(), 0, ptr::null_mut()) == ZE_RESULT_SUCCESS
        && (l.cmd_list_close)(init_list) == ZE_RESULT_SUCCESS
        && (l.cmd_queue_exec)(rt.queue, 1, &mut init_list, ptr::null_mut()) == ZE_RESULT_SUCCESS
        && (l.cmd_queue_sync)(rt.queue, u64::MAX) == ZE_RESULT_SUCCESS;
    (l.cmd_list_destroy)(init_list);
    if !ok {
        if let Some(destroy) = pfn_destroy {
            destroy(handle);
        }
        return None;
    }

    Some(Box::leak(Box::new(Graph {
        handle,
        inputs,
        outs,
    })))
}

fn node_list(
    g: &'static Graph,
    program_addr: usize,
    inputs: &[&Operand],
    outs: &[&Operand],
) -> Option<&'static NodeList> {
    if inputs.len() != g.inputs.len() || outs.len() != g.outs.len() || outs.is_empty() {
        return None;
    }
    if inputs.len() > MAX_INPUT_BINDINGS || outs.len() > MAX_OUTPUT_BINDINGS {
        return None;
    }
    // Key by the FULL binding set (program identity + every operand address), NOT just (first
    // output, program). The recorded execute list bakes each operand pointer in at record time and
    // is replayed verbatim, so the key must identify the exact set of pointers it was recorded with.
    // Keying on only the output aliases distinct instances: ggml's gallocr recycles one address
    // across non-overlapping tensors, and weightless glue subgraphs (norm×weight, residual-add,
    // silu) share ONE program/graph across every layer (no baked weight to distinguish them) — so
    // two different instances can collide on (output_ptr, program) and the second would replay the
    // first's now-stale/foreign input bindings. The NPU then reads an unmapped address → MMU
    // translation fault → device hang. Folding all operand pointers makes each distinct binding its
    // own list, and self-heals if an operand's address shifts across tokens (new key → re-record
    // against the live pointer). The program address is the built graph's own (stable, leaked)
    // address — the same discriminator role the transient blob's address used to play.
    let mut key = BindingKey {
        program: program_addr,
        generation: crate::backend::buffer_generation(),
        n_inputs: inputs.len() as u8,
        n_outputs: outs.len() as u8,
        inputs: [0; MAX_INPUT_BINDINGS],
        outputs: [0; MAX_OUTPUT_BINDINGS],
    };
    for (dst, op) in key.inputs.iter_mut().zip(inputs) {
        *dst = op.ptr as usize;
    }
    for (dst, op) in key.outputs.iter_mut().zip(outs) {
        *dst = op.ptr as usize;
    }
    LISTS.try_get_or_insert_with(key, || unsafe { build_node_list(g, inputs, outs) })?
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
        let result = pfn_set(handle, arg.index, op.ptr);
        if result != ZE_RESULT_SUCCESS {
            profile::note(&alloc::format!(
                "[npu] bind FAIL: arg={} owned ptr={:p} op_elems={} graph_elems={} ze={}\n",
                arg.index,
                op.ptr,
                op.elems,
                arg.elems,
                result
            ));
            return None;
        }
        Some(Bind {
            bounce: None,
            elems: 0,
        })
    } else {
        let bytes = arg.elems * 4;
        // NPU-mapped staging buffer, owned via `NpuHostAlloc` (no leaked raw pointer). `try_reserve`
        // so a missing runtime / OOM returns None rather than aborting.
        let mut buf = Vec::<u8, NpuHostAlloc>::new_in(NpuHostAlloc);
        buf.try_reserve_exact(bytes).ok()?;
        ptr::write_bytes(buf.as_mut_ptr(), 0, bytes);
        buf.set_len(bytes);
        let result = pfn_set(handle, arg.index, buf.as_ptr() as *const c_void);
        if result != ZE_RESULT_SUCCESS {
            profile::note(&alloc::format!(
                "[npu] bind FAIL: arg={} bounce={:p} op_elems={} graph_elems={} ze={}\n",
                arg.index,
                buf.as_ptr(),
                op.elems,
                arg.elems,
                result
            ));
            return None;
        }
        profile::event(crate::metrics::MetricEvent::StagingAllocation {
            allocation: buf.as_ptr() as u64,
            bytes: bytes as u64,
        });
        Some(Bind {
            bounce: Some(buf),
            elems: min(op.elems, arg.elems),
        })
    }
}

unsafe fn build_node_list(
    g: &Graph,
    inputs: &[&Operand],
    outs: &[&Operand],
) -> Option<&'static NodeList> {
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
    let create_result = (l.cmd_list_create)(rt.context, rt.device, &ld, &mut exec);
    if create_result != ZE_RESULT_SUCCESS {
        profile::note(&alloc::format!(
            "[npu] command-list create FAIL: ze={create_result}\n"
        ));
        return None;
    }
    let exec_result = pfn_exec(
        exec,
        g.handle,
        ptr::null_mut(),
        ptr::null_mut(),
        0,
        ptr::null_mut(),
    );
    let close_result = if exec_result == ZE_RESULT_SUCCESS {
        (l.cmd_list_close)(exec)
    } else {
        ZE_RESULT_SUCCESS
    };
    if exec_result != ZE_RESULT_SUCCESS || close_result != ZE_RESULT_SUCCESS {
        profile::note(&alloc::format!(
            "[npu] command-list record FAIL: execute ze={exec_result}, close ze={close_result}\n"
        ));
        (l.cmd_list_destroy)(exec);
        return None;
    }
    Some(Box::leak(Box::new(NodeList {
        exec,
        inputs: binds,
        outs: out_binds,
    })))
}

/// Run a single **baked-Constant matmul** program: reuse (or build) its graph — keyed by the weight
/// identity + M, so its transient blob is freed after [`build_graph`] — then dispatch this instance.
/// `compile` produces the native blob on a cold miss (see [`super::compile::compile_const_blob`]).
pub(super) fn run_const(
    weight_ptr: usize,
    m: i64,
    inputs: &[&Operand],
    outs: &[&Operand],
    compile: impl FnOnce() -> Option<Vec<u8>>,
) -> bool {
    let key = (crate::backend::buffer_generation(), weight_ptr, m);
    let Some(g) = CONST_GRAPHS
        .try_get_or_insert_with(key, || build_from(compile))
        .flatten()
    else {
        profile::note("[npu] run FAIL: const graph build None\n");
        return false;
    };
    dispatch(g, inputs, outs)
}

/// Run a **fused subgraph** program: reuse (or build) its graph — keyed by the structural
/// [`sg_semantic_key`](super::compile::sg_semantic_key), so weightless glue subgraphs share one graph
/// across layers and the transient blob is freed after build — then dispatch this instance.
pub(super) fn run_subgraph(
    semantic_key: crate::identity::ExactSemanticKey,
    inputs: &[&Operand],
    outs: &[&Operand],
    compile: impl FnOnce() -> Option<Vec<u8>>,
) -> bool {
    let key = (crate::backend::buffer_generation(), semantic_key);
    let Some(g) = SG_GRAPHS
        .try_get_or_insert_with(key, || build_from(compile))
        .flatten()
    else {
        profile::note("[npu] run FAIL: subgraph graph build None\n");
        return false;
    };
    dispatch(g, inputs, outs)
}

/// Dispatch one instance of an already-built program `g`: reuse (or record) this instance's execute
/// list, stage any foreign inputs, replay the list, wait, and copy back a foreign output. `inputs`
/// must match the program's input arg order; NPU-resident operands are bound directly (no copy),
/// foreign ones staged through bounces. The `node_list` key uses the graph's own (stable, leaked)
/// address as the per-program discriminator, replacing the transient blob's address.
fn dispatch(g: &'static Graph, inputs: &[&Operand], outs: &[&Operand]) -> bool {
    let Some(node) = node_list(g, g as *const Graph as usize, inputs, outs) else {
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
    let prof = profile::enabled() || crate::timeline::enabled();
    unsafe {
        let has_bounce = node
            .inputs
            .iter()
            .chain(node.outs.iter())
            .any(|b| b.bounce.is_some());
        // A host-visible staging buffer cannot be overwritten/read while an earlier submission is
        // in flight. Direct NPU-owned bindings can remain queued in order without this barrier.
        if has_bounce && !synchronize() {
            return false;
        }
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
        if async_enabled() && !has_bounce {
            PENDING.store(true, Ordering::Release);
            if profile::enabled() {
                profile::note("[npu] async queued (device/copy timing deferred to synchronize)\n");
            }
            return true;
        }
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
            let owned = node
                .inputs
                .iter()
                .chain(node.outs.iter())
                .filter(|b| b.bounce.is_none())
                .count() as u64;
            let bounced = (node.inputs.len() + node.outs.len()) as u64 - owned;
            if crate::timeline::enabled() {
                use crate::timeline_ring::{Domain, Kind};
                // Host stage/submit/copy → CPU; queue sync wait → NPU.
                crate::timeline::span(Domain::Cpu, Kind::StageIn, t0, t1, bytes_in);
                crate::timeline::span(Domain::Cpu, Kind::Submit, t1, tmid, 0);
                crate::timeline::span(Domain::Npu, Kind::Wait, tmid, t2, 0);
                crate::timeline::span(Domain::Cpu, Kind::CopyOut, t2, t3, bytes_out);
                if bytes_in > 0 {
                    crate::timeline::instant(Domain::Cpu, Kind::MemLoad, bytes_in);
                }
                if bytes_out > 0 {
                    crate::timeline::instant(Domain::Cpu, Kind::MemLoad, bytes_out);
                }
            }
            if profile::enabled() {
                profile::record(
                    t0,
                    t1,
                    tmid,
                    t2,
                    t3,
                    owned,
                    0,
                    0,
                    bounced,
                    bytes_in.saturating_add(bytes_out),
                    bytes_in,
                    bytes_out,
                );
            }
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
