//! Lightweight graph-split tracing for backend bring-up and placement analysis.
//!
//! `GGML_NPU_TRACE=1` emits one line per scheduler split and node to stderr. The format is
//! intentionally stable and line-oriented so runs can be checked into `benchmarks/` and compared
//! without linking a logging framework into this `no_std` cdylib.

use alloc::format;
use core::sync::atomic::{AtomicU8, Ordering};

use crate::ffi::*;
use crate::graph::{Dim, GraphIr, TensorRole};
use crate::host::{ConfigService, DiagnosticService, HostEnvironment};

fn enabled() -> bool {
    crate::host::builtin()
        .config()
        .is_nonempty(c"GGML_NPU_TRACE")
}

static PERF_ENABLED: AtomicU8 = AtomicU8::new(0);

/// Machine-readable performance events. This is deliberately separate from the verbose graph
/// trace: `GGML_PERF_METRICS=1` is suitable for controlled benchmarks, while the default hot path
/// pays one cached atomic load and emits nothing.
pub(crate) fn perf_enabled() -> bool {
    let state = PERF_ENABLED.load(Ordering::Acquire);
    if state != 0 {
        return state == 2;
    }
    let on = crate::host::builtin()
        .config()
        .is_nonempty(c"GGML_PERF_METRICS");
    PERF_ENABLED.store(if on { 2 } else { 1 }, Ordering::Release);
    on
}

fn emit(line: &str) {
    crate::host::builtin()
        .diagnostics()
        .write_stderr(line.as_bytes());
}

/// Emit one JSON object with a stable prefix. `fields` must contain comma-prefixed, JSON-safe
/// fields assembled from numeric values, digests, or fixed reason identifiers.
pub(crate) fn perf(event: &str, fields: &str) {
    if perf_enabled() {
        emit(&format!(
            "[ggml-perf] {{\"schema\":1,\"source\":\"ggml-npu\",\"event\":\"{event}\"{fields}}}\n"
        ));
    }
}

fn dim_value(dim: Dim) -> i64 {
    match dim {
        Dim::Static(value) => value,
        Dim::Bounded { max, .. } => max,
    }
}

fn storage_bytes(dtype: ggml_type, dims: [Dim; 4]) -> u64 {
    // SAFETY: graph tensor dtypes originate from live ggml tensors and type traits are static.
    let traits = unsafe { ggml_get_type_traits(dtype) };
    if traits.is_null() {
        return 0;
    }
    let traits = unsafe { &*traits };
    let block = u64::try_from(traits.blck_size).unwrap_or(0);
    let elements = dims.into_iter().fold(1u64, |product, dim| {
        product.saturating_mul(dim_value(dim).max(0) as u64)
    });
    if block == 0 || elements % block != 0 {
        return 0;
    }
    elements
        .checked_div(block)
        .and_then(|blocks| blocks.checked_mul(traits.type_size as u64))
        .unwrap_or(0)
}

/// Emit enough graph structure to reconstruct split/node histograms, operation mix, shape
/// specialization, and source-constant volume without retaining model-specific names.
pub(crate) fn graph(event: &str, graph: &GraphIr, reason: &str) {
    if !perf_enabled() {
        return;
    }
    let key = graph.semantic_key().digest().hex();
    let constant_bytes = graph
        .values
        .iter()
        .filter(|value| value.role == TensorRole::Constant)
        .map(|value| storage_bytes(value.dtype, value.dims))
        .fold(0u64, u64::saturating_add);
    perf(
        event,
        &format!(
            ",\"key\":\"{key}\",\"reason\":\"{reason}\",\"nodes\":{},\"inputs\":{},\"outputs\":{},\"values\":{},\"constant_source_bytes\":{}",
            graph.nodes.len(),
            graph.inputs.len(),
            graph.outputs.len(),
            graph.values.len(),
            constant_bytes,
        ),
    );
    for (index, node) in graph.nodes.iter().enumerate() {
        let output = graph.values.get(node.output as usize);
        let (dtype, dims) =
            output.map_or((0, [Dim::Static(0); 4]), |value| (value.dtype, value.dims));
        perf(
            "graph_node",
            &format!(
                ",\"key\":\"{key}\",\"index\":{index},\"op\":{},\"wire\":{},\"inputs\":{},\"dtype\":{},\"shape\":[{},{},{},{}]",
                node.op,
                GraphIr::wire_op(node).map_or(-1, |wire| wire as i32),
                node.inputs.len(),
                dtype,
                dim_value(dims[0]),
                dim_value(dims[1]),
                dim_value(dims[2]),
                dim_value(dims[3]),
            ),
        );
    }
    for (index, value) in graph.values.iter().enumerate() {
        if value.role != TensorRole::Constant {
            continue;
        }
        perf(
            "graph_constant",
            &format!(
                ",\"key\":\"{key}\",\"index\":{index},\"dtype\":{},\"source_bytes\":{},\"shape\":[{},{},{},{}]",
                value.dtype,
                storage_bytes(value.dtype, value.dims),
                dim_value(value.dims[0]),
                dim_value(value.dims[1]),
                dim_value(value.dims[2]),
                dim_value(value.dims[3]),
            ),
        );
    }
}

/// Emit the shape/stride/storage summary of a scheduler split before recognition.
pub(crate) unsafe fn split(cgraph: *mut ggml_cgraph) {
    if !enabled() || cgraph.is_null() {
        return;
    }
    let n = ggml_graph_n_nodes(cgraph);
    emit(&format!("[ggml-npu trace] split nodes={n}\n"));
    for i in 0..n {
        let t = ggml_graph_node(cgraph, i);
        if t.is_null() {
            emit(&format!("[ggml-npu trace] node={i} null\n"));
            continue;
        }
        let x = &*t;
        let name = core::ffi::CStr::from_ptr(x.name.as_ptr())
            .to_str()
            .unwrap_or("<non-utf8>");
        emit(&format!(
            "[ggml-npu trace] node={i} name={name} op={} dtype={} ne={},{},{},{} nb={},{},{},{} data={:p} owned={}\n",
            x.op,
            x.type_,
            x.ne[0],
            x.ne[1],
            x.ne[2],
            x.ne[3],
            x.nb[0],
            x.nb[1],
            x.nb[2],
            x.nb[3],
            x.data,
            crate::backend::owns_tensor(t),
        ));
    }
}
