//! Lightweight graph-split tracing for backend bring-up and placement analysis.
//!
//! `GGML_NPU_TRACE=1` emits one line per scheduler split and node to stderr. The format is
//! intentionally stable and line-oriented so runs can be checked into `benchmarks/` and compared
//! without linking a logging framework into this `no_std` cdylib.

use alloc::format;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::ffi::*;
use crate::host::{ConfigService, DiagnosticService, HostEnvironment};

fn enabled() -> bool {
    crate::host::builtin()
        .config()
        .is_nonempty(c"GGML_NPU_TRACE")
}

fn emit(line: &str) {
    crate::host::builtin()
        .diagnostics()
        .write_stderr(line.as_bytes());
}

/// Report the production quant-transcode kernel once when normal tracing is enabled.
pub(crate) fn quant_kernel(name: &str) {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if enabled() && !REPORTED.swap(true, Ordering::Relaxed) {
        emit(&format!("[ggml-npu trace] quant-kernel={name}\n"));
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
