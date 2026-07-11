//! Lightweight graph-split tracing for backend bring-up and placement analysis.
//!
//! `GGML_NPU_TRACE=1` emits one line per scheduler split and node to stderr. The format is
//! intentionally stable and line-oriented so runs can be checked into `benchmarks/` and compared
//! without linking a logging framework into this `no_std` cdylib.

use alloc::format;
use core::ffi::{c_char, c_int};

use crate::ffi::*;

extern "C" {
    fn getenv(name: *const c_char) -> *const c_char;
    fn write(fd: c_int, buf: *const u8, count: usize) -> isize;
}

fn enabled() -> bool {
    let p = unsafe { getenv(c"GGML_NPU_TRACE".as_ptr()) };
    !p.is_null() && unsafe { *p != 0 }
}

fn emit(line: &str) {
    unsafe {
        let _ = write(2, line.as_ptr(), line.len());
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
