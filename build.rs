//! Build script: generate ggml + Level Zero C-ABI bindings with bindgen.
//!
//! ggml headers come from the sibling llama.cpp checkout by default (override with
//! `GGML_DIR`). Level Zero types come from the vendored `ze_api.h`; we generate **types
//! and constants only** — the functions are resolved at runtime via `dlopen`/`dlsym`
//! (see `vendor/ivpu/ze.rs`), so the cdylib has no hard dependency on `libze_loader`.
//!
//! Linking: `ggml_*` helper symbols are left UNDEFINED and resolve at dlopen from the
//! host. Set `GGML_LIB_DIR` to link a prebuilt ggml instead.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolve clang's resource directory (freestanding headers: stddef.h, stdint.h). Prefer
/// the installed clang's own answer; fall back to scanning known locations.
fn clang_resource_dir() -> Option<String> {
    if let Ok(out) = Command::new("clang").arg("-print-resource-dir").output() {
        if out.status.success() {
            let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !dir.is_empty() && Path::new(&dir).join("include/stddef.h").exists() {
                return Some(dir);
            }
        }
    }
    for base in [
        "/usr/lib/clang",
        "/usr/lib64/clang",
        "/usr/lib64/llvm15/lib/clang",
        "/usr/lib/llvm-15/lib/clang",
    ] {
        let Ok(rd) = std::fs::read_dir(base) else {
            continue;
        };
        let mut vers: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        vers.sort();
        for v in vers.into_iter().rev() {
            if v.join("include/stddef.h").exists() {
                return Some(v.to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// Common bindgen settings: no_std output, clang resource dir, and host-target parsing
/// (cross builds need only a linker, not a cross libc sysroot — ggml/L0 ABI structs are
/// LP64-identical across x86_64 and aarch64-linux).
fn base_builder(header: &str, resource: &Option<String>, host: &str) -> bindgen::Builder {
    let mut b = bindgen::Builder::default()
        .header(header)
        .use_core()
        .clang_arg("-xc")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .layout_tests(false);
    if let Some(r) = resource {
        b = b.clang_arg(format!("-resource-dir={r}"));
    }
    if !host.is_empty() {
        b = b.clang_arg(format!("--target={host}"));
    }
    b
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let ggml_root = env::var("GGML_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../llama.cpp"));
    let inc = ggml_root.join("ggml/include");
    let src = ggml_root.join("ggml/src");
    assert!(
        inc.join("ggml-backend.h").exists(),
        "ggml headers not found under {} — set GGML_DIR to your llama.cpp checkout",
        ggml_root.display()
    );
    // ze_graph_ext.h pulls in ze_api.h + ze_graph_profiling_ext.h (same dir), so one bindgen
    // pass yields the core Level Zero types plus the NPU graph extension.
    let ze_dir = manifest.join("src/vendor/ivpu/level_zero");
    let ze_header = ze_dir.join("ze_graph_ext.h");
    assert!(
        ze_header.exists(),
        "vendored ze_graph_ext.h not found at {}",
        ze_header.display()
    );

    println!("cargo:rerun-if-changed=wrapper.h");
    println!(
        "cargo:rerun-if-changed={}",
        inc.join("ggml-backend.h").display()
    );
    println!("cargo:rerun-if-changed={}", ze_header.display());
    println!("cargo:rerun-if-env-changed=GGML_DIR");
    println!("cargo:rerun-if-env-changed=GGML_LIB_DIR");

    if let Ok(libdir) = env::var("GGML_LIB_DIR") {
        println!("cargo:rustc-link-search=native={libdir}");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
    }

    let resource = clang_resource_dir();
    let host = env::var("HOST").unwrap_or_default();
    // Neutralize bindgen's auto --target (would demand a cross sysroot); see base_builder.
    if !host.is_empty() {
        env::set_var("TARGET", &host);
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // ---- ggml bindings ----
    base_builder("wrapper.h", &resource, &host)
        .clang_arg(format!("-I{}", inc.display()))
        .clang_arg(format!("-I{}", src.display()))
        .allowlist_type("ggml_backend.*")
        .allowlist_type("ggml_tensor")
        .allowlist_type("ggml_cgraph")
        .allowlist_type("ggml_status")
        .allowlist_type("ggml_op")
        .allowlist_type("ggml_unary_op")
        .allowlist_type("ggml_glu_op")
        .allowlist_type("ggml_type")
        .allowlist_type("ggml_guid.*")
        // Dequant: ggml_get_type_traits(type)->to_float turns a quantized weight into f32 (pulls
        // in ggml_type_traits + ggml_to_float_t). Symbols resolve at dlopen from libggml-base.
        .allowlist_type("ggml_type_traits")
        .allowlist_function("ggml_get_type_traits")
        .allowlist_function("ggml_backend_buffer_init")
        // cgraph stays opaque; iterate via the public accessors instead of the private struct.
        .allowlist_function("ggml_graph_n_nodes")
        .allowlist_function("ggml_graph_node")
        .allowlist_function("ggml_is_contiguous")
        .allowlist_function("ggml_backend_buft_is_host")
        .allowlist_var("GGML_BACKEND_API_VERSION")
        .generate()
        .expect("bindgen failed for ggml")
        .write_to_file(out.join("bindings.rs"))
        .expect("write bindings.rs");

    // ---- Level Zero bindings (types + constants only; functions are dlsym'd) ----
    // ze_graph_ext.h uses `bool` without including <stdbool.h> (valid in C++, not C), so
    // force-include it for the C parse.
    base_builder(ze_header.to_str().unwrap(), &resource, &host)
        .clang_arg(format!("-I{}", ze_dir.display()))
        .clang_arg("-include")
        .clang_arg("stdbool.h")
        .allowlist_type("ze_.*")
        .allowlist_var("ZE_.*")
        .generate()
        .expect("bindgen failed for Level Zero")
        .write_to_file(out.join("ze_bindings.rs"))
        .expect("write ze_bindings.rs");
}
