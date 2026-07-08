//! Build script: generate ggml C-ABI bindings with bindgen.
//!
//! Header source tree is the sibling llama.cpp checkout by default; override with
//! `GGML_DIR=/path/to/llama.cpp`.
//!
//! Linking: by default we leave the `ggml_*` helper symbols (e.g.
//! ggml_backend_buffer_init) UNDEFINED. A cdylib is allowed undefined symbols; they
//! resolve at dlopen time from the host process (llama-cli / test-backend-ops), which
//! already has ggml loaded in its global scope. Set `GGML_LIB_DIR=/path/with/libggml`
//! to link against a prebuilt ggml instead (deterministic, catches typos early).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolve clang's resource directory (the freestanding headers: stddef.h, stdint.h,
/// stdbool.h). Prefer the installed clang's own answer; fall back to scanning known
/// locations for machines whose libclang can't self-locate them (e.g. Fedora's
/// lib/lib64 split).
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

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=GGML_DIR");
    println!("cargo:rerun-if-env-changed=GGML_LIB_DIR");

    if let Ok(libdir) = env::var("GGML_LIB_DIR") {
        println!("cargo:rustc-link-search=native={libdir}");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
    }

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", inc.display()))
        .clang_arg(format!("-I{}", src.display()))
        .clang_arg("-xc")
        // no_std crate: emit ::core::ffi::* instead of ::std::os::raw::*.
        .use_core();

    // Point libclang at clang's resource dir so the freestanding headers resolve
    // (some libclang installs don't auto-find them across a lib/lib64 split).
    if let Some(rd) = clang_resource_dir() {
        builder = builder.clang_arg(format!("-resource-dir={rd}"));
    }

    // Force host-target parsing. bindgen otherwise derives its clang `--target` from
    // cargo's $TARGET, which for a cross build would require a full cross libc sysroot
    // just to parse <stdlib.h>. ggml's ABI structs are LP64-identical across our x86_64
    // and aarch64-linux targets, so host-parsed bindings are correct for both — the cross
    // build then needs only a cross linker, not a sysroot. (Revisit for a non-LP64 or
    // Windows LLP64 host.)
    let host = env::var("HOST").unwrap_or_default();
    if !host.is_empty() {
        env::set_var("TARGET", &host);
        builder = builder.clang_arg(format!("--target={host}"));
    }

    let bindings = builder
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .allowlist_type("ggml_backend.*")
        .allowlist_type("ggml_tensor")
        .allowlist_type("ggml_cgraph")
        .allowlist_type("ggml_status")
        .allowlist_type("ggml_guid.*")
        .allowlist_function("ggml_backend_buffer_init")
        .allowlist_var("GGML_BACKEND_API_VERSION")
        .layout_tests(false)
        .generate()
        .expect("bindgen failed to generate ggml bindings");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
