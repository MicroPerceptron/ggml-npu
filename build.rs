//! Generate bindings from the same pinned ggml headers used to build the host.
use std::{env, path::PathBuf, process::Command};

fn main() {
    let root = env::var_os("GGML_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../llama.cpp")
        });
    let inc = root.join("ggml/include");
    let src = root.join("ggml/src");
    assert!(
        src.join("ggml-backend-impl.h").exists(),
        "set GGML_DIR to the pinned llama.cpp checkout"
    );
    for name in ["GGML_DIR", "GGML_LIB_DIR"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-changed=wrapper.h");
    for path in [
        inc.join("ggml.h"),
        inc.join("ggml-backend.h"),
        src.join("ggml-backend-impl.h"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    if let Some(libdir) = env::var_os("GGML_LIB_DIR") {
        println!(
            "cargo:rustc-link-search=native={}",
            PathBuf::from(libdir).display()
        );
        println!("cargo:rustc-link-lib=dylib=ggml-base");
    } else if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");
    }
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-xc")
        .clang_arg(format!("-I{}", inc.display()))
        .clang_arg(format!("-I{}", src.display()))
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .layout_tests(false)
        .derive_default(true)
        .allowlist_type("ggml_backend.*")
        .allowlist_type("ggml_tensor")
        .allowlist_type("ggml_cgraph")
        .allowlist_type("ggml_status")
        .allowlist_type("ggml_op")
        .allowlist_type("ggml_unary_op")
        .allowlist_type("ggml_type")
        .allowlist_type("ggml_guid.*")
        .allowlist_function("ggml_backend_buffer_init")
        .allowlist_function("ggml_backend_buft_is_host")
        .allowlist_function("ggml_graph_n_nodes")
        .allowlist_function("ggml_graph_node")
        .allowlist_var("GGML_BACKEND_API_VERSION");
    if let Ok(out) = Command::new("clang").arg("-print-resource-dir").output()
        && out.status.success()
    {
        builder = builder.clang_arg(format!(
            "-resource-dir={}",
            String::from_utf8_lossy(&out.stdout).trim()
        ));
    }
    builder
        .generate()
        .expect("generate ggml bindings")
        .write_to_file(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("bindings.rs"))
        .expect("write bindings");
}
