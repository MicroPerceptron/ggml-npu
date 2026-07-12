#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-$ROOT/../llama.cpp}"
MODE="${1:-all}" # native | openblas | flexiblas | mkl | all
JOBS="${JOBS:-$(nproc)}"

configure_and_build() {
    local name="$1"
    shift
    cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build-cpu-$name" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_SYCL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        "$@"
    cmake --build "$LLAMA_ROOT/build-cpu-$name" --target llama-bench -j "$JOBS"
}

build_native() {
    configure_and_build native -DGGML_BLAS=OFF -DGGML_NATIVE=ON
}

build_openblas() {
    if ! pkg-config --exists openblas64 && ! pkg-config --exists openblas; then
        echo "OpenBLAS development metadata is unavailable; skipping (runtime-only library found)." >&2
        return 0
    fi
    configure_and_build openblas -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON
}

build_flexiblas() {
    if ! pkg-config --exists flexiblas_api; then
        echo "FlexiBLAS development metadata is unavailable; skipping." >&2
        return 0
    fi
    configure_and_build flexiblas -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FlexiBLAS -DGGML_NATIVE=ON
}

build_mkl() {
    local oneapi="${ONEAPI_ROOT:-}"
    local icx="${MKL_CC:-}"
    local icpx="${MKL_CXX:-}"
    local mklroot="${MKLROOT:-}"
    local pkg_config_path="${PKG_CONFIG_PATH:-}"

    if [[ -n "$oneapi" ]]; then
        icx="${icx:-$oneapi/compiler/latest/bin/icx}"
        icpx="${icpx:-$oneapi/compiler/latest/bin/icpx}"
        mklroot="${mklroot:-$oneapi/mkl/latest}"
    else
        icx="${icx:-$(type -P icx || true)}"
        icpx="${icpx:-$(type -P icpx || true)}"
    fi
    if [[ -n "$mklroot" && -f "$mklroot/lib/pkgconfig/mkl-sdl.pc" ]]; then
        pkg_config_path="$mklroot/lib/pkgconfig${pkg_config_path:+:$pkg_config_path}"
    fi
    if [[ ! -x "$icx" || ! -x "$icpx" ]] ||
        ! PKG_CONFIG_PATH="$pkg_config_path" pkg-config --exists mkl-sdl; then
        echo "oneMKL development files or oneAPI compilers are unavailable; skipping." >&2
        return 0
    fi
    if [[ -z "$mklroot" ]]; then
        mklroot="$(PKG_CONFIG_PATH="$pkg_config_path" pkg-config --variable=prefix mkl-sdl)"
    fi
    MKLROOT="$mklroot" PKG_CONFIG_PATH="$pkg_config_path" \
        configure_and_build mkl \
            -DGGML_BLAS=ON \
            -DGGML_BLAS_VENDOR=Intel10_64lp \
            -DGGML_NATIVE=ON \
            -DCMAKE_C_COMPILER="$icx" \
            -DCMAKE_CXX_COMPILER="$icpx"
}

case "$MODE" in
    native) build_native ;;
    openblas) build_openblas ;;
    flexiblas) build_flexiblas ;;
    mkl) build_mkl ;;
    all)
        build_native
        build_openblas
        build_flexiblas
        build_mkl
        ;;
    *)
        echo "usage: $0 [native|openblas|flexiblas|mkl|all]" >&2
        exit 2
        ;;
esac
