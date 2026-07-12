#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

use std::ffi::{c_char, c_int, c_void, CString};
use std::hint::black_box;
use std::time::Instant;

#[allow(dead_code)]
#[path = "../src/ops/quant_kernel.rs"]
mod quant_kernel;
#[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
use quant_kernel::Avx2Kernel;
#[cfg(feature = "portable-simd")]
use quant_kernel::PortableSimdKernel;
use quant_kernel::{ProductionKernel, QuantKernel, ScalarKernel};

const BLOCKS: usize = 4096;
const BLOCK_ELEMS: usize = 256;
const Q4_BYTES: usize = 144;
const Q6_BYTES: usize = 210;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

type Dequantize = unsafe extern "C" fn(*const c_void, *mut f32, i64);

fn deterministic_bytes(bytes: usize, block_bytes: usize) -> Vec<u8> {
    let mut storage: Vec<u8> = (0..bytes)
        .map(|index| {
            (index as u8)
                .wrapping_mul(29)
                .wrapping_add(17)
                .rotate_left((index % 7) as u32)
        })
        .collect();
    for block in storage.chunks_exact_mut(block_bytes) {
        if block_bytes == Q4_BYTES {
            block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        } else {
            block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
    }
    storage
}

fn run_q4(kernel: impl QuantKernel, source: &[u8], output: &mut [u8]) {
    let (source_blocks, source_tail) = source.as_chunks::<Q4_BYTES>();
    let (output_blocks, output_tail) = output.as_chunks_mut::<128>();
    debug_assert!(source_tail.is_empty());
    debug_assert!(output_tail.is_empty());
    for (block, output) in source_blocks.iter().zip(output_blocks) {
        kernel.pack_u4_split_nibbles(block[16..144].try_into().unwrap(), output);
    }
}

fn run_q6(kernel: impl QuantKernel, source: &[u8], output: &mut [u8]) {
    let (source_blocks, source_tail) = source.as_chunks::<Q6_BYTES>();
    let (output_blocks, output_tail) = output.as_chunks_mut::<192>();
    debug_assert!(source_tail.is_empty());
    debug_assert!(output_tail.is_empty());
    for (block, output) in source_blocks.iter().zip(output_blocks) {
        kernel.pack_i6_quartets(
            block[0..128].try_into().unwrap(),
            block[128..192].try_into().unwrap(),
            output,
        );
    }
}

fn run_native(route: &str, format: &str, iterations: usize) -> u64 {
    let production = ProductionKernel::detect();
    #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
    let avx2 = Avx2Kernel::detect();
    match format {
        "q4" => {
            let source = deterministic_bytes(BLOCKS * Q4_BYTES, Q4_BYTES);
            let mut output = vec![0u8; BLOCKS * 128];
            for _ in 0..iterations {
                match route {
                    "scalar" => run_q4(ScalarKernel, black_box(&source), black_box(&mut output)),
                    "production" => run_q4(production, black_box(&source), black_box(&mut output)),
                    #[cfg(feature = "portable-simd")]
                    "portable" => {
                        run_q4(
                            PortableSimdKernel,
                            black_box(&source),
                            black_box(&mut output),
                        );
                    }
                    #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
                    "avx2" => run_q4(
                        avx2.expect("AVX2 is unavailable"),
                        black_box(&source),
                        black_box(&mut output),
                    ),
                    _ => panic!("requested route is not compiled"),
                }
            }
            output.iter().map(|&value| u64::from(value)).sum()
        }
        "q6" => {
            let source = deterministic_bytes(BLOCKS * Q6_BYTES, Q6_BYTES);
            let mut output = vec![0u8; BLOCKS * 192];
            for _ in 0..iterations {
                match route {
                    "scalar" => run_q6(ScalarKernel, black_box(&source), black_box(&mut output)),
                    "production" => run_q6(production, black_box(&source), black_box(&mut output)),
                    #[cfg(feature = "portable-simd")]
                    "portable" => {
                        run_q6(
                            PortableSimdKernel,
                            black_box(&source),
                            black_box(&mut output),
                        );
                    }
                    #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
                    "avx2" => run_q6(
                        avx2.expect("AVX2 is unavailable"),
                        black_box(&source),
                        black_box(&mut output),
                    ),
                    _ => panic!("requested route is not compiled"),
                }
            }
            output.iter().map(|&value| u64::from(value)).sum()
        }
        _ => panic!("format must be q4 or q6"),
    }
}

fn run_ggml(format: &str, iterations: usize) -> u64 {
    let path = CString::new(std::env::var("GGML_SIMD_GGML_BASE").expect("ggml library path"))
        .expect("valid path");
    let handle = unsafe { dlopen(path.as_ptr(), 2) };
    assert!(!handle.is_null(), "failed to load ggml library");
    let symbol = match format {
        "q4" => c"dequantize_row_q4_K",
        "q6" => c"dequantize_row_q6_K",
        _ => panic!("format must be q4 or q6"),
    };
    let function = unsafe { dlsym(handle, symbol.as_ptr()) };
    assert!(!function.is_null(), "missing ggml dequant function");
    let function: Dequantize = unsafe { core::mem::transmute(function) };
    let block_bytes = if format == "q4" { Q4_BYTES } else { Q6_BYTES };
    let source = deterministic_bytes(BLOCKS * block_bytes, block_bytes);
    let mut output = vec![0.0f32; BLOCKS * BLOCK_ELEMS];
    for _ in 0..iterations {
        unsafe {
            function(
                black_box(source.as_ptr()).cast(),
                black_box(output.as_mut_ptr()),
                (BLOCKS * BLOCK_ELEMS) as i64,
            );
        }
    }
    unsafe { dlclose(handle) };
    output
        .iter()
        .map(|value| u64::from(value.to_bits()))
        .fold(0, u64::wrapping_add)
}

fn main() {
    let route = std::env::var("QUANT_ROUTE").unwrap_or_else(|_| "scalar".into());
    let format = std::env::var("QUANT_FORMAT").unwrap_or_else(|_| "q4".into());
    let iterations = std::env::var("QUANT_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5_000);
    let start = Instant::now();
    let checksum = if route == "ggml" {
        run_ggml(&format, iterations)
    } else {
        run_native(&route, &format, iterations)
    };
    let elapsed = start.elapsed();
    let elements = BLOCKS * BLOCK_ELEMS * iterations;
    let throughput = elements as f64 / elapsed.as_secs_f64() / 1e9;
    eprintln!(
        "route={route} selected={} format={format} blocks={BLOCKS} iterations={iterations} elements={elements} elapsed_us={:.3} throughput_gelem_s={throughput:.6} checksum={checksum}",
        ProductionKernel::detect().name(),
        elapsed.as_secs_f64() * 1e6,
    );
}
