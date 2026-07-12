#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

use std::ffi::{c_char, c_int, c_void, CString};
use std::hint::black_box;
use std::ptr;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[allow(dead_code)]
#[path = "../src/ops/quant_kernel.rs"]
mod quant_kernel;
#[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
use quant_kernel::Avx2Kernel;
#[cfg(feature = "portable-simd")]
use quant_kernel::PortableSimdKernel;
use quant_kernel::{ProductionKernel, QuantKernel, ScalarKernel};

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

struct GgmlKernels {
    handle: *mut c_void,
    q4: Dequantize,
    q6: Dequantize,
}

impl GgmlKernels {
    fn load() -> Option<Self> {
        let path = std::env::var("GGML_SIMD_GGML_BASE").ok()?;
        let path = CString::new(path).ok()?;
        let handle = unsafe { dlopen(path.as_ptr(), 2) };
        if handle.is_null() {
            return None;
        }
        let q4 = unsafe { dlsym(handle, c"dequantize_row_q4_K".as_ptr()) };
        let q6 = unsafe { dlsym(handle, c"dequantize_row_q6_K".as_ptr()) };
        if q4.is_null() || q6.is_null() {
            unsafe { dlclose(handle) };
            return None;
        }
        Some(Self {
            handle,
            q4: unsafe { core::mem::transmute::<*mut c_void, Dequantize>(q4) },
            q6: unsafe { core::mem::transmute::<*mut c_void, Dequantize>(q6) },
        })
    }
}

impl Drop for GgmlKernels {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { dlclose(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

fn deterministic_bytes(bytes: usize, prefix: usize, block_bytes: usize) -> Vec<u8> {
    let mut storage = vec![0u8; prefix + bytes];
    for (index, byte) in storage[prefix..].iter_mut().enumerate() {
        *byte = (index as u8)
            .wrapping_mul(29)
            .wrapping_add(17)
            .rotate_left((index % 7) as u32);
    }
    for block in storage[prefix..].chunks_exact_mut(block_bytes) {
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
        let source: &[u8; 128] = block[16..144].try_into().unwrap();
        kernel.pack_u4_split_nibbles(source, output);
    }
}

fn run_q6(kernel: impl QuantKernel, source: &[u8], output: &mut [u8]) {
    let (source_blocks, source_tail) = source.as_chunks::<Q6_BYTES>();
    let (output_blocks, output_tail) = output.as_chunks_mut::<192>();
    debug_assert!(source_tail.is_empty());
    debug_assert!(output_tail.is_empty());
    for (block, output) in source_blocks.iter().zip(output_blocks) {
        let low: &[u8; 128] = block[0..128].try_into().unwrap();
        let high: &[u8; 64] = block[128..192].try_into().unwrap();
        kernel.pack_i6_quartets(low, high, output);
    }
}

fn assert_exact_routes(q4: &[u8], q6: &[u8], blocks: usize) {
    let mut expected_q4 = vec![0u8; blocks * 128];
    let mut expected_q6 = vec![0u8; blocks * 192];
    run_q4(ScalarKernel, q4, &mut expected_q4);
    run_q6(ScalarKernel, q6, &mut expected_q6);

    #[cfg(feature = "portable-simd")]
    {
        let mut actual_q4 = vec![0u8; blocks * 128];
        let mut actual_q6 = vec![0u8; blocks * 192];
        run_q4(PortableSimdKernel, q4, &mut actual_q4);
        run_q6(PortableSimdKernel, q6, &mut actual_q6);
        assert_eq!(actual_q4, expected_q4);
        assert_eq!(actual_q6, expected_q6);
    }

    #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
    if let Some(kernel) = Avx2Kernel::detect() {
        let mut actual_q4 = vec![0u8; blocks * 128];
        let mut actual_q6 = vec![0u8; blocks * 192];
        run_q4(kernel, q4, &mut actual_q4);
        run_q6(kernel, q6, &mut actual_q6);
        assert_eq!(actual_q4, expected_q4);
        assert_eq!(actual_q6, expected_q6);
    }
}

fn quant_kernels(criterion: &mut Criterion) {
    let ggml = GgmlKernels::load();
    let production = ProductionKernel::detect();
    #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
    let avx2 = Avx2Kernel::detect();
    eprintln!(
        "quant SIMD routes: production={} scalar=yes portable_simd={} avx2={} ggml_extern={}",
        production.name(),
        cfg!(feature = "portable-simd"),
        {
            #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
            {
                avx2.is_some()
            }
            #[cfg(not(all(feature = "portable-simd", target_arch = "x86_64")))]
            {
                false
            }
        },
        ggml.is_some()
    );
    eprintln!(
        "bounded scratch: native scalar=0 B, portable/avx2<=64 B/block, ggml={} B/element",
        core::mem::size_of::<f32>()
    );

    for (shape, blocks, prefix) in [
        ("one_block_aligned", 1usize, 0usize),
        ("tail_17_blocks_unaligned", 17, 1),
        ("tensor_k4096_n256", 4096, 0),
    ] {
        let q4_storage = deterministic_bytes(blocks * Q4_BYTES, prefix, Q4_BYTES);
        let q6_storage = deterministic_bytes(blocks * Q6_BYTES, prefix, Q6_BYTES);
        let q4 = &q4_storage[prefix..];
        let q6 = &q6_storage[prefix..];
        assert_exact_routes(q4, q6, blocks);

        let elems = blocks * BLOCK_ELEMS;
        let mut group = criterion.benchmark_group(format!("q4_k/{shape}"));
        group
            .throughput(Throughput::Elements(elems as u64))
            .warm_up_time(Duration::from_millis(400))
            .measurement_time(Duration::from_secs(1))
            .sample_size(20);
        group.bench_with_input(BenchmarkId::new("scalar_native", elems), &elems, |b, _| {
            let mut output = vec![0u8; blocks * 128];
            b.iter(|| run_q4(ScalarKernel, black_box(q4), black_box(&mut output)))
        });
        #[cfg(feature = "portable-simd")]
        group.bench_with_input(BenchmarkId::new("portable_simd", elems), &elems, |b, _| {
            let mut output = vec![0u8; blocks * 128];
            b.iter(|| run_q4(PortableSimdKernel, black_box(q4), black_box(&mut output)))
        });
        #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
        if let Some(kernel) = avx2 {
            group.bench_with_input(BenchmarkId::new("portable_avx2", elems), &elems, |b, _| {
                let mut output = vec![0u8; blocks * 128];
                b.iter(|| run_q4(kernel, black_box(q4), black_box(&mut output)))
            });
        }
        if let Some(ggml) = ggml.as_ref().filter(|_| prefix == 0) {
            group.bench_with_input(
                BenchmarkId::new("ggml_extern_dequant_f32", elems),
                &elems,
                |b, _| {
                    let mut output = vec![0.0f32; elems];
                    b.iter(|| unsafe {
                        (ggml.q4)(
                            black_box(q4.as_ptr()).cast(),
                            black_box(output.as_mut_ptr()),
                            elems as i64,
                        )
                    })
                },
            );
        }
        group.finish();

        let mut group = criterion.benchmark_group(format!("q6_k/{shape}"));
        group
            .throughput(Throughput::Elements(elems as u64))
            .warm_up_time(Duration::from_millis(400))
            .measurement_time(Duration::from_secs(1))
            .sample_size(20);
        group.bench_with_input(BenchmarkId::new("scalar_native", elems), &elems, |b, _| {
            let mut output = vec![0u8; blocks * 192];
            b.iter(|| run_q6(ScalarKernel, black_box(q6), black_box(&mut output)))
        });
        #[cfg(feature = "portable-simd")]
        group.bench_with_input(BenchmarkId::new("portable_simd", elems), &elems, |b, _| {
            let mut output = vec![0u8; blocks * 192];
            b.iter(|| run_q6(PortableSimdKernel, black_box(q6), black_box(&mut output)))
        });
        #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
        if let Some(kernel) = avx2 {
            group.bench_with_input(BenchmarkId::new("portable_avx2", elems), &elems, |b, _| {
                let mut output = vec![0u8; blocks * 192];
                b.iter(|| run_q6(kernel, black_box(q6), black_box(&mut output)))
            });
        }
        if let Some(ggml) = ggml.as_ref().filter(|_| prefix == 0) {
            group.bench_with_input(
                BenchmarkId::new("ggml_extern_dequant_f32", elems),
                &elems,
                |b, _| {
                    let mut output = vec![0.0f32; elems];
                    b.iter(|| unsafe {
                        (ggml.q6)(
                            black_box(q6.as_ptr()).cast(),
                            black_box(output.as_mut_ptr()),
                            elems as i64,
                        )
                    })
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, quant_kernels);
criterion_main!(benches);
