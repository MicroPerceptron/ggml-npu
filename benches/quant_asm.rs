#![feature(portable_simd)]

use std::hint::black_box;

#[allow(dead_code)]
#[path = "../src/ops/quant_kernel.rs"]
mod quant_kernel;

#[cfg(target_arch = "x86_64")]
use quant_kernel::Avx2Kernel;
use quant_kernel::{PortableSimdKernel, QuantKernel, ScalarKernel};

#[no_mangle]
#[inline(never)]
/// # Safety
/// `source` and `output` must reference non-overlapping 128-byte buffers.
pub unsafe extern "C" fn quant_scalar_q4(source: *const u8, output: *mut u8) {
    ScalarKernel.pack_u4_split_nibbles(&*source.cast(), &mut *output.cast());
}

#[no_mangle]
#[inline(never)]
/// # Safety
/// `source` and `output` must reference non-overlapping 128-byte buffers.
pub unsafe extern "C" fn quant_portable_q4(source: *const u8, output: *mut u8) {
    PortableSimdKernel.pack_u4_split_nibbles(&*source.cast(), &mut *output.cast());
}

#[no_mangle]
#[inline(never)]
#[cfg(target_arch = "x86_64")]
/// # Safety
/// `source` and `output` must reference non-overlapping 128-byte buffers.
pub unsafe extern "C" fn quant_avx2_q4(source: *const u8, output: *mut u8) {
    Avx2Kernel::detect()
        .expect("assembly inspection host lacks AVX2")
        .pack_u4_split_nibbles(&*source.cast(), &mut *output.cast());
}

#[no_mangle]
#[inline(never)]
/// # Safety
/// `low`, `high`, and `output` must reference non-overlapping 128-, 64-, and 192-byte buffers.
pub unsafe extern "C" fn quant_scalar_q6(low: *const u8, high: *const u8, output: *mut u8) {
    ScalarKernel.pack_i6_quartets(&*low.cast(), &*high.cast(), &mut *output.cast());
}

#[no_mangle]
#[inline(never)]
/// # Safety
/// `low`, `high`, and `output` must reference non-overlapping 128-, 64-, and 192-byte buffers.
pub unsafe extern "C" fn quant_portable_q6(low: *const u8, high: *const u8, output: *mut u8) {
    PortableSimdKernel.pack_i6_quartets(&*low.cast(), &*high.cast(), &mut *output.cast());
}

#[no_mangle]
#[inline(never)]
#[cfg(target_arch = "x86_64")]
/// # Safety
/// `low`, `high`, and `output` must reference non-overlapping 128-, 64-, and 192-byte buffers.
pub unsafe extern "C" fn quant_avx2_q6(low: *const u8, high: *const u8, output: *mut u8) {
    Avx2Kernel::detect()
        .expect("assembly inspection host lacks AVX2")
        .pack_i6_quartets(&*low.cast(), &*high.cast(), &mut *output.cast());
}

fn main() {
    let source = [0u8; 128];
    let high = [0u8; 64];
    let mut q4 = [0u8; 128];
    let mut q6 = [0u8; 192];
    unsafe {
        quant_scalar_q4(black_box(source.as_ptr()), black_box(q4.as_mut_ptr()));
        quant_portable_q4(black_box(source.as_ptr()), black_box(q4.as_mut_ptr()));
        #[cfg(target_arch = "x86_64")]
        quant_avx2_q4(black_box(source.as_ptr()), black_box(q4.as_mut_ptr()));
        quant_scalar_q6(
            black_box(source.as_ptr()),
            black_box(high.as_ptr()),
            black_box(q6.as_mut_ptr()),
        );
        quant_portable_q6(
            black_box(source.as_ptr()),
            black_box(high.as_ptr()),
            black_box(q6.as_mut_ptr()),
        );
        #[cfg(target_arch = "x86_64")]
        {
            quant_avx2_q6(
                black_box(source.as_ptr()),
                black_box(high.as_ptr()),
                black_box(q6.as_mut_ptr()),
            );
        }
    }
    black_box((q4, q6));
}
