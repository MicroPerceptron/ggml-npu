//! Neutral integer packing kernels used below source-format adapters.
//!
//! The contract describes logical bit packing, not a CPU ISA or compiler backend. Optional
//! implementations can select architecture features internally while callers retain one exact
//! byte-level contract.

/// Exact integer transforms needed by the source-native 4-bit and 6-bit adapters.
pub trait QuantKernel {
    /// Convert four 32-byte chunks whose low/high nibbles represent consecutive groups into a
    /// dense logical-order 4-bit stream.
    fn pack_u4_split_nibbles(&self, source: &[u8; 128], output: &mut [u8; 128]);

    /// Combine low-nibble and high-two-bit planes, convert offset-binary values to signed 6-bit
    /// two's-complement, and emit a dense logical-order stream.
    fn pack_i6_quartets(&self, low: &[u8; 128], high: &[u8; 64], output: &mut [u8; 192]);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarKernel;

impl QuantKernel for ScalarKernel {
    fn pack_u4_split_nibbles(&self, source: &[u8; 128], output: &mut [u8; 128]) {
        for chunk in 0..4 {
            let source = &source[chunk * 32..chunk * 32 + 32];
            let output = &mut output[chunk * 32..chunk * 32 + 32];
            for pair in 0..16 {
                let even = source[pair * 2];
                let odd = source[pair * 2 + 1];
                output[pair] = (even & 0x0f) | ((odd & 0x0f) << 4);
                output[16 + pair] = (even >> 4) | (odd & 0xf0);
            }
        }
    }

    fn pack_i6_quartets(&self, low: &[u8; 128], high: &[u8; 64], output: &mut [u8; 192]) {
        for half in 0..2 {
            for quarter in 0..4 {
                for lane in (0..32).step_by(4) {
                    let quants = [
                        decode_i6(low, high, half, quarter, lane),
                        decode_i6(low, high, half, quarter, lane + 1),
                        decode_i6(low, high, half, quarter, lane + 2),
                        decode_i6(low, high, half, quarter, lane + 3),
                    ];
                    let destination = half * 96 + quarter * 24 + lane / 4 * 3;
                    pack_i6x4(quants, &mut output[destination..destination + 3]);
                }
            }
        }
    }
}

/// Kernel selected by the normal production build.
///
/// Selection never relies on global target features. The portable route is baseline-safe, while
/// the x86-64 AVX2 specialization is reached only after cached OS-aware runtime detection.
#[derive(Clone, Copy, Debug)]
pub enum ProductionKernel {
    Scalar(ScalarKernel),
    #[cfg(all(
        feature = "portable-simd",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "arm",
            target_arch = "aarch64"
        )
    ))]
    Portable(PortableSimdKernel),
    #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
    Avx2Q6(Avx2Kernel),
}

impl ProductionKernel {
    pub fn detect() -> Self {
        #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
        if let Some(kernel) = Avx2Kernel::detect() {
            return Self::Avx2Q6(kernel);
        }

        #[cfg(all(
            feature = "portable-simd",
            any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "arm",
                target_arch = "aarch64"
            )
        ))]
        {
            return Self::Portable(PortableSimdKernel);
        }

        #[allow(unreachable_code)]
        Self::Scalar(ScalarKernel)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar(_) => "scalar",
            #[cfg(all(
                feature = "portable-simd",
                any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "arm",
                    target_arch = "aarch64"
                )
            ))]
            Self::Portable(_) => "portable-simd",
            #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
            Self::Avx2Q6(_) => "portable-simd+avx2-q6",
        }
    }
}

impl Default for ProductionKernel {
    fn default() -> Self {
        Self::detect()
    }
}

impl QuantKernel for ProductionKernel {
    fn pack_u4_split_nibbles(&self, source: &[u8; 128], output: &mut [u8; 128]) {
        match self {
            Self::Scalar(kernel) => kernel.pack_u4_split_nibbles(source, output),
            #[cfg(all(
                feature = "portable-simd",
                any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "arm",
                    target_arch = "aarch64"
                )
            ))]
            Self::Portable(kernel) => kernel.pack_u4_split_nibbles(source, output),
            #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
            Self::Avx2Q6(_) => PortableSimdKernel.pack_u4_split_nibbles(source, output),
        }
    }

    fn pack_i6_quartets(&self, low: &[u8; 128], high: &[u8; 64], output: &mut [u8; 192]) {
        match self {
            Self::Scalar(kernel) => kernel.pack_i6_quartets(low, high, output),
            #[cfg(all(
                feature = "portable-simd",
                any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "arm",
                    target_arch = "aarch64"
                )
            ))]
            Self::Portable(kernel) => kernel.pack_i6_quartets(low, high, output),
            #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
            Self::Avx2Q6(kernel) => kernel.pack_i6_quartets(low, high, output),
        }
    }
}

#[cfg(feature = "portable-simd")]
#[derive(Clone, Copy, Debug, Default)]
pub struct PortableSimdKernel;

#[cfg(feature = "portable-simd")]
impl QuantKernel for PortableSimdKernel {
    fn pack_u4_split_nibbles(&self, source: &[u8; 128], output: &mut [u8; 128]) {
        use core::simd::Simd;

        let mask = Simd::<u8, 32>::splat(0x0f);
        let shift = Simd::<u8, 32>::splat(4);
        for chunk in 0..4 {
            let packed = Simd::<u8, 32>::from_slice(&source[chunk * 32..chunk * 32 + 32]);
            let low = (packed & mask).to_array();
            let high = (packed >> shift).to_array();
            let output = &mut output[chunk * 32..chunk * 32 + 32];
            for pair in 0..16 {
                output[pair] = low[pair * 2] | (low[pair * 2 + 1] << 4);
                output[16 + pair] = high[pair * 2] | (high[pair * 2 + 1] << 4);
            }
        }
    }

    fn pack_i6_quartets(&self, low: &[u8; 128], high: &[u8; 64], output: &mut [u8; 192]) {
        use core::simd::Simd;

        let low_mask = Simd::<u8, 32>::splat(0x0f);
        let high_mask = Simd::<u8, 32>::splat(0x03);
        let signed_bias = Simd::<u8, 32>::splat(0x20);
        for half in 0..2 {
            let low_a = Simd::<u8, 32>::from_slice(&low[half * 64..half * 64 + 32]);
            let low_b = Simd::<u8, 32>::from_slice(&low[half * 64 + 32..half * 64 + 64]);
            let high = Simd::<u8, 32>::from_slice(&high[half * 32..half * 32 + 32]);
            for quarter in 0..4 {
                let low = match quarter {
                    0 => low_a & low_mask,
                    1 => low_b & low_mask,
                    2 => low_a >> Simd::splat(4),
                    _ => low_b >> Simd::splat(4),
                };
                let high = (high >> Simd::splat(quarter as u8 * 2)) & high_mask;
                let quants = (low | (high << Simd::splat(4))) ^ signed_bias;
                let quants = quants.to_array();
                for lane in (0..32).step_by(4) {
                    let destination = half * 96 + quarter * 24 + lane / 4 * 3;
                    pack_i6x4(
                        [
                            quants[lane],
                            quants[lane + 1],
                            quants[lane + 2],
                            quants[lane + 3],
                        ],
                        &mut output[destination..destination + 3],
                    );
                }
            }
        }
    }
}

#[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
cpufeatures::new!(quant_avx2, "avx2");

/// AVX2 specialization constructed only after OS-aware runtime feature detection.
#[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub struct Avx2Kernel(());

#[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
impl Avx2Kernel {
    pub fn detect() -> Option<Self> {
        quant_avx2::get().then_some(Self(()))
    }
}

#[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
impl QuantKernel for Avx2Kernel {
    fn pack_u4_split_nibbles(&self, source: &[u8; 128], output: &mut [u8; 128]) {
        // Construction proves both CPU and OS AVX2 support for this process.
        unsafe { pack_u4_split_nibbles_avx2(source, output) }
    }

    fn pack_i6_quartets(&self, low: &[u8; 128], high: &[u8; 64], output: &mut [u8; 192]) {
        // Construction proves both CPU and OS AVX2 support for this process.
        unsafe { pack_i6_quartets_avx2(low, high, output) }
    }
}

#[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn pack_u4_split_nibbles_avx2(source: &[u8; 128], output: &mut [u8; 128]) {
    use core::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_packus_epi16,
        _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi8, _mm256_srli_epi16,
        _mm256_storeu_si256,
    };

    let nibble_mask = _mm256_set1_epi8(0x0f);
    let pair_weights = _mm256_set1_epi16(0x1001);
    for chunk in 0..4 {
        let source = source.as_ptr().add(chunk * 32).cast::<__m256i>();
        let packed = _mm256_loadu_si256(source);
        let low = _mm256_and_si256(packed, nibble_mask);
        let high = _mm256_and_si256(_mm256_srli_epi16::<4>(packed), nibble_mask);
        let low_pairs = _mm256_maddubs_epi16(low, pair_weights);
        let high_pairs = _mm256_maddubs_epi16(high, pair_weights);
        let lane_ordered =
            _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi16(low_pairs, high_pairs));
        let output = output.as_mut_ptr().add(chunk * 32).cast::<__m256i>();
        _mm256_storeu_si256(output, lane_ordered);
    }
}

#[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn pack_i6_quartets_avx2(low: &[u8; 128], high: &[u8; 64], output: &mut [u8; 192]) {
    use core::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi8,
        _mm256_slli_epi16, _mm256_srli_epi16, _mm256_storeu_si256, _mm256_xor_si256,
    };

    let low_mask = _mm256_set1_epi8(0x0f);
    let high_mask = _mm256_set1_epi8(0x03);
    let signed_bias = _mm256_set1_epi8(0x20);
    for half in 0..2 {
        let low_a = _mm256_loadu_si256(low.as_ptr().add(half * 64).cast::<__m256i>());
        let low_b = _mm256_loadu_si256(low.as_ptr().add(half * 64 + 32).cast::<__m256i>());
        let high = _mm256_loadu_si256(high.as_ptr().add(half * 32).cast::<__m256i>());

        for quarter in 0..4 {
            let low = match quarter {
                0 => _mm256_and_si256(low_a, low_mask),
                1 => _mm256_and_si256(low_b, low_mask),
                2 => _mm256_and_si256(_mm256_srli_epi16::<4>(low_a), low_mask),
                _ => _mm256_and_si256(_mm256_srli_epi16::<4>(low_b), low_mask),
            };
            let high = match quarter {
                0 => _mm256_and_si256(high, high_mask),
                1 => _mm256_and_si256(_mm256_srli_epi16::<2>(high), high_mask),
                2 => _mm256_and_si256(_mm256_srli_epi16::<4>(high), high_mask),
                _ => _mm256_and_si256(_mm256_srli_epi16::<6>(high), high_mask),
            };
            let quants = _mm256_xor_si256(
                _mm256_or_si256(low, _mm256_slli_epi16::<4>(high)),
                signed_bias,
            );
            let mut unpacked = [0u8; 32];
            _mm256_storeu_si256(unpacked.as_mut_ptr().cast::<__m256i>(), quants);
            for lane in (0..32).step_by(4) {
                let destination = half * 96 + quarter * 24 + lane / 4 * 3;
                pack_i6x4(
                    [
                        unpacked[lane],
                        unpacked[lane + 1],
                        unpacked[lane + 2],
                        unpacked[lane + 3],
                    ],
                    &mut output[destination..destination + 3],
                );
            }
        }
    }
}

#[inline]
fn decode_i6(low: &[u8; 128], high: &[u8; 64], half: usize, quarter: usize, lane: usize) -> u8 {
    let low_base = half * 64;
    let high_byte = high[half * 32 + lane];
    let quant = match quarter {
        0 => (low[low_base + lane] & 0x0f) | ((high_byte & 0x03) << 4),
        1 => (low[low_base + 32 + lane] & 0x0f) | (((high_byte >> 2) & 0x03) << 4),
        2 => (low[low_base + lane] >> 4) | (((high_byte >> 4) & 0x03) << 4),
        _ => (low[low_base + 32 + lane] >> 4) | (((high_byte >> 6) & 0x03) << 4),
    };
    quant ^ 0x20
}

#[inline]
fn pack_i6x4(values: [u8; 4], output: &mut [u8]) {
    output[0] = values[0] | (values[1] << 6);
    output[1] = (values[1] >> 2) | (values[2] << 4);
    output[2] = (values[2] >> 4) | (values[3] << 2);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic<const N: usize>(seed: u8) -> [u8; N] {
        core::array::from_fn(|index| {
            (index as u8)
                .wrapping_mul(29)
                .wrapping_add(seed)
                .rotate_left((index % 7) as u32)
        })
    }

    #[test]
    fn scalar_packing_is_stable_for_edges() {
        let mut u4_source = deterministic::<128>(17);
        u4_source[0] = 0x00;
        u4_source[1] = 0xff;
        let mut u4 = [0u8; 128];
        ScalarKernel.pack_u4_split_nibbles(&u4_source, &mut u4);
        assert_eq!(&u4[..4], &[0xf0, 0x3d, 0x48, 0xcf]);

        let low = deterministic::<128>(31);
        let high = deterministic::<64>(73);
        let mut i6 = [0u8; 192];
        ScalarKernel.pack_i6_quartets(&low, &high, &mut i6);
        assert_eq!(&i6[..6], &[0x3f, 0x5a, 0xcc, 0x99, 0x35, 0xab]);
    }

    #[cfg(feature = "portable-simd")]
    #[test]
    fn portable_simd_is_byte_exact() {
        let u4_source = deterministic::<128>(17);
        let low = deterministic::<128>(31);
        let high = deterministic::<64>(73);
        assert_kernels_equal(ScalarKernel, PortableSimdKernel, &u4_source, &low, &high);
    }

    #[test]
    fn production_kernel_is_observable_and_byte_exact() {
        let u4_source = deterministic::<128>(17);
        let low = deterministic::<128>(31);
        let high = deterministic::<64>(73);
        let kernel = ProductionKernel::detect();
        assert!(["scalar", "portable-simd", "portable-simd+avx2-q6"].contains(&kernel.name()));
        assert_kernels_equal(ScalarKernel, kernel, &u4_source, &low, &high);
    }

    #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
    #[test]
    fn runtime_avx2_is_byte_exact_when_available() {
        let Some(kernel) = Avx2Kernel::detect() else {
            return;
        };
        let u4_source = deterministic::<128>(17);
        let low = deterministic::<128>(31);
        let high = deterministic::<64>(73);
        assert_kernels_equal(ScalarKernel, kernel, &u4_source, &low, &high);
    }

    fn assert_kernels_equal(
        expected: impl QuantKernel,
        actual: impl QuantKernel,
        u4_source: &[u8; 128],
        low: &[u8; 128],
        high: &[u8; 64],
    ) {
        let mut expected_u4 = [0u8; 128];
        let mut actual_u4 = [0u8; 128];
        expected.pack_u4_split_nibbles(u4_source, &mut expected_u4);
        actual.pack_u4_split_nibbles(u4_source, &mut actual_u4);
        assert_eq!(actual_u4, expected_u4);

        let mut expected_i6 = [0u8; 192];
        let mut actual_i6 = [0u8; 192];
        expected.pack_i6_quartets(low, high, &mut expected_i6);
        actual.pack_i6_quartets(low, high, &mut actual_i6);
        assert_eq!(actual_i6, expected_i6);
    }
}
