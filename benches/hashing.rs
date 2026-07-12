use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

extern crate alloc;
#[allow(dead_code, unused_imports)]
#[path = "../src/identity.rs"]
mod identity;
use identity::{SemanticHasher, IDENTITY_SCHEMA_VERSION};

const FIVE_MIB: usize = 5 * 1024 * 1024;
const MOE_BANK_CHUNKS: usize = 24;

fn deterministic_bytes(bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(17))
        .collect()
}

fn naive_fields(domain: &[u8], fields: usize) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ggml-npu-semantic-identity\0");
    hasher.update(&IDENTITY_SCHEMA_VERSION.to_le_bytes());
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for index in 0..fields {
        let label = b"scalar";
        let value = (index as u64).to_le_bytes();
        hasher.update(&[1]);
        hasher.update(&(label.len() as u64).to_le_bytes());
        hasher.update(label);
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(&value);
    }
    hasher.finalize()
}

fn buffered_fields(domain: &[u8], fields: usize) -> [u8; 32] {
    let mut hasher = SemanticHasher::new(domain);
    for index in 0..fields {
        hasher.u64(b"scalar", index as u64);
    }
    hasher.finish().0
}

fn legacy_fnv_fields(domain: &[u8], fields: usize) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in domain
        .iter()
        .copied()
        .chain((0..fields).flat_map(|index| (index as u64).to_le_bytes()))
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn metadata_benchmarks(criterion: &mut Criterion) {
    assert_eq!(
        naive_fields(b"bench/tiny", 8).as_bytes(),
        &buffered_fields(b"bench/tiny", 8)
    );
    for (name, fields) in [("tiny_metadata", 8usize), ("graph_ir_metadata", 256)] {
        let encoded_bytes = fields * (1 + 8 + 6 + 8 + 8);
        let mut group = criterion.benchmark_group(name);
        group
            .throughput(Throughput::Bytes(encoded_bytes as u64))
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2))
            .sample_size(50);
        group.bench_with_input(
            BenchmarkId::new("blake3_naive_updates", fields),
            &fields,
            |b, &n| b.iter(|| black_box(naive_fields(black_box(name.as_bytes()), black_box(n)))),
        );
        group.bench_with_input(
            BenchmarkId::new("blake3_buffered_frames", fields),
            &fields,
            |b, &n| b.iter(|| black_box(buffered_fields(black_box(name.as_bytes()), black_box(n)))),
        );
        group.bench_with_input(
            BenchmarkId::new("legacy_fnv_baseline", fields),
            &fields,
            |b, &n| {
                b.iter(|| black_box(legacy_fnv_fields(black_box(name.as_bytes()), black_box(n))))
            },
        );
        group.finish();
    }
}

fn constant_benchmarks(criterion: &mut Criterion) {
    let constant = deterministic_bytes(FIVE_MIB);
    let mut group = criterion.benchmark_group("dense_constant_5mib");
    group
        .throughput(Throughput::Bytes(FIVE_MIB as u64))
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(20);
    group.bench_function("blake3_builtin_simd", |b| {
        b.iter(|| black_box(blake3::hash(black_box(&constant))))
    });
    group.bench_function("blake3_framed_direct_stream", |b| {
        b.iter(|| {
            let mut hasher = SemanticHasher::new(b"bench/dense-constant");
            hasher.field(b"constant", black_box(&constant));
            black_box(hasher.finish())
        })
    });
    group.bench_function("legacy_fnv_baseline", |b| {
        b.iter(|| {
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for &byte in black_box(&constant).iter() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            black_box(hash)
        })
    });
    group.finish();

    let mut group = criterion.benchmark_group("routed_moe_chunk_stream");
    group
        .throughput(Throughput::Bytes((FIVE_MIB * MOE_BANK_CHUNKS) as u64))
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(10);
    group.bench_function("blake3_builtin_chunk_stream", |b| {
        b.iter(|| {
            let mut hasher = blake3::Hasher::new();
            for index in 0..MOE_BANK_CHUNKS {
                hasher.update(&(index as u64).to_le_bytes());
                hasher.update(black_box(&constant));
            }
            black_box(hasher.finalize())
        })
    });
    group.bench_function("blake3_framed_chunk_stream", |b| {
        b.iter(|| {
            let mut hasher = SemanticHasher::new(b"bench/moe-bank");
            for index in 0..MOE_BANK_CHUNKS {
                hasher.u64(b"expert-bank-index", index as u64);
                hasher.field(b"expert-bank", black_box(&constant));
            }
            black_box(hasher.finish())
        })
    });
    group.finish();
}

fn hashing(criterion: &mut Criterion) {
    eprintln!(
        "BLAKE3 selected platform: {:?}, SIMD degree {}",
        blake3::platform::Platform::detect(),
        blake3::platform::Platform::detect().simd_degree()
    );
    metadata_benchmarks(criterion);
    constant_benchmarks(criterion);
}

criterion_group!(benches, hashing);
criterion_main!(benches);
