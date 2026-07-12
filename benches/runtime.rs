use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use goblin::elf::{header, section_header, Elf};

extern crate alloc;
#[allow(dead_code, unused_imports)]
#[path = "../src/vendor/ivpu/cache.rs"]
mod cache;
#[allow(dead_code, unused_imports)]
#[path = "../src/identity.rs"]
mod identity;

use cache::Cache;
use identity::{ExactKeyBuilder, ExactSemanticKey};

fn graph_key(nodes: usize) -> ExactSemanticKey {
    let mut key = ExactKeyBuilder::new(b"bench/graph-ir");
    key.u64(b"nodes", nodes as u64)
        .u64(b"values", (nodes * 3) as u64);
    for node in 0..nodes {
        key.u64(b"node-index", node as u64)
            .u32(b"node-op", (node % 27) as u32)
            .u64(b"shape-k", 2048)
            .u64(b"shape-m", 512)
            .u64(b"stride", 8192);
    }
    key.finish()
}

fn identity_and_cache(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("canonical_graph_identity");
    group
        .throughput(Throughput::Elements(32))
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50);
    group.bench_function("encode_32_nodes", |b| {
        b.iter(|| black_box(graph_key(black_box(32))))
    });
    group.finish();

    let cache = Cache::new(64);
    let keys: Vec<_> = (0..64)
        .map(|index| {
            let mut key = ExactKeyBuilder::new(b"bench/cache");
            key.u64(b"index", index);
            key.finish()
        })
        .collect();
    for (index, key) in keys.iter().cloned().enumerate() {
        cache.try_get_or_insert_with(key, || index as u64).unwrap();
    }
    let miss = {
        let mut key = ExactKeyBuilder::new(b"bench/cache");
        key.u64(b"index", 65);
        key.finish()
    };
    let mut group = criterion.benchmark_group("exact_key_cache");
    group
        .throughput(Throughput::Elements(1))
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50);
    group.bench_function("hit_64_entry", |b| {
        b.iter(|| black_box(cache.get(black_box(&keys[31]))))
    });
    group.bench_function("miss_64_entry", |b| {
        b.iter(|| black_box(cache.get(black_box(&miss))))
    });
    group.bench_function("capacity_rejection", |b| {
        b.iter(|| black_box(cache.try_get_or_insert_with(miss.clone(), || 65)))
    });
    group.finish();
}

fn minimal_intel_elf() -> Vec<u8> {
    const PLATFORM: u32 = 0x8aaa_aaaf;
    let mut elf = vec![0u8; 320];
    elf[..7].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 0]);
    elf[16..18].copy_from_slice(&header::ET_REL.to_le_bytes());
    elf[18..20].copy_from_slice(&header::EM_NONE.to_le_bytes());
    elf[40..48].copy_from_slice(&64u64.to_le_bytes());
    elf[52..54].copy_from_slice(&64u16.to_le_bytes());
    elf[58..60].copy_from_slice(&64u16.to_le_bytes());
    elf[60..62].copy_from_slice(&3u16.to_le_bytes());
    elf[62..64].copy_from_slice(&2u16.to_le_bytes());
    let platform = 128;
    elf[platform + 4..platform + 8].copy_from_slice(&PLATFORM.to_le_bytes());
    elf[platform + 24..platform + 32].copy_from_slice(&256u64.to_le_bytes());
    elf[platform + 32..platform + 40].copy_from_slice(&40u64.to_le_bytes());
    elf[platform + 48..platform + 56].copy_from_slice(&8u64.to_le_bytes());
    elf[256..264].copy_from_slice(&32u64.to_le_bytes());
    elf[272..280].copy_from_slice(&1u64.to_le_bytes());
    elf[280..288].copy_from_slice(&8u64.to_le_bytes());
    elf[288..296].copy_from_slice(&4u64.to_le_bytes());
    let strings = 192;
    elf[strings + 4..strings + 8].copy_from_slice(&section_header::SHT_STRTAB.to_le_bytes());
    elf[strings + 24..strings + 32].copy_from_slice(&296u64.to_le_bytes());
    elf[strings + 32..strings + 40].copy_from_slice(&1u64.to_le_bytes());
    elf[strings + 48..strings + 56].copy_from_slice(&1u64.to_le_bytes());
    elf
}

fn artifact_parsing(criterion: &mut Criterion) {
    let artifact = minimal_intel_elf();
    let mut group = criterion.benchmark_group("artifact_goblin_parse");
    group
        .throughput(Throughput::Bytes(artifact.len() as u64))
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50);
    group.bench_with_input(
        BenchmarkId::new("elf64_zero_copy", artifact.len()),
        &artifact,
        |b, artifact| b.iter(|| black_box(Elf::parse(black_box(artifact)).unwrap())),
    );
    group.finish();
}

fn lfm_contract_metadata(criterion: &mut Criterion) {
    let operations = [
        ("rmsnorm", 2048u64, 512u64),
        ("rope", 64, 512),
        ("ssmconv", 2048, 512),
        ("moe-routing", 32, 512),
        ("flash-attention", 64, 512),
    ];
    let mut group = criterion.benchmark_group("lfm_operation_contract_metadata");
    group
        .throughput(Throughput::Elements(1))
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50);
    for (name, width, tokens) in operations {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut key = ExactKeyBuilder::new(b"bench/lfm-contract");
                key.field(b"operation", black_box(name.as_bytes()))
                    .u64(b"width", black_box(width))
                    .u64(b"tokens", black_box(tokens))
                    .u64(b"dtype", 0)
                    .u64(b"placement", 1);
                black_box(key.finish())
            })
        });
    }
    group.finish();
}

fn runtime(criterion: &mut Criterion) {
    identity_and_cache(criterion);
    artifact_parsing(criterion);
    lfm_contract_metadata(criterion);
}

criterion_group!(benches, runtime);
criterion_main!(benches);
