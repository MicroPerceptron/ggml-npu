use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

extern crate alloc;
#[allow(unused_imports)]
#[path = "../src/artifact_cache.rs"]
mod artifact_cache;
#[allow(unused_imports)]
#[path = "../src/identity.rs"]
mod identity;

use artifact_cache::{
    ArtifactKey, CacheCompatibility, CodecDescriptor, RecordHeader, RECORD_HEADER_BYTES,
};
use identity::SemanticDigest;

const CORPUS: &[&str] = &[
    "ir2-2fc11a4f22b54cfc-n3.blob",
    "ir2-25db148dace09539-n8.blob",
    "ir2-082d22694112e163-n8.blob",
    "ir2-27d892a231afd50b-n8.blob",
    "ir2-00ca97ae85b4313b-n3.blob",
];

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest([byte; 32])
}

fn compatibility() -> CacheCompatibility {
    CacheCompatibility {
        artifact_schema: 9,
        compiler: digest(1),
        pipeline: digest(2),
        toolchain: digest(3),
        target: digest(4),
        firmware: digest(5),
        runtime: digest(6),
    }
}

fn corpus() -> Vec<(String, Vec<u8>)> {
    let Some(root) = env::var_os("GGML_NPU_CODEC_CORPUS_DIR").map(PathBuf::from) else {
        eprintln!("GGML_NPU_CODEC_CORPUS_DIR is unset; using deterministic 8 MiB fallback fixture");
        return vec![(
            "deterministic-fallback".into(),
            (0..8 * 1024 * 1024)
                .map(|index| ((index * 31 + index / 4096) & 0xff) as u8)
                .collect(),
        )];
    };
    CORPUS
        .iter()
        .filter_map(|name| {
            let path = root.join(name);
            match fs::read(&path) {
                Ok(bytes) => Some(((*name).into(), bytes)),
                Err(error) => {
                    eprintln!("skip {}: {error}", path.display());
                    None
                }
            }
        })
        .collect()
}

fn benchmarks(criterion: &mut Criterion) {
    for (name, source) in corpus() {
        let compressed = zstd::bulk::compress(&source, 1).expect("compress fixture");
        let header = RecordHeader {
            artifact_schema: 9,
            key: ArtifactKey(digest(7)),
            compatibility: compatibility(),
            codec: CodecDescriptor {
                id: u32::from_le_bytes(*b"ZSTD"),
                version: 10_507,
                options: 1,
            },
            compressed_bytes: compressed.len() as u64,
            uncompressed_bytes: source.len() as u64,
            uncompressed_digest: SemanticDigest(*blake3::hash(&source).as_bytes()),
        };
        let record = header
            .encode(compressed.clone())
            .expect("encode benchmark record");
        assert_eq!(record.len(), RECORD_HEADER_BYTES + compressed.len());
        eprintln!(
            "{name}: {} -> {} bytes ({:.3}x)",
            source.len(),
            compressed.len(),
            source.len() as f64 / compressed.len() as f64
        );

        let mut group = criterion.benchmark_group(format!("artifact-cache/{name}"));
        group
            .throughput(Throughput::Bytes(source.len() as u64))
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(2))
            .sample_size(10);
        group.bench_with_input(
            BenchmarkId::new("zstd-level-1-compress", source.len()),
            &source,
            |bencher, source| {
                bencher.iter(|| {
                    black_box(zstd::bulk::compress(black_box(source), 1).expect("compress"))
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("zstd-level-1-promote", source.len()),
            &compressed,
            |bencher, compressed| {
                bencher.iter(|| {
                    let output = zstd::bulk::decompress(black_box(compressed), source.len())
                        .expect("decompress");
                    black_box(blake3::hash(&output));
                    black_box(output)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("record-header-parse", record.len()),
            &record,
            |bencher, record| {
                bencher.iter(|| black_box(RecordHeader::parse(black_box(record)).expect("parse")))
            },
        );
        group.finish();
    }
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
