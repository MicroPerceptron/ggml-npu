//! One-process-per-sample codec comparison for native NPU artifacts.
//!
//! The suite mode deliberately launches a fresh child for every codec/artifact pair so startup
//! latency and `VmHWM` are not polluted by a previously tested implementation. Large artifacts are
//! referenced in place and are never copied into the repository.

use std::env;
use std::ffi::c_void;
use std::fs;
use std::hint::black_box;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use libloading::Library;
use serde_json::{Value, json};
use structured_zstd::decoding::StreamingDecoder;
use structured_zstd::encoding::{CompressionLevel, compress_to_vec};

type CompressBound = unsafe extern "C" fn(usize) -> usize;
type Compress = unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize, i32) -> usize;
type Decompress = unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize) -> usize;
type IsError = unsafe extern "C" fn(usize) -> u32;
type VersionNumber = unsafe extern "C" fn() -> u32;

struct SystemZstd {
    _library: Library,
    compress_bound: CompressBound,
    compress: Compress,
    decompress: Decompress,
    is_error: IsError,
    version: u32,
}

fn load_system_zstd() -> Result<SystemZstd, String> {
    let library = unsafe { Library::new("libzstd.so.1") }.map_err(|error| error.to_string())?;
    let compress_bound = *unsafe { library.get::<CompressBound>(b"ZSTD_compressBound\0") }
        .map_err(|error| error.to_string())?;
    let compress = *unsafe { library.get::<Compress>(b"ZSTD_compress\0") }
        .map_err(|error| error.to_string())?;
    let decompress = *unsafe { library.get::<Decompress>(b"ZSTD_decompress\0") }
        .map_err(|error| error.to_string())?;
    let is_error =
        *unsafe { library.get::<IsError>(b"ZSTD_isError\0") }.map_err(|error| error.to_string())?;
    let version_number = *unsafe { library.get::<VersionNumber>(b"ZSTD_versionNumber\0") }
        .map_err(|error| error.to_string())?;
    let version = unsafe { version_number() };
    Ok(SystemZstd {
        _library: library,
        compress_bound,
        compress,
        decompress,
        is_error,
        version,
    })
}

fn system_zstd() -> Result<&'static SystemZstd, String> {
    static API: OnceLock<Result<SystemZstd, String>> = OnceLock::new();
    API.get_or_init(load_system_zstd)
        .as_ref()
        .map_err(Clone::clone)
}

fn system_compress(input: &[u8], level: i32) -> Result<Vec<u8>, String> {
    let api = system_zstd()?;
    let bound = unsafe { (api.compress_bound)(input.len()) };
    let mut output = vec![0; bound];
    let written = unsafe {
        (api.compress)(
            output.as_mut_ptr().cast(),
            output.len(),
            input.as_ptr().cast(),
            input.len(),
            level,
        )
    };
    if unsafe { (api.is_error)(written) } != 0 || written > output.len() {
        return Err("system Zstandard compression failed".into());
    }
    output.truncate(written);
    Ok(output)
}

fn system_decompress(input: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let api = system_zstd()?;
    let mut output = vec![0; expected];
    let written = unsafe {
        (api.decompress)(
            output.as_mut_ptr().cast(),
            output.len(),
            input.as_ptr().cast(),
            input.len(),
        )
    };
    if unsafe { (api.is_error)(written) } != 0 || written != expected {
        return Err("system Zstandard decompression failed".into());
    }
    Ok(output)
}

fn candidates() -> Vec<&'static str> {
    let mut candidates = vec!["zstd-system--3", "zstd-system-1"];
    #[cfg(feature = "bundled-zstd")]
    candidates.extend(["zstd-bundled--3", "zstd-bundled-1"]);
    candidates.extend([
        "structured-zstd--3",
        "structured-zstd-1",
        "zrip--3",
        "zrip-1",
        "lz4-flex-reference",
    ]);
    if let Ok(filter) = env::var("GGML_NPU_CODEC_CANDIDATES") {
        candidates.retain(|candidate| filter.split(',').any(|selected| selected == *candidate));
    }
    candidates
}

fn compress(codec: &str, input: &[u8]) -> Result<Vec<u8>, String> {
    match codec {
        "zstd-system--3" => system_compress(input, -3),
        "zstd-system-1" => system_compress(input, 1),
        #[cfg(feature = "bundled-zstd")]
        "zstd-bundled--3" => zstd::bulk::compress(input, -3).map_err(|error| error.to_string()),
        #[cfg(feature = "bundled-zstd")]
        "zstd-bundled-1" => zstd::bulk::compress(input, 1).map_err(|error| error.to_string()),
        "structured-zstd--3" => Ok(compress_to_vec(input, CompressionLevel::from_level(-3))),
        "structured-zstd-1" => Ok(compress_to_vec(input, CompressionLevel::from_level(1))),
        "zrip--3" => zrip::compress(input, -3).map_err(|error| error.to_string()),
        "zrip-1" => zrip::compress(input, 1).map_err(|error| error.to_string()),
        "lz4-flex-reference" => Ok(lz4_flex::compress_prepend_size(input)),
        _ => Err(format!("unknown codec {codec}")),
    }
}

fn decompress(codec: &str, input: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    match codec {
        "zstd-system--3" | "zstd-system-1" => system_decompress(input, expected),
        #[cfg(feature = "bundled-zstd")]
        "zstd-bundled--3" | "zstd-bundled-1" => {
            zstd::bulk::decompress(input, expected).map_err(|error| error.to_string())
        }
        "structured-zstd--3" | "structured-zstd-1" => {
            let mut decoder = StreamingDecoder::new(input).map_err(|error| error.to_string())?;
            let mut output = Vec::with_capacity(expected);
            decoder
                .read_to_end(&mut output)
                .map_err(|error| error.to_string())?;
            Ok(output)
        }
        "zrip--3" | "zrip-1" => zrip::decompress(input).map_err(|error| error.to_string()),
        "lz4-flex-reference" => {
            lz4_flex::decompress_size_prepended(input).map_err(|error| error.to_string())
        }
        _ => Err(format!("unknown codec {codec}")),
    }
}

fn vm_hwm_kib() -> u64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")?
                    .split_ascii_whitespace()
                    .next()?
                    .parse()
                    .ok()
            })
        })
        .unwrap_or(0)
}

fn cpu_model() -> Option<String> {
    fs::read_to_string("/proc/cpuinfo")
        .ok()?
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name.trim() == "model name").then(|| value.trim().to_owned())
        })
}

fn process_status_field(name: &str) -> Option<String> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            let (field, value) = line.split_once(':')?;
            (field == name).then(|| value.trim().to_owned())
        })
}

fn cpu_governor(cpu: usize) -> Option<String> {
    fs::read_to_string(format!(
        "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
    ))
    .ok()
    .map(|value| value.trim().to_owned())
}

fn cpu_frequency_khz(cpu: usize, field: &str) -> Option<u64> {
    fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{field}"))
        .ok()?
        .trim()
        .parse()
        .ok()
}

fn benchmark_cpu_state() -> Value {
    let state = |cpu| {
        json!({
            "cpu": cpu,
            "governor": cpu_governor(cpu),
            "min_khz": cpu_frequency_khz(cpu, "scaling_min_freq"),
            "max_khz": cpu_frequency_khz(cpu, "scaling_max_freq"),
            "current_khz": cpu_frequency_khz(cpu, "scaling_cur_freq"),
        })
    };
    json!([state(2), state(3)])
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn runtime_cpu_features() -> Value {
    json!({
        "sse2": std::arch::is_x86_feature_detected!("sse2"),
        "sse4_2": std::arch::is_x86_feature_detected!("sse4.2"),
        "avx": std::arch::is_x86_feature_detected!("avx"),
        "avx2": std::arch::is_x86_feature_detected!("avx2"),
        "bmi1": std::arch::is_x86_feature_detected!("bmi1"),
        "bmi2": std::arch::is_x86_feature_detected!("bmi2"),
        "f16c": std::arch::is_x86_feature_detected!("f16c"),
        "fma": std::arch::is_x86_feature_detected!("fma"),
        "lzcnt": std::arch::is_x86_feature_detected!("lzcnt"),
        "movbe": std::arch::is_x86_feature_detected!("movbe"),
        "popcnt": std::arch::is_x86_feature_detected!("popcnt"),
        "xsave": std::arch::is_x86_feature_detected!("xsave"),
    })
}

#[cfg(target_arch = "aarch64")]
fn runtime_cpu_features() -> Value {
    json!({
        "neon": std::arch::is_aarch64_feature_detected!("neon"),
        "sve": std::arch::is_aarch64_feature_detected!("sve"),
        "sve2": std::arch::is_aarch64_feature_detected!("sve2"),
    })
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
fn runtime_cpu_features() -> Value {
    json!({})
}

fn system_library_path() -> Option<String> {
    let maps = fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines()
        .filter_map(|line| line.split_ascii_whitespace().last())
        .find(|path| path.contains("/libzstd.so"))
        .map(str::to_owned)
}

#[cfg(feature = "zrip-simd")]
fn zrip_runtime_tier() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("cmpxchg16b")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("lzcnt")
            && std::arch::is_x86_feature_detected!("movbe")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("xsave")
        {
            return "fearless_simd-avx2";
        }
        if std::arch::is_x86_feature_detected!("sse4.2")
            && std::arch::is_x86_feature_detected!("cmpxchg16b")
            && std::arch::is_x86_feature_detected!("popcnt")
        {
            return "fearless_simd-sse4.2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return "fearless_simd-neon";
    }
    "fearless_simd-fallback"
}

#[cfg(not(feature = "zrip-simd"))]
fn zrip_runtime_tier() -> &'static str {
    "scalar-feature-build"
}

fn build_metadata(executable: &Path) -> Value {
    let system = system_zstd().ok();
    #[cfg(feature = "bundled-zstd")]
    let bundled_version = Some(zstd::zstd_safe::version_number());
    #[cfg(not(feature = "bundled-zstd"))]
    let bundled_version: Option<u32> = None;
    json!({
        "variant": env::var("GGML_NPU_BENCH_VARIANT").unwrap_or_else(|_| "manual".into()),
        "cargo_features": {
            "bundled_zstd": cfg!(feature = "bundled-zstd"),
            "structured_simd": cfg!(feature = "structured-simd"),
            "zrip_simd": cfg!(feature = "zrip-simd"),
            "lz4_safe": cfg!(feature = "lz4-safe"),
            "zstd_no_asm": cfg!(feature = "zstd-no-asm"),
            "zstd_fat_lto": cfg!(feature = "zstd-fat-lto"),
            "zstd_thin_lto": cfg!(feature = "zstd-thin-lto"),
            "zstd_thin_minified": cfg!(feature = "zstd-thin-minified"),
            "zstd_pkg_config": cfg!(feature = "zstd-pkg-config"),
        },
        "compiler": {
            "rustc": env::var("GGML_NPU_BENCH_RUSTC").unwrap_or_else(|_| "unreported".into()),
            "cargo": env::var("GGML_NPU_BENCH_CARGO").unwrap_or_else(|_| "unreported".into()),
            "cc": env::var("GGML_NPU_BENCH_CC").unwrap_or_else(|_| "unreported".into()),
            "rustflags": env::var("GGML_NPU_BENCH_RUSTFLAGS").unwrap_or_default(),
            "cflags": env::var("GGML_NPU_BENCH_CFLAGS").unwrap_or_default(),
            "profile": "release opt-level=3 lto=thin codegen-units=1",
        },
        "target": {
            "triple": env::var("GGML_NPU_BENCH_TARGET").unwrap_or_else(|_| format!("{}-unknown-linux-gnu", env::consts::ARCH)),
            "arch": env::consts::ARCH,
            "os": env::consts::OS,
            "compile_time_features": {
                "sse2": cfg!(target_feature = "sse2"),
                "sse4_2": cfg!(target_feature = "sse4.2"),
                "avx": cfg!(target_feature = "avx"),
                "avx2": cfg!(target_feature = "avx2"),
                "bmi2": cfg!(target_feature = "bmi2"),
                "neon": cfg!(target_feature = "neon"),
            },
            "cpu_model": cpu_model(),
            "cpu_affinity": process_status_field("Cpus_allowed_list"),
            "requested_cpu_affinity": env::var("GGML_NPU_BENCH_AFFINITY").unwrap_or_else(|_| "unreported".into()),
            "benchmark_cpu_state": benchmark_cpu_state(),
            "runtime_features": runtime_cpu_features(),
        },
        "execution": {
            "mode": env::var("GGML_NPU_BENCH_EXECUTION_MODE").unwrap_or_else(|_| "unreported".into()),
            "global_lock": env::var("GGML_NPU_BENCH_LOCK").unwrap_or_else(|_| "unreported".into()),
            "lock_mode": env::var("GGML_NPU_BENCH_LOCK_MODE").unwrap_or_else(|_| "unreported".into()),
        },
        "binary_bytes": fs::metadata(executable).map(|metadata| metadata.len()).unwrap_or(0),
        "providers": {
            "system_zstd": {
                "crate_features": "none; libloading stable C ABI",
                "runtime_version": system.map(|api| api.version),
                "mapped_library": system_library_path(),
            },
            "bundled_zstd": {
                "crate": cfg!(feature = "bundled-zstd").then_some("zstd 0.13.3 / zstd-safe 7.2.4 / zstd-sys 2.0.16+zstd.1.5.7"),
                "runtime_version": bundled_version,
                "assembly": cfg!(feature = "bundled-zstd") && !cfg!(feature = "zstd-no-asm"),
                "portable_runtime_dispatch": "BMI2 on GCC/Clang x86; x86_64 HUF assembly unless zstd-no-asm; SSE2/NEON selected by C target macros",
            },
            "structured_zstd": {
                "crate": "structured-zstd 0.0.48",
                "active_decode_kernel": structured_zstd::active_cpu_kernel_name(),
                "encoder_acceleration": "BMI2 Huffman bitstream dispatch on x86 and SVE2-tagged histogram path on aarch64 are independent of kernel_* features",
                "decode_runtime_dispatch": cfg!(feature = "structured-simd"),
                "vbmi2_enabled": false,
            },
            "zrip": {
                "crate": "zrip 0.8.3 / fearless_simd 0.5.0",
                "active_decode_kernel": zrip_runtime_tier(),
                "encoder_explicit_simd": false,
                "runtime_dispatch": cfg!(feature = "zrip-simd"),
            },
            "lz4_flex": {
                "crate": "lz4_flex 0.13.1",
                "safe_paths": cfg!(feature = "lz4-safe"),
                "checked_decode": cfg!(feature = "lz4-safe"),
                "runtime_dispatch": false,
                "compile_time_avx_small_copy": cfg!(target_feature = "avx"),
            },
        },
    })
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn mb_per_second(bytes: usize, elapsed: Duration) -> f64 {
    bytes as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64()
}

fn one(
    codec: &str,
    path: &Path,
    iterations: usize,
    process_started: Instant,
) -> Result<Value, String> {
    let frequency_start = benchmark_cpu_state();
    let input = fs::read(path).map_err(|error| error.to_string())?;
    let rss_after_input_kib = vm_hwm_kib();

    let started = Instant::now();
    let first_compressed = compress(codec, black_box(&input))?;
    let startup_compress = started.elapsed();
    let started = Instant::now();
    let first_output = decompress(codec, black_box(&first_compressed), input.len())?;
    let startup_decompress = started.elapsed();
    if first_output != input {
        return Err("round-trip mismatch".into());
    }
    let cold_main_to_first_roundtrip = process_started.elapsed();

    let mut compression_samples = Vec::with_capacity(iterations);
    let mut compressed = first_compressed;
    for _ in 0..iterations {
        let started = Instant::now();
        compressed = compress(codec, black_box(&input))?;
        compression_samples.push(started.elapsed());
    }

    let mut decompression_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let output = decompress(codec, black_box(&compressed), input.len())?;
        decompression_samples.push(started.elapsed());
        if output != input {
            return Err("repeated round-trip mismatch".into());
        }
        black_box(output);
    }

    let compression_median = median(&mut compression_samples);
    let decompression_median = median(&mut decompression_samples);
    let peak_rss_kib = vm_hwm_kib();
    let frequency_end = benchmark_cpu_state();
    Ok(json!({
        "codec": codec,
        "artifact": path.file_name().and_then(|name| name.to_str()).unwrap_or("unknown"),
        "input_bytes": input.len(),
        "compressed_bytes": compressed.len(),
        "ratio_uncompressed_over_compressed": input.len() as f64 / compressed.len() as f64,
        "startup_compress_ns": startup_compress.as_nanos() as u64,
        "startup_decompress_ns": startup_decompress.as_nanos() as u64,
        "cold_main_to_first_roundtrip_ns": cold_main_to_first_roundtrip.as_nanos() as u64,
        "steady_compress_median_ns": compression_median.as_nanos() as u64,
        "steady_decompress_median_ns": decompression_median.as_nanos() as u64,
        "steady_compress_mib_s": mb_per_second(input.len(), compression_median),
        "steady_decompress_mib_s": mb_per_second(input.len(), decompression_median),
        "rss_after_input_kib": rss_after_input_kib,
        "peak_rss_kib": peak_rss_kib,
        "peak_scratch_delta_kib": peak_rss_kib.saturating_sub(rss_after_input_kib),
        "cpu_state_start": frequency_start,
        "cpu_state_end": frequency_end,
        "iterations": iterations,
    }))
}

fn iterations_for(path: &Path) -> usize {
    match fs::metadata(path).map(|metadata| metadata.len()) {
        Ok(bytes) if bytes >= 64 * 1024 * 1024 => 5,
        Ok(bytes) if bytes >= 8 * 1024 * 1024 => 7,
        _ => 15,
    }
}

fn summarize(candidates: &[&str], samples: &[Value]) -> Value {
    let summaries: Vec<Value> = candidates
        .iter()
        .map(|codec| {
            let selected: Vec<&Value> = samples
                .iter()
                .filter(|sample| sample["codec"].as_str() == Some(codec))
                .collect();
            let input_bytes: u64 = selected
                .iter()
                .filter_map(|sample| sample["input_bytes"].as_u64())
                .sum();
            let compressed_bytes: u64 = selected
                .iter()
                .filter_map(|sample| sample["compressed_bytes"].as_u64())
                .sum();
            let compression_ns: u64 = selected
                .iter()
                .filter_map(|sample| sample["steady_compress_median_ns"].as_u64())
                .sum();
            let decompression_ns: u64 = selected
                .iter()
                .filter_map(|sample| sample["steady_decompress_median_ns"].as_u64())
                .sum();
            let peak_scratch_kib = selected
                .iter()
                .filter_map(|sample| sample["peak_scratch_delta_kib"].as_u64())
                .max()
                .unwrap_or(0);
            let throughput = |elapsed_ns: u64| {
                if elapsed_ns == 0 {
                    0.0
                } else {
                    input_bytes as f64 / (1024.0 * 1024.0) / (elapsed_ns as f64 / 1e9)
                }
            };
            json!({
                "codec": codec,
                "artifact_count": selected.len(),
                "input_bytes": input_bytes,
                "compressed_bytes": compressed_bytes,
                "aggregate_ratio": input_bytes as f64 / compressed_bytes as f64,
                "aggregate_compress_mib_s": throughput(compression_ns),
                "aggregate_decompress_mib_s": throughput(decompression_ns),
                "max_peak_scratch_delta_kib": peak_scratch_kib,
            })
        })
        .collect();
    Value::Array(summaries)
}

fn suite(output: &Path, artifacts: &[PathBuf]) -> Result<(), String> {
    let executable = env::current_exe().map_err(|error| error.to_string())?;
    let candidates = candidates();
    let mut samples = Vec::new();
    for artifact in artifacts {
        for codec in &candidates {
            let child_started = Instant::now();
            let child = Command::new(&executable)
                .arg("--one")
                .arg(codec)
                .arg(artifact)
                .arg(iterations_for(artifact).to_string())
                .output()
                .map_err(|error| error.to_string())?;
            let child_wall = child_started.elapsed();
            if !child.status.success() {
                return Err(format!(
                    "{codec} failed for {}: {}",
                    artifact.display(),
                    String::from_utf8_lossy(&child.stderr)
                ));
            }
            let sample: Value =
                serde_json::from_slice(&child.stdout).map_err(|error| error.to_string())?;
            let mut sample = sample;
            sample["fresh_process_wall_ns"] = json!(child_wall.as_nanos() as u64);
            eprintln!(
                "{} {} ratio={:.3} enc={:.1} MiB/s dec={:.1} MiB/s",
                sample["codec"].as_str().unwrap_or("?"),
                sample["artifact"].as_str().unwrap_or("?"),
                sample["ratio_uncompressed_over_compressed"]
                    .as_f64()
                    .unwrap_or(0.0),
                sample["steady_compress_mib_s"].as_f64().unwrap_or(0.0),
                sample["steady_decompress_mib_s"].as_f64().unwrap_or(0.0),
            );
            samples.push(sample);
        }
    }
    let result = json!({
        "schema": 2,
        "method": "fresh process per codec/artifact; first-operation compression/decompression timings include provider initialization; cold-main timing includes argument parse and input read; median warm iterations; VmHWM peak delta after input load",
        "build": build_metadata(&executable),
        "candidates": candidates,
        "summary": summarize(&candidates, &samples),
        "samples": samples,
    });
    let encoded = serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?;
    fs::write(output, encoded).map_err(|error| error.to_string())
}

fn usage() -> String {
    format!(
        "usage:\n  cache-codec-bench --one <codec> <artifact> <iterations>\n  cache-codec-bench --suite <output.json> <artifact>...\ncodecs: {}",
        candidates().join(", ")
    )
}

fn run(process_started: Instant) -> Result<(), String> {
    let mut args = env::args_os();
    let _program = args.next();
    match args
        .next()
        .and_then(|arg| arg.into_string().ok())
        .as_deref()
    {
        Some("--one") => {
            let codec = args
                .next()
                .and_then(|arg| arg.into_string().ok())
                .ok_or_else(usage)?;
            let path = args.next().map(PathBuf::from).ok_or_else(usage)?;
            let iterations = args
                .next()
                .and_then(|arg| arg.into_string().ok())
                .and_then(|arg| arg.parse().ok())
                .ok_or_else(usage)?;
            let value = one(&codec, &path, iterations, process_started)?;
            println!(
                "{}",
                serde_json::to_string(&value).map_err(|error| error.to_string())?
            );
            Ok(())
        }
        Some("--suite") => {
            let output = args.next().map(PathBuf::from).ok_or_else(usage)?;
            let artifacts: Vec<PathBuf> = args.map(PathBuf::from).collect();
            if artifacts.is_empty() {
                return Err(usage());
            }
            suite(&output, &artifacts)
        }
        _ => Err(usage()),
    }
}

fn main() {
    let process_started = Instant::now();
    if let Err(error) = run(process_started) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
