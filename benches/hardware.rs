use std::env;
use std::ffi::{c_char, c_int, c_void, CString};
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, Criterion, SamplingMode, Throughput,
};

const ELEMENTS: usize = 32;
const DIRECT: u32 = 0;
const LEVEL_ZERO: u32 = 1;
const RTLD_NOW: c_int = 2;

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

type Create = unsafe extern "C" fn(
    u32,
    *const u8,
    usize,
    *const *const f32,
    *const usize,
    usize,
) -> *mut c_void;
type Execute = unsafe extern "C" fn(*mut c_void) -> c_int;
type CopyOutput = unsafe extern "C" fn(*const c_void, *mut f32, usize) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void);

struct Api {
    library: *mut c_void,
    create: Create,
    execute: Execute,
    copy_output: CopyOutput,
    destroy: Destroy,
}

impl Api {
    fn load() -> Self {
        let path = env::var_os("GGML_NPU_BENCH_BACKEND")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target/release/libggml_npu.so"));
        let path = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        // SAFETY: path is NUL-terminated and the handle remains open for all loaded symbols.
        let library = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        if library.is_null() {
            // SAFETY: dlerror is a process-local diagnostic string.
            let error = unsafe { dlerror() };
            let message = if error.is_null() {
                "unknown dlopen error".into()
            } else {
                // SAFETY: non-null dlerror result is NUL-terminated.
                unsafe { std::ffi::CStr::from_ptr(error) }
                    .to_string_lossy()
                    .into_owned()
            };
            panic!("failed to load benchmark backend: {message}");
        }
        unsafe fn symbol<T: Copy>(library: *mut c_void, name: &std::ffi::CStr) -> T {
            let address = dlsym(library, name.as_ptr());
            assert!(
                !address.is_null(),
                "missing symbol {}",
                name.to_string_lossy()
            );
            std::mem::transmute_copy(&address)
        }
        // SAFETY: each symbol is exported with the matching C signature.
        unsafe {
            Self {
                library,
                create: symbol(library, c"ggml_npu_hardware_fixture_create"),
                execute: symbol(library, c"ggml_npu_hardware_fixture_execute"),
                copy_output: symbol(library, c"ggml_npu_hardware_fixture_copy_output"),
                destroy: symbol(library, c"ggml_npu_hardware_fixture_destroy"),
            }
        }
    }
}

impl Drop for Api {
    fn drop(&mut self) {
        // SAFETY: handle came from dlopen and all fixtures have already been dropped.
        unsafe { dlclose(self.library) };
    }
}

struct Fixture<'a> {
    api: &'a Api,
    pointer: *mut c_void,
}

impl<'a> Fixture<'a> {
    fn new(api: &'a Api, kind: u32, artifact: &[u8], input0: &[f32], input1: &[f32]) -> Self {
        let pointers = [input0.as_ptr(), input1.as_ptr()];
        let lengths = [input0.len(), input1.len()];
        // SAFETY: all slices remain live for construction and the fixture copies/stages inputs.
        let pointer = unsafe {
            (api.create)(
                kind,
                artifact.as_ptr(),
                artifact.len(),
                pointers.as_ptr(),
                lengths.as_ptr(),
                pointers.len(),
            )
        };
        assert!(
            !pointer.is_null(),
            "fixture creation failed for kind {kind}"
        );
        Self { api, pointer }
    }

    fn execute(&mut self) {
        // SAFETY: pointer is a live fixture uniquely borrowed through self.
        assert_eq!(unsafe { (self.api.execute)(self.pointer) }, 0);
    }

    fn output(&self) -> [f32; ELEMENTS] {
        let mut output = [0.0f32; ELEMENTS];
        // SAFETY: pointer is live and output capacity is exact.
        assert_eq!(
            unsafe { (self.api.copy_output)(self.pointer, output.as_mut_ptr(), output.len()) },
            ELEMENTS as c_int
        );
        output
    }
}

impl Drop for Fixture<'_> {
    fn drop(&mut self) {
        // SAFETY: each fixture is destroyed exactly once.
        unsafe { (self.api.destroy)(self.pointer) };
    }
}

#[derive(Debug)]
struct Resources {
    fds: usize,
    rss_kib: Option<u64>,
    device_bytes: Option<u64>,
}

fn resources() -> Resources {
    let fds = fs::read_dir("/proc/self/fd")
        .map(|entries| entries.count())
        .unwrap_or(0);
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let rss_kib = status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .and_then(|line| line.split_ascii_whitespace().nth(1))
        .and_then(|value| value.parse().ok());
    let device_bytes = fs::read_to_string("/sys/class/accel/accel0/device/npu_memory_utilization")
        .ok()
        .and_then(|value| value.trim().parse().ok());
    Resources {
        fds,
        rss_kib,
        device_bytes,
    }
}

fn inputs() -> ([f32; ELEMENTS], [f32; ELEMENTS]) {
    (
        core::array::from_fn(|index| index as f32 * 0.125 - 1.5),
        core::array::from_fn(|index| ((index * 7) % 17) as f32 * 0.25 - 2.0),
    )
}

fn expected(input0: &[f32; ELEMENTS], input1: &[f32; ELEMENTS]) -> [f32; ELEMENTS] {
    core::array::from_fn(|index| input0[index] + input1[index])
}

fn verify(output: &[f32; ELEMENTS], oracle: &[f32; ELEMENTS]) {
    for (index, (actual, expected)) in output.iter().zip(oracle).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "output mismatch at {index}"
        );
    }
}

fn configure(group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(2));
    group.measurement_time(Duration::from_millis(5));
    group.throughput(Throughput::Elements(ELEMENTS as u64));
}

fn timed_execute(fixture: &mut Fixture<'_>, readback: bool) -> Duration {
    let start = Instant::now();
    fixture.execute();
    if readback {
        black_box(fixture.output());
    }
    start.elapsed()
}

fn hardware(criterion: &mut Criterion) {
    let start = resources();
    let api = Api::load();
    let artifact_path = env::var_os("GGML_NPU_TEST_ARTIFACT")
        .map(PathBuf::from)
        .expect("set GGML_NPU_TEST_ARTIFACT to the bounded Intel NPU test artifact");
    let artifact = fs::read(&artifact_path).expect("missing proven tiny artifact");
    assert!(artifact.len() <= 1024 * 1024);
    let (input0, input1) = inputs();
    let oracle = expected(&input0, &input1);

    for kind in [DIRECT, LEVEL_ZERO] {
        let mut fixture = Fixture::new(&api, kind, &artifact, &input0, &input1);
        fixture.execute();
        verify(&fixture.output(), &oracle);
    }
    let steady = resources();

    let mut group = criterion.benchmark_group("ivpu_tiny_setup_load_bind");
    configure(&mut group);
    for (name, kind) in [("direct", DIRECT), ("level_zero", LEVEL_ZERO)] {
        group.bench_function(name, |bencher| {
            bencher.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| {
                        let start = Instant::now();
                        drop(Fixture::new(&api, kind, &artifact, &input0, &input1));
                        start.elapsed()
                    })
                    .sum()
            });
        });
    }
    group.finish();

    let mut group = criterion.benchmark_group("ivpu_tiny_cold_setup_execute");
    configure(&mut group);
    for (name, kind) in [("direct", DIRECT), ("level_zero", LEVEL_ZERO)] {
        group.bench_function(name, |bencher| {
            bencher.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| {
                        let start = Instant::now();
                        let mut fixture = Fixture::new(&api, kind, &artifact, &input0, &input1);
                        fixture.execute();
                        verify(&fixture.output(), &oracle);
                        start.elapsed()
                    })
                    .sum()
            });
        });
    }
    group.finish();

    let mut direct = Fixture::new(&api, DIRECT, &artifact, &input0, &input1);
    let mut level_zero = Fixture::new(&api, LEVEL_ZERO, &artifact, &input0, &input1);
    for (group_name, readback) in [
        ("ivpu_tiny_warm_submit_wait", false),
        ("ivpu_tiny_warm_execute_readback", true),
    ] {
        let mut group = criterion.benchmark_group(group_name);
        configure(&mut group);
        group.bench_function("direct", |bencher| {
            bencher.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| timed_execute(&mut direct, readback))
                    .sum()
            });
        });
        verify(&direct.output(), &oracle);
        group.bench_function("level_zero", |bencher| {
            bencher.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| timed_execute(&mut level_zero, readback))
                    .sum()
            });
        });
        verify(&level_zero.output(), &oracle);
        group.finish();
    }
    drop((direct, level_zero));

    let mut group = criterion.benchmark_group("ivpu_tiny_host_control");
    configure(&mut group);
    group.bench_function("copy_128_bytes", |bencher| {
        bencher.iter(|| {
            let mut output = [0.0f32; ELEMENTS];
            output.copy_from_slice(black_box(&oracle));
            black_box(output)
        });
    });
    group.finish();

    let after = resources();
    eprintln!("hardware benchmark resources start={start:?} steady={steady:?} after={after:?}");
    assert_eq!(after.fds, steady.fds);
    if let (Some(before), Some(after)) = (steady.device_bytes, after.device_bytes) {
        assert_eq!(after, before);
    }
    if let (Some(before), Some(after)) = (steady.rss_kib, after.rss_kib) {
        assert!(after <= before.saturating_add(16 * 1024));
    }
}

criterion_group!(hardware_benches, hardware);
criterion_main!(hardware_benches);
