use std::future::Future;
use std::hint::black_box;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Wake, Waker};

use criterion::{criterion_group, criterion_main, Criterion};
use ggml_npu::runtime::async_runtime::*;
use ggml_npu::runtime::Deadline;

struct BenchWait;

impl AtomicWait for BenchWait {
    fn wait_u32(
        &self,
        state: &AtomicU32,
        expected: u32,
        _deadline: Deadline,
    ) -> Result<WaitChange, WaitError> {
        Ok(if state.load(Ordering::Acquire) == expected {
            WaitChange::Spurious
        } else {
            WaitChange::Changed
        })
    }

    fn wake_u32(&self, _state: &AtomicU32, _count: WakeCount) -> Result<(), WaitError> {
        Ok(())
    }
}

struct BenchClock;

impl MonotonicClock for BenchClock {
    fn now_ns(&self) -> Result<u64, WaitError> {
        Ok(0)
    }
}

#[derive(Default)]
struct CountWake(AtomicUsize);

impl Wake for CountWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn ready(value: u64) -> Completion {
    let id = OperationId::new(value).unwrap();
    let epoch = DeviceEpoch::new(1).unwrap();
    CompletionControl::ready(Ok(CompletionInfo {
        operation_id: id,
        device_epoch: epoch,
        queue_sequence: value,
        progress: Progress::default(),
    }))
}

fn async_runtime(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("async_runtime");
    group.bench_function("completion_create_ready", |bencher| {
        bencher.iter(|| black_box(ready(1)))
    });

    let mut completion = ready(2);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    group.bench_function("poll_ready", |bencher| {
        bencher.iter(|| black_box(Pin::new(&mut completion).poll(&mut context)))
    });

    let wake = Arc::new(CountWake::default());
    let waker = Waker::from(Arc::clone(&wake));
    group.bench_function("wake_by_ref", |bencher| {
        bencher.iter(|| {
            waker.wake_by_ref();
            black_box(wake.0.load(Ordering::Relaxed))
        })
    });

    group.bench_function("spawn_and_run", |bencher| {
        bencher.iter(|| {
            let executor = CallerExecutor::new(1);
            let _handle = executor.spawn(async {}).ok().unwrap();
            black_box(executor.run_until_stalled(1))
        })
    });

    let wait = WaitContext::new(
        Arc::new(BenchWait),
        Arc::new(BenchClock),
        Arc::new(NoProgress),
    );
    let completions = [ready(3), ready(4), ready(5), ready(6)];
    group.bench_function("wait_all_four_ready", |bencher| {
        bencher.iter(|| {
            black_box(wait_many(&completions, WaitMode::All, &wait, Deadline::Infinite).unwrap())
        })
    });

    let executor = CallerExecutor::new(1);
    group.bench_function("block_on_ready", |bencher| {
        bencher.iter(|| {
            black_box(
                executor
                    .block_on(async { 42u64 }, &wait, Deadline::Infinite)
                    .unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, async_runtime);
criterion_main!(benches);
