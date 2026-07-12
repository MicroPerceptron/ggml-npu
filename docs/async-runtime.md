# Host-neutral native async runtime

This foundation makes Rust `Future`, `async fn`, and `.await` the first-class typed API. Explicit
polling exists below it for dynamic providers, host reactors, and the C ABI. The runtime is
`no_std + alloc`, caller-driven, bounded, and instance-local: it creates no implicit thread and
installs no process-global executor.

## Typed API

`Submitter::submit` consumes the complete `OwnedOperation`. The operation strongly owns its queue,
program, allocation-backed `BindingLease`s, and mutable per-invocation resources. A provider may
return the operation only when it proves rejection happened before acceptance. Indeterminate
acceptance returns a completion retained by `InFlightRegistry`.

```rust
async fn execute<Q>(
    queue: &Q,
    operation: Q::Operation,
) -> Result<CompletionInfo, RunError<Q::Operation>>
where
    Q: AsyncQueue,
{
    let info = queue.submit(operation)?.await?;
    Ok(info)
}
```

`Completion` implements `core::future::Future<Output = CompletionResult>`. It also supplies
strictly nonblocking `try_complete`, explicit cancellation request, blocking `wait`, and
`wait_many` in any/all modes. Wait timeout is `WaitStatus::TimedOut` or
`WaitManyOutcome::TimedOut`; it never changes operation state. Dropping any observer unregisters
that observer and never requests cancellation.

Each cloned completion receives its own waiter registration. Polling registers the current waker
and then rechecks terminal state, closing the register/complete race without a replaceable
single-observer `AtomicWaker`.

## Ownership and state

Execution completion has two independent facts:

1. logical terminal status: success, execution failure, acknowledged cancellation, or device loss;
2. provider-confirmed resource reclamation safety.

The in-flight registry retains detached operations until both are true. Device loss can fan out a
logical error immediately, while the lost epoch retains queue, program, bindings, allocations, and
invocation resources until teardown proves the device can no longer access them. Cancellation is
request/acknowledgement; successful execution may win the race.

Execution, visibility, and interop ownership are separate future chains.
`VisibilityTransition` consumes a lease and produces a lease for a generic host/device/coherent
visibility domain only after a provider completion. `OwnershipTransition` uses adapter-defined
owner marker types. Either transition quarantines its lease on failure. Foreign handle schemas,
cache maintenance, queue-family rules, and fence ownership remain adapter-owned.

## Reactor and executors

`CallerExecutor` stores at most its configured number of tasks. `spawn` rejects at capacity,
`run_until_stalled` polls a bounded caller-selected number, `block_on` drives spawned tasks and an
injected `ProgressDriver`, and `shutdown` cancels retained task futures. A task wake only schedules
that task; no poll path blocks.

`CallerDrivenBlockingExecutor` is the safe fallback for transports that expose only blocking
operations. It has a bounded queue and requires an explicit `run_one` call outside `Future::poll`.
The host port allows a future host integration to inject an autonomous bounded worker executor,
but this change does not introduce `std`, pthread policy, or hidden workers. Moving a blocking
syscall to a worker would free the caller; it would not make the transport itself faster.

`HostEnvironment` now composes three focused additions:

- `AtomicWait`, used for completion sleep/wake and eventually cache single-flight coordination;
- an instance-local reactor factory for documented pollable wake sources;
- an explicitly bounded blocking-executor factory.

The existing clock service remains the source of absolute monotonic time.

The Linux atomic service uses private rustix futex wait/wake. The Linux reactor uses an
instance-local rustix epoll descriptor and duplicates each registered pollable source for the
registration lifetime. Its eventfd-backed test source proves register, wake, drain, and unregister
without hardware. The reactor deliberately does **not** register the IVPU device fd: device-fd
readability is not job completion.

The deterministic mock supplies virtual time, scripted completions, spurious waits, wake-source
registration, and bounded caller-driven execution.

## Dynamic and C boundary

`CompletionAbiVTable` is `#[repr(C)]` and contains only:

- opaque `*mut c_void` context;
- paired `retain` and `release`;
- nonblocking `try_poll` into a primitive-only result;
- C callback `register_wake` and quiescing `unregister_wake`;
- `request_cancel`.

No Rust `Future`, trait object, `Waker`, `Result`, or Rust-owned layout crosses the ABI. The Rust
adapter allocates one private wake bridge per observer and frees it only after successful callback
unregistration. A provider violating unregister quiescence causes the bridge to be leaked rather
than freed unsafely.

## Compatibility and backend events

`RuntimeHalAdapter` converts the current synchronous `RuntimeHal` into
`OwnedOperation -> Completion`. It preserves behavior by performing submit/wait/destroy during
`submit` and returning an already-ready future. It never blocks from `Future::poll` and claims no
overlap.

After the zero-copy ownership leases merge, a ggml backend event should contain an exact immutable
snapshot:

```rust
struct BackendEventSnapshot {
    completion: Completion,
    queue_sequence: u64,
    required_visibility: Option<Completion>,
}
```

Recording snapshots the queue tail. Synchronizing waits for exactly that snapshot and required
consumer visibility. Waiting inserts a provider dependency when supported, otherwise blocks on
that event only. Freeing drops one observer. Void C callbacks latch errors for the next
status-returning backend call.

No llama.cpp file is changed here. IVPU early return must wait for the post-foundation zero-copy
leases and per-invocation command slots. Staged output paths remain synchronous because raw output
pointers do not provide an owned destination lifetime.

## Dependency assessment

- `core::future` and `alloc::task::Wake` are sufficient for the typed future, task wakers, and the
  bounded executor implemented here.
- `futures-core`/`futures-task` 0.3.32 support `no_std`; the umbrella defaults enable `std` and an
  executor. `AtomicWaker` intentionally has replace-one-observer semantics, so it does not solve
  this completion's multiple-observer requirement.
- `async-task` 4.7.1 works with default features disabled and `alloc`, but supplies task storage and
  scheduling primitives rather than a reactor or completion ownership model.
- `maitake` is `no_std`-first and its alloc scheduler is architecturally compatible, but it brings a
  larger scheduler/task model and static-scheduler choices that do not materially improve this
  small bounded caller-driven implementation.
- `polling` is a standard-library platform layer; existing rustix epoll support already covers the
  Linux host adapter.

No async dependency was added. The existing rustix dependency gained only its `thread` feature for
futex support.

## Cache integration

The cache worktree's `WaitProvider` should be replaced in two merge steps:

1. adapt shared `AtomicWait::wait_u32/wake_u32` to the legacy `wait_while/wake_all` trait;
2. inject `AtomicWait` directly and delete `WaitProvider`.

Cache single-flight wait/wake remains an atomic condition protocol, not a device completion.
If cache code later becomes a separate crate, move only the tiny atomic-wait contract to a
dependency-leaf host-contract crate.

## IVPU gap

Current IVPU command-BO wait is authoritative but blocking, and timeout zero is not a guaranteed
strict nonblocking status query. The device fd is not a completion source. Therefore this
foundation cannot yet autonomously wake an IVPU completion future. The existing synchronous
compatibility path remains correct. A future provider may use an explicitly injected bounded
blocking executor, or a real kernel completion source if one becomes available. No device/host
overlap is claimed until provider integration and measurement prove it.
