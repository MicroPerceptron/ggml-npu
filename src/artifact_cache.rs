//! Codec- and platform-neutral artifact cache contracts and in-memory policy.
//!
//! Core retains compressed records only. Filesystem, process coordination, sleeping and codec
//! mechanics are supplied by the host adapter; no host or accelerator API appears in this module.

#![allow(dead_code)]

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::hint::spin_loop;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::identity::SemanticDigest;

pub const RECORD_MAGIC: [u8; 8] = *b"GGNPUC01";
pub const RECORD_SCHEMA: u16 = 1;
pub const RECORD_HEADER_BYTES: usize = 352;
pub const MAX_RECORD_BYTES: usize = 512 * 1024 * 1024 + RECORD_HEADER_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactKey(pub SemanticDigest);

impl ArtifactKey {
    pub const fn bytes(self) -> [u8; 32] {
        self.0 .0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheCompatibility {
    pub artifact_schema: u32,
    pub compiler: SemanticDigest,
    pub pipeline: SemanticDigest,
    pub toolchain: SemanticDigest,
    pub target: SemanticDigest,
    pub firmware: SemanticDigest,
    pub runtime: SemanticDigest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodecDescriptor {
    pub id: u32,
    pub version: u32,
    pub options: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheBudgets {
    pub ram_bytes: u64,
    pub ram_entry_bytes: u64,
    pub disk_bytes: u64,
    pub disk_low_water_bytes: u64,
}

impl CacheBudgets {
    pub const fn new(ram_bytes: u64, ram_entry_bytes: u64, disk_bytes: u64) -> Self {
        Self {
            ram_bytes,
            ram_entry_bytes,
            disk_bytes,
            disk_low_water_bytes: disk_bytes.saturating_mul(9) / 10,
        }
    }

    pub const fn valid(self) -> bool {
        self.ram_entry_bytes <= self.ram_bytes
            && self.disk_low_water_bytes <= self.disk_bytes
            && self.disk_bytes <= usize::MAX as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheError {
    Allocation,
    Overflow,
    RecordTooLarge,
    BadMagic,
    UnsupportedRecord,
    HeaderIntegrity,
    LengthMismatch,
    KeyMismatch,
    CompatibilityMismatch,
    CodecMismatch,
    PayloadIntegrity,
    Compression,
    Store,
}

pub trait CompressionProvider: Sync {
    fn descriptor(&self) -> CodecDescriptor;
    fn compress(&self, source: &[u8]) -> Result<Vec<u8>, CacheError>;
    fn decompress(&self, source: &[u8], uncompressed_bytes: usize) -> Result<Vec<u8>, CacheError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreReport {
    pub evicted_bytes: u64,
}

pub trait ArtifactStore: Sync {
    fn load(&self, key: ArtifactKey, max_bytes: usize) -> Result<Option<Vec<u8>>, CacheError>;
    fn store(
        &self,
        key: ArtifactKey,
        record: &[u8],
        budget_bytes: u64,
        low_water_bytes: u64,
    ) -> Result<StoreReport, CacheError>;
    fn quarantine(&self, key: ArtifactKey) -> Result<(), CacheError>;
}

/// Host wait/wake seam used by per-key single-flight. Implementations may use a futex or equivalent.
pub trait WaitProvider: Sync {
    fn wait_while(&self, state: &AtomicU32, expected: u32);
    fn wake_all(&self, state: &AtomicU32);
}

pub trait CacheClock: Sync {
    fn now_ns(&self) -> u64;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheTier {
    Resident,
    Memory,
    Persistent,
    Compiler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BypassReason {
    Disabled,
    Oversized,
    FrequencyRejected,
    Pinned,
    Budget,
    Incompatible,
    StoreUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionReason {
    Capacity,
    Generation,
    Corrupt,
    Incompatible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOperation {
    Compress,
    Decompress,
    DiskRead,
    DiskWrite,
    Compile,
    Promote,
    SingleFlightWait,
    ResidentRehydrate,
}

pub trait CacheObserver: Sync {
    fn hit(&self, _tier: CacheTier, _bytes: u64) {}
    fn miss(&self, _tier: CacheTier) {}
    fn bypass(&self, _tier: CacheTier, _reason: BypassReason, _bytes: u64) {}
    fn eviction(&self, _tier: CacheTier, _reason: EvictionReason, _bytes: u64) {}
    fn corruption(&self, _tier: CacheTier) {}
    fn latency(&self, _operation: CacheOperation, _elapsed_ns: u64, _bytes: u64) {}
    fn compression_ratio(&self, _uncompressed_bytes: u64, _compressed_bytes: u64) {}
    fn bytes(&self, _tier: CacheTier, _current: u64, _budget: u64) {}
}

pub struct NoopCacheObserver;
impl CacheObserver for NoopCacheObserver {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordHeader {
    pub artifact_schema: u32,
    pub key: ArtifactKey,
    pub compatibility: CacheCompatibility,
    pub codec: CodecDescriptor,
    pub compressed_bytes: u64,
    pub uncompressed_bytes: u64,
    pub uncompressed_digest: SemanticDigest,
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, CacheError> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or(CacheError::LengthMismatch)?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, CacheError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(CacheError::LengthMismatch)?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, CacheError> {
    Ok(read_u32(bytes, offset)? as i32)
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, CacheError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(CacheError::LengthMismatch)?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

fn read_digest(bytes: &[u8], offset: usize) -> Result<SemanticDigest, CacheError> {
    let value = bytes
        .get(offset..offset + 32)
        .ok_or(CacheError::LengthMismatch)?;
    let mut digest = [0; 32];
    digest.copy_from_slice(value);
    Ok(SemanticDigest(digest))
}

fn write_digest(target: &mut [u8], offset: usize, digest: SemanticDigest) {
    target[offset..offset + 32].copy_from_slice(&digest.0);
}

impl RecordHeader {
    const HEADER_DIGEST_OFFSET: usize = 320;

    pub fn encode(self, compressed: Vec<u8>) -> Result<Vec<u8>, CacheError> {
        if compressed.len() as u64 != self.compressed_bytes {
            return Err(CacheError::LengthMismatch);
        }
        let total = RECORD_HEADER_BYTES
            .checked_add(compressed.len())
            .ok_or(CacheError::Overflow)?;
        if total > MAX_RECORD_BYTES {
            return Err(CacheError::RecordTooLarge);
        }
        let mut record = Vec::new();
        record
            .try_reserve_exact(total)
            .map_err(|_| CacheError::Allocation)?;
        record.resize(RECORD_HEADER_BYTES, 0);
        record[0..8].copy_from_slice(&RECORD_MAGIC);
        record[8..10].copy_from_slice(&RECORD_SCHEMA.to_le_bytes());
        record[10..12].copy_from_slice(&(RECORD_HEADER_BYTES as u16).to_le_bytes());
        record[12..16].copy_from_slice(&self.artifact_schema.to_le_bytes());
        record[16..20].copy_from_slice(&self.codec.id.to_le_bytes());
        record[20..24].copy_from_slice(&self.codec.version.to_le_bytes());
        record[24..28].copy_from_slice(&self.codec.options.to_le_bytes());
        record[32..40].copy_from_slice(&self.compressed_bytes.to_le_bytes());
        record[40..48].copy_from_slice(&self.uncompressed_bytes.to_le_bytes());
        write_digest(&mut record, 48, self.key.0);
        write_digest(&mut record, 80, self.uncompressed_digest);
        write_digest(&mut record, 112, self.compatibility.compiler);
        write_digest(&mut record, 144, self.compatibility.pipeline);
        write_digest(&mut record, 176, self.compatibility.toolchain);
        write_digest(&mut record, 208, self.compatibility.target);
        write_digest(&mut record, 240, self.compatibility.firmware);
        write_digest(&mut record, 272, self.compatibility.runtime);
        record[304..308].copy_from_slice(&self.compatibility.artifact_schema.to_le_bytes());
        let header_digest = blake3::hash(&record[..Self::HEADER_DIGEST_OFFSET]);
        record[Self::HEADER_DIGEST_OFFSET..RECORD_HEADER_BYTES]
            .copy_from_slice(header_digest.as_bytes());
        record.extend_from_slice(&compressed);
        Ok(record)
    }

    pub fn parse(record: &[u8]) -> Result<(Self, &[u8]), CacheError> {
        if record.len() < RECORD_HEADER_BYTES {
            return Err(CacheError::LengthMismatch);
        }
        if record[..8] != RECORD_MAGIC {
            return Err(CacheError::BadMagic);
        }
        if read_u16(record, 8)? != RECORD_SCHEMA
            || read_u16(record, 10)? as usize != RECORD_HEADER_BYTES
        {
            return Err(CacheError::UnsupportedRecord);
        }
        let actual = blake3::hash(&record[..Self::HEADER_DIGEST_OFFSET]);
        if actual.as_bytes() != &record[Self::HEADER_DIGEST_OFFSET..RECORD_HEADER_BYTES] {
            return Err(CacheError::HeaderIntegrity);
        }
        let compressed_bytes = read_u64(record, 32)?;
        let uncompressed_bytes = read_u64(record, 40)?;
        let compressed_len = usize::try_from(compressed_bytes).map_err(|_| CacheError::Overflow)?;
        if RECORD_HEADER_BYTES
            .checked_add(compressed_len)
            .ok_or(CacheError::Overflow)?
            != record.len()
        {
            return Err(CacheError::LengthMismatch);
        }
        let compatibility = CacheCompatibility {
            artifact_schema: read_u32(record, 304)?,
            compiler: read_digest(record, 112)?,
            pipeline: read_digest(record, 144)?,
            toolchain: read_digest(record, 176)?,
            target: read_digest(record, 208)?,
            firmware: read_digest(record, 240)?,
            runtime: read_digest(record, 272)?,
        };
        let header = Self {
            artifact_schema: read_u32(record, 12)?,
            key: ArtifactKey(read_digest(record, 48)?),
            compatibility,
            codec: CodecDescriptor {
                id: read_u32(record, 16)?,
                version: read_u32(record, 20)?,
                options: read_i32(record, 24)?,
            },
            compressed_bytes,
            uncompressed_bytes,
            uncompressed_digest: read_digest(record, 80)?,
        };
        Ok((header, &record[RECORD_HEADER_BYTES..]))
    }

    pub fn ensure_compatible(
        &self,
        key: ArtifactKey,
        compatibility: CacheCompatibility,
        codec: CodecDescriptor,
    ) -> Result<(), CacheError> {
        if self.key != key {
            return Err(CacheError::KeyMismatch);
        }
        if self.artifact_schema != compatibility.artifact_schema
            || self.compatibility != compatibility
        {
            return Err(CacheError::CompatibilityMismatch);
        }
        if self.codec != codec {
            return Err(CacheError::CodecMismatch);
        }
        Ok(())
    }
}

struct SpinMutex<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

impl<T> SpinMutex<T> {
    const fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    fn lock(&self) -> SpinGuard<'_, T> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }
        SpinGuard { mutex: self }
    }
}

// SAFETY: `value` is accessed only while the atomic lock is held.
unsafe impl<T: Send> Sync for SpinMutex<T> {}

struct SpinGuard<'a, T> {
    mutex: &'a SpinMutex<T>,
}

impl<T> core::ops::Deref for SpinGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: this guard holds the mutex.
        unsafe { &*self.mutex.value.get() }
    }
}

impl<T> core::ops::DerefMut for SpinGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: this guard uniquely owns access while holding the mutex.
        unsafe { &mut *self.mutex.value.get() }
    }
}

impl<T> Drop for SpinGuard<'_, T> {
    fn drop(&mut self) {
        self.mutex.locked.store(false, Ordering::Release);
    }
}

const SKETCH_WIDTH: usize = 1024;

struct FrequencySketch {
    counters: [u8; SKETCH_WIDTH],
    samples: u32,
}

impl FrequencySketch {
    const fn new() -> Self {
        Self {
            counters: [0; SKETCH_WIDTH],
            samples: 0,
        }
    }

    fn indices(key: ArtifactKey) -> [usize; 4] {
        let bytes = key.bytes();
        let mut words = [0u64; 4];
        for (index, word) in words.iter_mut().enumerate() {
            let offset = index * 8;
            *word = u64::from_le_bytes(
                bytes[offset..offset + 8]
                    .try_into()
                    .expect("fixed digest chunk"),
            );
        }
        [
            words[0] as usize & (SKETCH_WIDTH - 1),
            words[1].rotate_left(13) as usize & (SKETCH_WIDTH - 1),
            words[2].rotate_left(29) as usize & (SKETCH_WIDTH - 1),
            words[3].rotate_left(47) as usize & (SKETCH_WIDTH - 1),
        ]
    }

    fn observe(&mut self, key: ArtifactKey) {
        for index in Self::indices(key) {
            self.counters[index] = self.counters[index].saturating_add(1);
        }
        self.samples = self.samples.saturating_add(1);
        if self.samples >= (SKETCH_WIDTH as u32) * 10 {
            for counter in &mut self.counters {
                *counter >>= 1;
            }
            self.samples = 0;
        }
    }

    fn estimate(&self, key: ArtifactKey) -> u8 {
        Self::indices(key)
            .into_iter()
            .map(|index| self.counters[index])
            .min()
            .unwrap_or(0)
    }
}

struct RamEntry {
    key: ArtifactKey,
    record: Arc<[u8]>,
    sequence: u64,
}

struct RamState {
    entries: Vec<RamEntry>,
    bytes: u64,
    sequence: u64,
    sketch: FrequencySketch,
}

impl RamState {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            sequence: 0,
            sketch: FrequencySketch::new(),
        }
    }

    fn get(&mut self, key: ArtifactKey) -> Option<Arc<[u8]>> {
        self.sketch.observe(key);
        self.entries
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| Arc::clone(&entry.record))
    }

    fn admit(
        &mut self,
        key: ArtifactKey,
        record: Arc<[u8]>,
        budgets: CacheBudgets,
        observer: &dyn CacheObserver,
    ) -> bool {
        self.sketch.observe(key);
        let bytes = record.len() as u64;
        if budgets.ram_bytes == 0 {
            observer.bypass(CacheTier::Memory, BypassReason::Disabled, bytes);
            return false;
        }
        if bytes > budgets.ram_entry_bytes || bytes > budgets.ram_bytes {
            observer.bypass(CacheTier::Memory, BypassReason::Oversized, bytes);
            return false;
        }
        if let Some(existing) = self.entries.iter().position(|entry| entry.key == key) {
            let old = self.entries.swap_remove(existing);
            self.bytes = self.bytes.saturating_sub(old.record.len() as u64);
        }
        let candidate_frequency = self.sketch.estimate(key);
        while self.bytes.saturating_add(bytes) > budgets.ram_bytes {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| {
                    (
                        self.sketch.estimate(entry.key),
                        entry.sequence,
                        entry.key.bytes(),
                    )
                })
                .map(|(index, entry)| (index, self.sketch.estimate(entry.key)));
            let Some((index, victim_frequency)) = victim else {
                observer.bypass(CacheTier::Memory, BypassReason::Budget, bytes);
                return false;
            };
            if candidate_frequency < victim_frequency {
                observer.bypass(CacheTier::Memory, BypassReason::FrequencyRejected, bytes);
                return false;
            }
            let victim = self.entries.swap_remove(index);
            let victim_bytes = victim.record.len() as u64;
            self.bytes = self.bytes.saturating_sub(victim_bytes);
            observer.eviction(CacheTier::Memory, EvictionReason::Capacity, victim_bytes);
        }
        self.sequence = self.sequence.wrapping_add(1);
        self.bytes = self.bytes.saturating_add(bytes);
        self.entries.push(RamEntry {
            key,
            record,
            sequence: self.sequence,
        });
        observer.bytes(CacheTier::Memory, self.bytes, budgets.ram_bytes);
        true
    }
}

const FLIGHT_RUNNING: u32 = 1;
const FLIGHT_COMPLETE: u32 = 2;

struct Flight {
    key: ArtifactKey,
    state: AtomicU32,
}

struct FlightState {
    flights: Vec<Arc<Flight>>,
}

impl FlightState {
    const fn new() -> Self {
        Self {
            flights: Vec::new(),
        }
    }
}

enum FlightRole<'a> {
    Leader(FlightLeader<'a>),
    Waiter(Arc<Flight>),
}

struct FlightLeader<'a> {
    registry: &'a SingleFlight,
    flight: Arc<Flight>,
    waiter: &'a dyn WaitProvider,
}

impl Drop for FlightLeader<'_> {
    fn drop(&mut self) {
        self.flight.state.store(FLIGHT_COMPLETE, Ordering::Release);
        self.waiter.wake_all(&self.flight.state);
        let mut state = self.registry.state.lock();
        if let Some(index) = state
            .flights
            .iter()
            .position(|candidate| Arc::ptr_eq(candidate, &self.flight))
        {
            state.flights.swap_remove(index);
        }
    }
}

struct SingleFlight {
    state: SpinMutex<FlightState>,
}

impl SingleFlight {
    const fn new() -> Self {
        Self {
            state: SpinMutex::new(FlightState::new()),
        }
    }

    fn begin<'a>(
        &'a self,
        key: ArtifactKey,
        waiter: &'a dyn WaitProvider,
    ) -> Result<FlightRole<'a>, CacheError> {
        let mut state = self.state.lock();
        if let Some(flight) = state.flights.iter().find(|flight| flight.key == key) {
            return Ok(FlightRole::Waiter(Arc::clone(flight)));
        }
        state
            .flights
            .try_reserve(1)
            .map_err(|_| CacheError::Allocation)?;
        let flight = Arc::new(Flight {
            key,
            state: AtomicU32::new(FLIGHT_RUNNING),
        });
        state.flights.push(Arc::clone(&flight));
        Ok(FlightRole::Leader(FlightLeader {
            registry: self,
            flight,
            waiter,
        }))
    }
}

pub struct TieredArtifactCache {
    ram: SpinMutex<RamState>,
    flights: SingleFlight,
}

impl TieredArtifactCache {
    pub const fn new() -> Self {
        Self {
            ram: SpinMutex::new(RamState::new()),
            flights: SingleFlight::new(),
        }
    }

    fn promote(
        &self,
        key: ArtifactKey,
        compatibility: CacheCompatibility,
        record: Arc<[u8]>,
        compression: &dyn CompressionProvider,
        clock: &dyn CacheClock,
        observer: &dyn CacheObserver,
    ) -> Result<Vec<u8>, CacheError> {
        let (header, payload) = RecordHeader::parse(&record)?;
        header.ensure_compatible(key, compatibility, compression.descriptor())?;
        let uncompressed_bytes =
            usize::try_from(header.uncompressed_bytes).map_err(|_| CacheError::Overflow)?;
        if uncompressed_bytes > MAX_RECORD_BYTES {
            return Err(CacheError::RecordTooLarge);
        }
        let started = clock.now_ns();
        let output = compression.decompress(payload, uncompressed_bytes)?;
        let elapsed = clock.now_ns().saturating_sub(started);
        observer.latency(
            CacheOperation::Decompress,
            elapsed,
            uncompressed_bytes as u64,
        );
        observer.latency(CacheOperation::Promote, elapsed, uncompressed_bytes as u64);
        if output.len() != uncompressed_bytes {
            return Err(CacheError::LengthMismatch);
        }
        if blake3::hash(&output).as_bytes() != &header.uncompressed_digest.0 {
            return Err(CacheError::PayloadIntegrity);
        }
        Ok(output)
    }

    fn memory_lookup(
        &self,
        key: ArtifactKey,
        compatibility: CacheCompatibility,
        compression: &dyn CompressionProvider,
        clock: &dyn CacheClock,
        observer: &dyn CacheObserver,
    ) -> Result<Option<Vec<u8>>, CacheError> {
        let record = self.ram.lock().get(key);
        let Some(record) = record else {
            observer.miss(CacheTier::Memory);
            return Ok(None);
        };
        match self.promote(key, compatibility, record, compression, clock, observer) {
            Ok(output) => {
                observer.hit(CacheTier::Memory, output.len() as u64);
                Ok(Some(output))
            }
            Err(error) => {
                observer.corruption(CacheTier::Memory);
                let mut ram = self.ram.lock();
                if let Some(index) = ram.entries.iter().position(|entry| entry.key == key) {
                    let entry = ram.entries.swap_remove(index);
                    let bytes = entry.record.len() as u64;
                    ram.bytes = ram.bytes.saturating_sub(bytes);
                    observer.eviction(CacheTier::Memory, EvictionReason::Corrupt, bytes);
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn persistent_lookup(
        &self,
        key: ArtifactKey,
        compatibility: CacheCompatibility,
        budgets: CacheBudgets,
        compression: &dyn CompressionProvider,
        store: &dyn ArtifactStore,
        clock: &dyn CacheClock,
        observer: &dyn CacheObserver,
    ) -> Result<Option<Vec<u8>>, CacheError> {
        let started = clock.now_ns();
        let loaded = store.load(key, MAX_RECORD_BYTES);
        observer.latency(
            CacheOperation::DiskRead,
            clock.now_ns().saturating_sub(started),
            loaded
                .as_ref()
                .ok()
                .and_then(|record| record.as_ref())
                .map_or(0, |record| record.len() as u64),
        );
        let Some(record) = loaded? else {
            observer.miss(CacheTier::Persistent);
            return Ok(None);
        };
        let record: Arc<[u8]> = Arc::from(record.into_boxed_slice());
        match self.promote(
            key,
            compatibility,
            Arc::clone(&record),
            compression,
            clock,
            observer,
        ) {
            Ok(output) => {
                observer.hit(CacheTier::Persistent, record.len() as u64);
                observer.latency(CacheOperation::ResidentRehydrate, 0, output.len() as u64);
                self.ram.lock().admit(key, record, budgets, observer);
                Ok(Some(output))
            }
            Err(CacheError::CompatibilityMismatch | CacheError::CodecMismatch) => {
                observer.bypass(
                    CacheTier::Persistent,
                    BypassReason::Incompatible,
                    record.len() as u64,
                );
                store.quarantine(key)?;
                Ok(None)
            }
            Err(error) => {
                observer.corruption(CacheTier::Persistent);
                store.quarantine(key)?;
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lookup(
        &self,
        key: ArtifactKey,
        compatibility: CacheCompatibility,
        budgets: CacheBudgets,
        compression: &dyn CompressionProvider,
        store: &dyn ArtifactStore,
        clock: &dyn CacheClock,
        observer: &dyn CacheObserver,
    ) -> Result<Option<Vec<u8>>, CacheError> {
        match self.memory_lookup(key, compatibility, compression, clock, observer) {
            Ok(Some(output)) => return Ok(Some(output)),
            Ok(None) => {}
            Err(_) => {}
        }
        self.persistent_lookup(
            key,
            compatibility,
            budgets,
            compression,
            store,
            clock,
            observer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_or_compile(
        &self,
        key: ArtifactKey,
        compatibility: CacheCompatibility,
        budgets: CacheBudgets,
        compression: &dyn CompressionProvider,
        store: &dyn ArtifactStore,
        waiter: &dyn WaitProvider,
        clock: &dyn CacheClock,
        observer: &dyn CacheObserver,
        compile: impl FnOnce() -> Option<Vec<u8>>,
    ) -> Option<Vec<u8>> {
        if !budgets.valid() {
            observer.bypass(CacheTier::Compiler, BypassReason::Budget, 0);
            return compile();
        }
        if let Ok(Some(output)) = self.lookup(
            key,
            compatibility,
            budgets,
            compression,
            store,
            clock,
            observer,
        ) {
            return Some(output);
        }
        match self.flights.begin(key, waiter).ok()? {
            FlightRole::Waiter(flight) => {
                let started = clock.now_ns();
                waiter.wait_while(&flight.state, FLIGHT_RUNNING);
                observer.latency(
                    CacheOperation::SingleFlightWait,
                    clock.now_ns().saturating_sub(started),
                    0,
                );
                self.lookup(
                    key,
                    compatibility,
                    budgets,
                    compression,
                    store,
                    clock,
                    observer,
                )
                .ok()
                .flatten()
            }
            FlightRole::Leader(_leader) => {
                // No cache or single-flight lock is held across compile, compression or I/O.
                let started = clock.now_ns();
                let output = compile()?;
                observer.latency(
                    CacheOperation::Compile,
                    clock.now_ns().saturating_sub(started),
                    output.len() as u64,
                );
                let started = clock.now_ns();
                let compressed = match compression.compress(&output) {
                    Ok(compressed) => compressed,
                    Err(_) => {
                        observer.bypass(
                            CacheTier::Memory,
                            BypassReason::Disabled,
                            output.len() as u64,
                        );
                        return Some(output);
                    }
                };
                observer.latency(
                    CacheOperation::Compress,
                    clock.now_ns().saturating_sub(started),
                    output.len() as u64,
                );
                observer.compression_ratio(output.len() as u64, compressed.len() as u64);
                let header = RecordHeader {
                    artifact_schema: compatibility.artifact_schema,
                    key,
                    compatibility,
                    codec: compression.descriptor(),
                    compressed_bytes: compressed.len() as u64,
                    uncompressed_bytes: output.len() as u64,
                    uncompressed_digest: SemanticDigest(*blake3::hash(&output).as_bytes()),
                };
                let record = header.encode(compressed).ok()?;
                let record_bytes = record.len() as u64;
                if budgets.disk_bytes == 0 {
                    observer.bypass(CacheTier::Persistent, BypassReason::Disabled, record_bytes);
                } else if record_bytes > budgets.disk_bytes {
                    observer.bypass(CacheTier::Persistent, BypassReason::Oversized, record_bytes);
                } else {
                    let started = clock.now_ns();
                    let stored = store.store(
                        key,
                        &record,
                        budgets.disk_bytes,
                        budgets.disk_low_water_bytes,
                    );
                    observer.latency(
                        CacheOperation::DiskWrite,
                        clock.now_ns().saturating_sub(started),
                        record_bytes,
                    );
                    match stored {
                        Ok(report) if report.evicted_bytes != 0 => observer.eviction(
                            CacheTier::Persistent,
                            EvictionReason::Capacity,
                            report.evicted_bytes,
                        ),
                        Ok(_) => {}
                        Err(_) => observer.bypass(
                            CacheTier::Persistent,
                            BypassReason::StoreUnavailable,
                            record_bytes,
                        ),
                    }
                }
                self.ram
                    .lock()
                    .admit(key, Arc::from(record.into_boxed_slice()), budgets, observer);
                Some(output)
            }
        }
    }
}

impl Default for TieredArtifactCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::sync::Arc;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex};
    use std::thread;

    const CODEC: CodecDescriptor = CodecDescriptor {
        id: 7,
        version: 2,
        options: -1,
    };

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest([byte; 32])
    }

    fn key(byte: u8) -> ArtifactKey {
        ArtifactKey(digest(byte))
    }

    fn compatibility(byte: u8) -> CacheCompatibility {
        CacheCompatibility {
            artifact_schema: 3,
            compiler: digest(byte),
            pipeline: digest(byte.wrapping_add(1)),
            toolchain: digest(byte.wrapping_add(2)),
            target: digest(byte.wrapping_add(3)),
            firmware: digest(byte.wrapping_add(4)),
            runtime: digest(byte.wrapping_add(5)),
        }
    }

    struct IdentityCodec;

    impl CompressionProvider for IdentityCodec {
        fn descriptor(&self) -> CodecDescriptor {
            CODEC
        }

        fn compress(&self, source: &[u8]) -> Result<Vec<u8>, CacheError> {
            Ok(source.to_vec())
        }

        fn decompress(
            &self,
            source: &[u8],
            _uncompressed_bytes: usize,
        ) -> Result<Vec<u8>, CacheError> {
            Ok(source.to_vec())
        }
    }

    #[derive(Default)]
    struct MemoryStore(Mutex<BTreeMap<[u8; 32], Vec<u8>>>);

    impl ArtifactStore for MemoryStore {
        fn load(&self, key: ArtifactKey, _max_bytes: usize) -> Result<Option<Vec<u8>>, CacheError> {
            Ok(self.0.lock().unwrap().get(&key.bytes()).cloned())
        }

        fn store(
            &self,
            key: ArtifactKey,
            record: &[u8],
            _budget_bytes: u64,
            _low_water_bytes: u64,
        ) -> Result<StoreReport, CacheError> {
            self.0.lock().unwrap().insert(key.bytes(), record.to_vec());
            Ok(StoreReport::default())
        }

        fn quarantine(&self, key: ArtifactKey) -> Result<(), CacheError> {
            self.0.lock().unwrap().remove(&key.bytes());
            Ok(())
        }
    }

    #[derive(Default)]
    struct ThreadWait {
        mutex: Mutex<()>,
        condition: Condvar,
    }

    impl WaitProvider for ThreadWait {
        fn wait_while(&self, state: &AtomicU32, expected: u32) {
            let mut guard = self.mutex.lock().unwrap();
            while state.load(Ordering::Acquire) == expected {
                guard = self.condition.wait(guard).unwrap();
            }
        }

        fn wake_all(&self, _state: &AtomicU32) {
            self.condition.notify_all();
        }
    }

    struct TestClock;

    impl CacheClock for TestClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    fn budgets() -> CacheBudgets {
        CacheBudgets::new(4096, 2048, 16384)
    }

    #[test]
    fn record_round_trip_is_pointer_free_and_integrity_checked() {
        let payload = vec![1, 2, 3, 4];
        let header = RecordHeader {
            artifact_schema: 3,
            key: key(1),
            compatibility: compatibility(1),
            codec: CODEC,
            compressed_bytes: payload.len() as u64,
            uncompressed_bytes: payload.len() as u64,
            uncompressed_digest: SemanticDigest(*blake3::hash(&payload).as_bytes()),
        };
        let encoded = header.encode(payload.clone()).unwrap();
        let (decoded, bytes) = RecordHeader::parse(&encoded).unwrap();
        assert_eq!(decoded, header);
        assert_eq!(bytes, payload);
        assert_eq!(RECORD_HEADER_BYTES, 352);
    }

    #[test]
    fn record_rejects_overflow_corruption_codec_and_compatibility() {
        let header = RecordHeader {
            artifact_schema: 3,
            key: key(1),
            compatibility: compatibility(1),
            codec: CODEC,
            compressed_bytes: 1,
            uncompressed_bytes: 1,
            uncompressed_digest: digest(9),
        };
        let mut encoded = header.encode(vec![9]).unwrap();
        encoded[120] ^= 1;
        assert_eq!(
            RecordHeader::parse(&encoded),
            Err(CacheError::HeaderIntegrity)
        );
        let encoded = header.encode(vec![9]).unwrap();
        let (decoded, _) = RecordHeader::parse(&encoded).unwrap();
        assert_eq!(
            decoded.ensure_compatible(
                key(1),
                compatibility(1),
                CodecDescriptor {
                    version: 3,
                    ..CODEC
                }
            ),
            Err(CacheError::CodecMismatch)
        );
        assert_eq!(
            decoded.ensure_compatible(key(1), compatibility(2), CODEC),
            Err(CacheError::CompatibilityMismatch)
        );
    }

    #[test]
    fn tinylfu_admission_is_byte_bounded_scan_resistant_and_deterministic() {
        let observer = NoopCacheObserver;
        let mut ram = RamState::new();
        let budgets = CacheBudgets::new(1000, 600, 0);
        let hot: Arc<[u8]> = Arc::from(vec![0; 500].into_boxed_slice());
        assert!(ram.admit(key(1), hot, budgets, &observer));
        for _ in 0..8 {
            assert!(ram.get(key(1)).is_some());
        }
        for candidate in 2..20 {
            let scan: Arc<[u8]> = Arc::from(vec![candidate; 600].into_boxed_slice());
            assert!(!ram.admit(key(candidate), scan, budgets, &observer));
        }
        assert_eq!(ram.bytes, 500);
        assert_eq!(ram.entries.len(), 1);
        assert_eq!(ram.entries[0].key, key(1));
        let oversized: Arc<[u8]> = Arc::from(vec![0; 601].into_boxed_slice());
        assert!(!ram.admit(key(99), oversized, budgets, &observer));
    }

    #[test]
    fn promotion_keeps_only_compressed_record_and_checks_payload_digest() {
        let cache = TieredArtifactCache::new();
        let store = MemoryStore::default();
        let waiter = ThreadWait::default();
        let observer = NoopCacheObserver;
        let first = cache
            .get_or_compile(
                key(7),
                compatibility(7),
                budgets(),
                &IdentityCodec,
                &store,
                &waiter,
                &TestClock,
                &observer,
                || Some(vec![4, 5, 6]),
            )
            .unwrap();
        assert_eq!(first, vec![4, 5, 6]);
        let second = cache
            .get_or_compile(
                key(7),
                compatibility(7),
                budgets(),
                &IdentityCodec,
                &store,
                &waiter,
                &TestClock,
                &observer,
                || panic!("warm cache must not compile"),
            )
            .unwrap();
        assert_eq!(second, first);
        assert_eq!(cache.ram.lock().bytes, (RECORD_HEADER_BYTES + 3) as u64);
    }

    #[test]
    fn concurrent_miss_compiles_once_per_key() {
        let cache = Arc::new(TieredArtifactCache::new());
        let store = Arc::new(MemoryStore::default());
        let waiter = Arc::new(ThreadWait::default());
        let compiles = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let cache = Arc::clone(&cache);
            let store = Arc::clone(&store);
            let waiter = Arc::clone(&waiter);
            let compiles = Arc::clone(&compiles);
            workers.push(thread::spawn(move || {
                cache.get_or_compile(
                    key(8),
                    compatibility(8),
                    budgets(),
                    &IdentityCodec,
                    store.as_ref(),
                    waiter.as_ref(),
                    &TestClock,
                    &NoopCacheObserver,
                    || {
                        compiles.fetch_add(1, Ordering::Relaxed);
                        thread::sleep(core::time::Duration::from_millis(20));
                        Some(vec![8; 128])
                    },
                )
            }));
        }
        for worker in workers {
            assert_eq!(worker.join().unwrap().unwrap(), vec![8; 128]);
        }
        assert_eq!(compiles.load(Ordering::Relaxed), 1);
    }
}
