//! Control-plane primitives. The controller handles metadata only, never tensors.
//!
//! Two pieces define the async semantics:
//!  - `StalenessBudget`: capacity-based admission (AReaL inequality).
//!  - `VersionLedger`: per-token policy versions, so partial rollout produces
//!    correct importance ratios instead of a single trajectory-level label.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StalenessBudget {
    pub max_staleness: u64,
    pub consumer_batch_size: u32,
    pub concurrency: u32,
}

impl StalenessBudget {
    /// capacity = min(concurrency,
    ///   (max_staleness + current_version + 1) * consumer_bs - (accepted + running))
    pub fn capacity(&self, current_version: u64, accepted: u64, running: u64) -> u32 {
        let upper = self
            .max_staleness
            .saturating_add(current_version)
            .saturating_add(1)
            .saturating_mul(self.consumer_batch_size as u64);
        let remaining = upper.saturating_sub(accepted.saturating_add(running));
        remaining.min(self.concurrency as u64) as u32
    }

    /// AReaL `on_version_recovered`: re-base admitted work after a checkpoint
    /// restore so the next admission window is bounded instead of bursting.
    pub fn rebase_after_recovery(&self, recovered_version: u64) -> u64 {
        recovered_version.saturating_mul(self.consumer_batch_size as u64)
    }

    /// A token is stale if `current_version - token_version > max_staleness`.
    pub fn is_stale(&self, current_version: u64, token_version: u64) -> bool {
        current_version.saturating_sub(token_version) > self.max_staleness
    }
}

/// Token-bucket admission shaper: bounds trajectories/sec independently of
/// concurrency. Unlike a sleep-based limiter it never blocks the event loop.
#[derive(Debug, Clone)]
pub struct TokenBucketRateLimiter {
    rate_per_sec: f64,
    burst: u64,
    tokens: f64,
    last_ms: u128,
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

impl TokenBucketRateLimiter {
    pub fn new(rate_per_sec: f64, burst: u64) -> Self {
        assert!(rate_per_sec.is_finite() && rate_per_sec > 0.0, "rate must be positive");
        assert!(burst >= 1, "burst must be positive");
        Self {
            rate_per_sec,
            burst,
            tokens: burst as f64,
            last_ms: now_ms(),
        }
    }

    fn refill(&mut self, now_ms: u128) {
        let elapsed_ms = now_ms.saturating_sub(self.last_ms);
        self.tokens = (self.tokens + elapsed_ms as f64 * self.rate_per_sec / 1000.0)
            .min(self.burst as f64);
        self.last_ms = now_ms;
    }

    pub fn try_acquire(&mut self, count: u64) -> bool {
        self.try_acquire_at(count, now_ms())
    }

    pub fn try_acquire_at(&mut self, count: u64, at_ms: u128) -> bool {
        self.refill(at_ms);
        if self.tokens >= count as f64 {
            self.tokens -= count as f64;
            true
        } else {
            false
        }
    }

    pub fn available(&mut self) -> f64 {
        self.refill(now_ms());
        self.tokens
    }

    pub fn retry_after_ms(&mut self, count: u64) -> u64 {
        self.retry_after_ms_at(count, now_ms())
    }

    pub fn retry_after_ms_at(&mut self, count: u64, at_ms: u128) -> u64 {
        self.refill(at_ms);
        if self.tokens >= count as f64 {
            return 0;
        }
        let missing = count as f64 - self.tokens;
        (missing * 1000.0 / self.rate_per_sec).ceil().max(0.0) as u64
    }
}

/// Batched paced wake-up for a waiter queue. The async transport owns the
/// waiters; this primitive decides how many to release on each tick so a burst
/// cannot stampede the trainer. Credits accrue with time and are spent in
/// whole batches, and stored credits are capped at one batch.
#[derive(Debug, Clone)]
pub struct AsyncPacer {
    rate_per_sec: f64,
    batch_size: u32,
    credits: f64,
    last_ms: u128,
}

impl AsyncPacer {
    pub fn new(rate_per_sec: f64, batch_size: u32) -> Self {
        assert!(rate_per_sec.is_finite() && rate_per_sec > 0.0, "rate must be positive");
        assert!(batch_size >= 1, "batch size must be positive");
        Self {
            rate_per_sec,
            batch_size,
            credits: 0.0,
            last_ms: now_ms(),
        }
    }

    fn refill(&mut self, at_ms: u128) {
        let elapsed = at_ms.saturating_sub(self.last_ms);
        self.credits = (self.credits + elapsed as f64 * self.rate_per_sec / 1000.0)
            .min(self.batch_size as f64);
        self.last_ms = at_ms;
    }

    /// How many of `waiting` requests may proceed now.
    pub fn wake(&mut self, waiting: u32) -> u32 {
        self.wake_at(waiting, now_ms())
    }

    pub fn wake_at(&mut self, waiting: u32, at_ms: u128) -> u32 {
        self.refill(at_ms);
        if waiting == 0 {
            return 0;
        }
        let granted = (self.credits.floor() as u32).min(waiting).min(self.batch_size);
        self.credits -= granted as f64;
        granted
    }

    pub fn retry_after_ms(&mut self) -> u64 {
        self.retry_after_ms_at(now_ms())
    }

    pub fn retry_after_ms_at(&mut self, at_ms: u128) -> u64 {
        self.refill(at_ms);
        if self.credits >= 1.0 {
            return 0;
        }
        ((1.0 - self.credits) * 1000.0 / self.rate_per_sec)
            .ceil()
            .max(0.0) as u64
    }
}

/// A reusable slot pool with TTL leases. Each slot carries a unique token, so
/// a late release from an expired lease cannot free a slot that has since been
/// handed to a different request. This is the token-pool backpressure primitive
/// used to bound in-flight rollouts independently of the staleness budget.
#[derive(Debug, Clone)]
pub struct TokenSlotSemaphore {
    slots: Vec<Slot>,
    next_token: u64,
    ttl_ms: u128,
}

#[derive(Debug, Clone, Copy)]
struct Slot {
    /// 0 means free; otherwise the lease token currently holding the slot.
    token: u64,
    deadline_ms: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotLease {
    pub index: usize,
    pub token: u64,
}

impl TokenSlotSemaphore {
    pub fn new(capacity: usize, ttl_secs: f64) -> Self {
        assert!(capacity >= 1, "capacity must be positive");
        assert!(ttl_secs.is_finite() && ttl_secs > 0.0, "ttl must be positive");
        Self {
            slots: vec![Slot { token: 0, deadline_ms: 0 }; capacity],
            next_token: 1,
            ttl_ms: (ttl_secs * 1000.0) as u128,
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn try_acquire(&mut self) -> Option<SlotLease> {
        self.try_acquire_at(now_ms())
    }

    pub fn try_acquire_at(&mut self, at_ms: u128) -> Option<SlotLease> {
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if slot.token == 0 || at_ms >= slot.deadline_ms {
                let token = self.next_token;
                self.next_token = self.next_token.wrapping_add(1).max(1);
                *slot = Slot {
                    token,
                    deadline_ms: at_ms.saturating_add(self.ttl_ms),
                };
                return Some(SlotLease { index, token });
            }
        }
        None
    }

    /// Releases a slot only if the lease token still matches. Returns false for
    /// a stale or forged lease, leaving the current holder untouched.
    pub fn release(&mut self, lease: SlotLease) -> bool {
        match self.slots.get_mut(lease.index) {
            Some(slot) if lease.token != 0 && slot.token == lease.token => {
                slot.token = 0;
                slot.deadline_ms = 0;
                true
            }
            _ => false,
        }
    }

    /// Frees every lease whose TTL has elapsed. Returns the number reclaimed.
    pub fn reap(&mut self, at_ms: u128) -> usize {
        let mut freed = 0;
        for slot in &mut self.slots {
            if slot.token != 0 && at_ms >= slot.deadline_ms {
                slot.token = 0;
                slot.deadline_ms = 0;
                freed += 1;
            }
        }
        freed
    }

    /// Number of slots available at `at_ms` (expired leases are reclaimed).
    pub fn available(&mut self, at_ms: u128) -> usize {
        self.reap(at_ms);
        self.slots.iter().filter(|slot| slot.token == 0).count()
    }

    /// Number of live leases at `at_ms`, without mutating the pool.
    pub fn in_use(&self, at_ms: u128) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.token != 0 && at_ms < slot.deadline_ms)
            .count()
    }
}

#[derive(Debug, Default)]
pub struct VersionLedger {
    versions: HashMap<u64, Vec<u64>>,
}

impl VersionLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one trajectory. Returns an error if lengths do not align with the
    /// response tokens, since a misaligned version vector silently corrupts the
    /// importance ratio.
    pub fn record(
        &mut self,
        request_id: u64,
        token_count: usize,
        versions: Vec<u64>,
    ) -> Result<(), LedgerError> {
        if versions.len() != token_count {
            return Err(LedgerError::LengthMismatch {
                tokens: token_count,
                versions: versions.len(),
            });
        }
        if versions.is_empty() {
            return Err(LedgerError::Empty);
        }
        self.versions.insert(request_id, versions);
        Ok(())
    }

    pub fn forget(&mut self, request_id: u64) -> bool {
        self.versions.remove(&request_id).is_some()
    }

    pub fn len(&self) -> usize {
        self.versions.len()
    }

    /// Versions of a single trajectory (cloned; the ledger stays authoritative).
    pub fn versions(&self, request_id: u64) -> Option<&[u64]> {
        self.versions.get(&request_id).map(Vec::as_slice)
    }

    /// Count tokens produced by policy versions older than the budget allows.
    pub fn stale_tokens(&self, budget: &StalenessBudget, current_version: u64) -> usize {
        self.versions
            .values()
            .flat_map(|vs| vs.iter())
            .filter(|&&v| budget.is_stale(current_version, v))
            .count()
    }

    /// Total tokens tracked across all trajectories.
    pub fn token_count(&self) -> usize {
        self.versions.values().map(Vec::len).sum()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum LedgerError {
    LengthMismatch { tokens: usize, versions: usize },
    Empty,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_follows_inequality() {
        let budget = StalenessBudget {
            max_staleness: 4,
            consumer_batch_size: 8,
            concurrency: 32,
        };
        // (4 + 7 + 1) * 8 = 96; 96 - (10 + 6) = 80; min(32, 80) = 32
        assert_eq!(budget.capacity(7, 10, 6), 32);
        assert_eq!(budget.capacity(0, 0, 0), 32); // capped by concurrency
        assert_eq!(budget.capacity(0, 40, 0), 0);
        assert_eq!(budget.capacity(0, 100, 0), 0);
    }

    #[test]
    fn pacer_wakes_in_batches_without_stampede() {
        let mut pacer = AsyncPacer::new(100.0, 3);
        // 100/sec => one credit per 10ms; no credits at t=0.
        assert_eq!(pacer.wake_at(10, 0), 0);
        assert_eq!(pacer.wake_at(10, 20), 2); // two credits accrued
        assert_eq!(pacer.wake_at(10, 20), 0); // spent, no refill
        assert_eq!(pacer.wake_at(10, 50), 3); // capped by batch_size
        assert_eq!(pacer.wake_at(2, 80), 2); // capped by waiting
        assert_eq!(pacer.retry_after_ms_at(80), 0); // one credit left over
    }

    #[test]
    fn token_slots_are_bounded_reusable_and_lease_guarded() {
        let mut sem = TokenSlotSemaphore::new(2, 1.0);
        let a = sem.try_acquire_at(0).unwrap();
        let b = sem.try_acquire_at(0).unwrap();
        assert_ne!(a.index, b.index);
        assert_eq!(sem.try_acquire_at(0), None); // pool exhausted
        assert_eq!(sem.available(0), 0);
        assert_eq!(sem.in_use(0), 2);

        // A forged/expired token cannot release someone else's slot.
        assert!(!sem.release(SlotLease { index: a.index, token: a.token + 100 }));
        assert!(sem.release(a));
        assert_eq!(sem.available(0), 1);
        let c = sem.try_acquire_at(0).unwrap();
        assert_eq!(c.index, a.index); // freed slot reused

        // Once the TTL elapses the lease is invalid and the slot is reclaimed.
        assert_eq!(sem.reap(2_000), 2);
        assert_eq!(sem.available(2_000), 2);
        assert!(!sem.release(b));
        assert!(!sem.release(c));
    }

    #[test]
    fn token_bucket_shapes_rate_and_refills() {
        let mut limiter = TokenBucketRateLimiter::new(10.0, 2);
        assert!(limiter.try_acquire_at(2, 0));
        assert!(!limiter.try_acquire_at(1, 0)); // burst exhausted
        assert_eq!(limiter.retry_after_ms_at(1, 0), 100); // 1 token / 10 per sec
        assert!(limiter.try_acquire_at(1, 100)); // one token refilled
        assert!(limiter.available() <= 2.0);
        assert!(!limiter.try_acquire_at(3, 100_000)); // capped by burst
    }

    #[test]
    fn recovery_rebase_bounds_next_admission_window() {
        let budget = StalenessBudget {
            max_staleness: 4,
            consumer_batch_size: 8,
            concurrency: 32,
        };
        let rebased = budget.rebase_after_recovery(7);
        assert_eq!(rebased, 56);
        assert_eq!(budget.capacity(7, rebased, 0), 32);

        // With a wide concurrency cap the difference is visible: a cold
        // controller would admit 64 tokens, a resumed one only 8.
        let wide = StalenessBudget {
            max_staleness: 0,
            consumer_batch_size: 8,
            concurrency: 100,
        };
        assert_eq!(wide.capacity(7, 0, 0), 64);
        assert_eq!(wide.capacity(7, wide.rebase_after_recovery(7), 0), 8);
    }

    #[test]
    fn staleness_is_per_version() {
        let budget = StalenessBudget {
            max_staleness: 2,
            consumer_batch_size: 1,
            concurrency: 1,
        };
        assert!(!budget.is_stale(7, 5));
        assert!(budget.is_stale(7, 4));
    }

    #[test]
    fn ledger_rejects_misaligned_versions() {
        let mut ledger = VersionLedger::new();
        assert_eq!(
            ledger.record(1, 3, vec![1, 2]),
            Err(LedgerError::LengthMismatch {
                tokens: 3,
                versions: 2
            })
        );
        assert!(ledger.record(2, 2, vec![1, 2]).is_ok());
        assert_eq!(ledger.token_count(), 2);
    }

    #[test]
    fn mixed_version_trajectory_is_counted_correctly() {
        let mut ledger = VersionLedger::new();
        // A partially rolled-out trajectory: early tokens from version 3,
        // later tokens from version 5 after an in-flight weight update.
        ledger.record(1, 4, vec![3, 3, 5, 5]).unwrap();
        let budget = StalenessBudget {
            max_staleness: 1,
            consumer_batch_size: 8,
            concurrency: 32,
        };
        assert_eq!(ledger.stale_tokens(&budget, 5), 2);
        assert_eq!(ledger.versions(1), Some(&[3, 3, 5, 5][..]));
    }
}
