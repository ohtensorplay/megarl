//! Data-plane lease bookkeeping and remote weight-write primitives.
//!
//! Frames in `megarl-protocol` carry only metadata and handles. A worker owns
//! the handle while holding a lease; the controller never touches the payload.
//! Expired leases are reclaimed so a dead rollout worker cannot pin shared
//! memory forever.
//!
//! Weight updates are one-sided remote writes into registered memory regions.
//! The provider trait is the seam where a real RDMA/NIXL backend plugs in; the
//! in-process provider makes the protocol testable without RDMA hardware.

use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lease {
    pub id: u64,
    pub handle: u64,
    pub expires_at: Instant,
}

#[derive(Debug, Default)]
pub struct LeaseTable {
    next_id: u64,
    leases: HashMap<u64, Lease>,
}

impl LeaseTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn acquire(&mut self, handle: u64, ttl: Duration) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.leases.insert(
            id,
            Lease {
                id,
                handle,
                expires_at: Instant::now() + ttl,
            },
        );
        id
    }

    /// Acquire using an explicit clock, useful for deterministic tests and for
    /// driving the table from the controller's monotonic clock.
    pub fn acquire_at(&mut self, handle: u64, ttl: Duration, now: Instant) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.leases.insert(
            id,
            Lease {
                id,
                handle,
                expires_at: now + ttl,
            },
        );
        id
    }

    pub fn renew(&mut self, id: u64, ttl: Duration) -> bool {
        self.renew_at(id, ttl, Instant::now())
    }

    pub fn renew_at(&mut self, id: u64, ttl: Duration, now: Instant) -> bool {
        match self.leases.get_mut(&id) {
            Some(lease) => {
                lease.expires_at = now + ttl;
                true
            }
            None => false,
        }
    }

    pub fn release(&mut self, id: u64) -> Option<u64> {
        self.leases.remove(&id).map(|lease| lease.handle)
    }

    pub fn get(&self, id: u64) -> Option<&Lease> {
        self.leases.get(&id)
    }

    pub fn len(&self) -> usize {
        self.leases.len()
    }

    /// Returns handles whose leases expired at `now`, removing their entries.
    pub fn reclaim_expired_at(&mut self, now: Instant) -> Vec<u64> {
        let expired: Vec<u64> = self
            .leases
            .iter()
            .filter(|(_, lease)| lease.expires_at <= now)
            .map(|(&id, _)| id)
            .collect();
        expired
            .into_iter()
            .filter_map(|id| self.leases.remove(&id).map(|lease| lease.handle))
            .collect()
    }

    pub fn reclaim_expired(&mut self) -> Vec<u64> {
        self.reclaim_expired_at(Instant::now())
    }
}

/// A registered remote memory region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRegion {
    pub key: u32,
    pub size: u64,
}

/// Seam for a real one-sided RDMA backend (NIXL / ibverbs / RoCE).
/// Implementations only need to register buffers and perform remote writes.
pub trait RdmaProvider {
    fn register(&mut self, data: &[u8]) -> Result<MemoryRegion, String>;
    fn write(&mut self, region: &MemoryRegion, offset: u64, data: &[u8]) -> Result<(), String>;
    fn read(&self, region: &MemoryRegion) -> Result<Vec<u8>, String>;
    fn deregister(&mut self, region: &MemoryRegion);
}

/// In-process provider with RDMA write semantics. Used for tests and for a
/// colocated worker; a real provider swaps in behind the same trait.
#[derive(Debug, Default)]
pub struct LocalRdmaProvider {
    regions: HashMap<u32, Vec<u8>>,
    next_key: u32,
}

impl RdmaProvider for LocalRdmaProvider {
    fn register(&mut self, data: &[u8]) -> Result<MemoryRegion, String> {
        let key = self.next_key;
        self.next_key = self.next_key.wrapping_add(1);
        self.regions.insert(key, data.to_vec());
        Ok(MemoryRegion {
            key,
            size: data.len() as u64,
        })
    }

    fn write(&mut self, region: &MemoryRegion, offset: u64, data: &[u8]) -> Result<(), String> {
        let buffer = self
            .regions
            .get_mut(&region.key)
            .ok_or_else(|| format!("unknown region key {}", region.key))?;
        let start = offset as usize;
        let end = start
            .checked_add(data.len())
            .ok_or_else(|| "write overflow".to_string())?;
        if end > buffer.len() {
            return Err(format!(
                "write [{start},{end}) out of bounds for region of {} bytes",
                buffer.len()
            ));
        }
        buffer[start..end].copy_from_slice(data);
        Ok(())
    }

    fn read(&self, region: &MemoryRegion) -> Result<Vec<u8>, String> {
        self.regions
            .get(&region.key)
            .cloned()
            .ok_or_else(|| format!("unknown region key {}", region.key))
    }

    fn deregister(&mut self, region: &MemoryRegion) {
        self.regions.remove(&region.key);
    }
}

/// One tensor-rank's slice of a globally shaped tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShard {
    pub name: String,
    pub rank: u32,
    pub tp_size: u32,
    pub shard_dim: usize,
    pub shape: Vec<u64>,
    pub global_offset: u64,
    pub local_offset: u64,
    pub length: u64,
}

/// Split a tensor across tensor-parallel ranks along `shard_dim`.
/// Each rank owns one contiguous slice; no rank holds the full tensor.
pub fn plan_tp_shards(name: &str, shape: &[u64], tp_size: u32, shard_dim: usize) -> Vec<TensorShard> {
    assert!(tp_size > 0, "tp_size must be positive");
    assert!(shard_dim < shape.len(), "shard_dim out of range");
    let elements: u64 = shape.iter().product();
    assert_eq!(elements % tp_size as u64, 0, "shard dim must divide evenly");
    let per_rank = elements / tp_size as u64;
    (0..tp_size)
        .map(|rank| TensorShard {
            name: name.to_string(),
            rank,
            tp_size,
            shard_dim,
            shape: shape.to_vec(),
            global_offset: rank as u64 * per_rank,
            local_offset: 0,
            length: per_rank,
        })
        .collect()
}

/// Map a global byte range onto the rank-local ranges it touches. This is how
/// a shard-local delta is routed without a rank-0 all-gather.
pub fn map_range_to_shards(
    shards: &[TensorShard],
    offset: u64,
    length: u64,
) -> Vec<(u32, u64, u64, u64)> {
    let mut out = Vec::new();
    for shard in shards {
        let end = shard.global_offset + shard.length;
        let start = offset.max(shard.global_offset);
        let stop = (offset + length).min(end);
        if start < stop {
            out.push((
                shard.rank,
                start,
                start - shard.global_offset + shard.local_offset,
                stop - start,
            ));
        }
    }
    out
}

/// Apply a range write to a rank-local region through the RDMA provider.
pub fn write_shard<P: RdmaProvider>(
    provider: &mut P,
    region: &MemoryRegion,
    local_offset: u64,
    data: &[u8],
) -> Result<(), String> {
    provider.write(region, local_offset, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_release_roundtrip() {
        let mut table = LeaseTable::new();
        let id = table.acquire(0xABCD, Duration::from_secs(30));
        assert_eq!(table.get(id).map(|l| l.handle), Some(0xABCD));
        assert_eq!(table.release(id), Some(0xABCD));
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn reclaims_only_expired_leases() {
        let now = Instant::now();
        let mut table = LeaseTable::new();
        let old = table.acquire_at(11, Duration::from_secs(1), now);
        let fresh = table.acquire_at(22, Duration::from_secs(10), now);

        let reclaimed = table.reclaim_expired_at(now + Duration::from_secs(2));
        assert_eq!(reclaimed, vec![11]);
        assert_eq!(table.get(old), None);
        assert_eq!(table.get(fresh).map(|l| l.handle), Some(22));
    }

    #[test]
    fn renew_extends_expiry() {
        let now = Instant::now();
        let mut table = LeaseTable::new();
        let id = table.acquire_at(99, Duration::from_secs(1), now);
        assert!(table.renew_at(id, Duration::from_secs(10), now + Duration::from_millis(500)));
        assert!(table
            .reclaim_expired_at(now + Duration::from_secs(2))
            .is_empty());
        assert!(!table.renew_at(u64::MAX, Duration::from_secs(1), now));
    }

    #[test]
    fn rdma_register_write_read_roundtrip() {
        let mut provider = LocalRdmaProvider::default();
        let region = provider.register(&[0u8; 16]).unwrap();
        provider.write(&region, 4, &[1, 2, 3, 4]).unwrap();
        let data = provider.read(&region).unwrap();
        assert_eq!(&data[4..8], &[1, 2, 3, 4]);
        assert!(provider.write(&region, 14, &[1, 2, 3]).is_err());
        provider.deregister(&region);
        assert!(provider.read(&region).is_err());
    }

    #[test]
    fn tp_shards_are_disjoint_and_complete() {
        let shards = plan_tp_shards("w", &[8, 4], 4, 0);
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0].global_offset, 0);
        assert_eq!(shards[3].global_offset, 24);
        assert_eq!(shards.iter().map(|s| s.length).sum::<u64>(), 32);
        let mapped = map_range_to_shards(&shards, 7, 4);
        assert_eq!(mapped, vec![(0, 7, 7, 1), (1, 8, 0, 3)]);
    }

    #[test]
    fn shard_local_write_updates_only_own_rank() {
        let mut provider = LocalRdmaProvider::default();
        let shards = plan_tp_shards("w", &[8], 4, 0);
        let regions: Vec<MemoryRegion> = shards
            .iter()
            .map(|_| provider.register(&[0u8; 2]).unwrap())
            .collect();
        for (rank, _, local_offset, length) in map_range_to_shards(&shards, 3, 3) {
            write_shard(&mut provider, &regions[rank as usize], local_offset, &vec![9; length as usize]).unwrap();
        }
        assert_eq!(provider.read(&regions[1]).unwrap(), vec![0, 9]);
        assert_eq!(provider.read(&regions[2]).unwrap(), vec![9, 9]);
        assert_eq!(provider.read(&regions[0]).unwrap(), vec![0, 0]);
    }
}
