//! Model-version registry — the control-plane state machine that coordinates a
//! new policy version rolling out to inference workers.
//!
//! A version moves through a strict lifecycle:
//!
//! ```text
//!   publish      commit            (next publish/commit ...)
//!   ------> STAGED ------> COMMITTED ------> RETIRED
//!                  \                         (drains)
//!                   \--> ABORTED   (publish failed, never serves traffic)
//! ```
//!
//! Only the COMMITTED version's workers are eligible for **new** traffic. A
//! request captures its worker at admission time, so committing a new version
//! never re-routes in-flight work: the retired version's workers drain
//! naturally. This is the cross-process publish -> commit / in-flight abort
//! handshake AReaL relies on, expressed as an explicit, testable state machine
//! (rather than an implicit side effect of a weight reload).
//!
//! The registry is pure logic (no HTTP, no GPU): the gateway exposes it over
//! HTTP and uses [`ModelRegistry::active_workers`] to constrain routing.

use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Lifecycle state of one published model version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionState {
    /// Published, not yet serving new traffic.
    Staged,
    /// Active: new traffic routes to these workers.
    Committed,
    /// Superseded; no new traffic, in-flight work may still be draining.
    Retired,
    /// Publish failed; never became active.
    Aborted,
}

impl VersionState {
    pub fn as_str(self) -> &'static str {
        match self {
            VersionState::Staged => "staged",
            VersionState::Committed => "committed",
            VersionState::Retired => "retired",
            VersionState::Aborted => "aborted",
        }
    }
}

/// One published (model, version) pair.
#[derive(Debug, Clone)]
pub struct VersionEntry {
    pub version: u64,
    /// SHA-256 digest of the weights being served, as lowercase hex.
    pub digest: String,
    /// Workers that hold this version.
    pub workers: Vec<String>,
    pub state: VersionState,
    pub published_at_ms: u128,
}

#[derive(Debug, Clone)]
struct ModelState {
    /// Versions ordered by number so snapshots and GC are deterministic.
    versions: BTreeMap<u64, VersionEntry>,
    /// The version new traffic is routed to, if any.
    active: Option<u64>,
    /// Highest version ever published. Enforces monotonicity: a stale or
    /// replayed publish (checkpoint restore re-basing) cannot resurrect an old
    /// version or overwrite a newer one.
    highest_seen: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryError {
    /// `got` is older than a version already published for this model.
    StaleVersion { got: u64, highest: u64 },
    /// This exact version was already published.
    DuplicateVersion(u64),
    /// digest is not a 64-char lowercase SHA-256 hex string.
    BadDigest,
    /// A version must name at least one worker.
    NoWorkers,
    /// The model has no versions at all.
    UnknownModel,
    /// The version does not exist for this model.
    UnknownVersion(u64),
    /// Only a staged version can be committed or aborted.
    NotStaged(u64),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::StaleVersion { got, highest } => {
                write!(f, "version {got} is stale (highest seen is {highest})")
            }
            RegistryError::DuplicateVersion(v) => write!(f, "version {v} already published"),
            RegistryError::BadDigest => write!(f, "digest must be 64 lowercase hex chars (SHA-256)"),
            RegistryError::NoWorkers => write!(f, "a version needs at least one worker"),
            RegistryError::UnknownModel => write!(f, "unknown model"),
            RegistryError::UnknownVersion(v) => write!(f, "unknown version {v}"),
            RegistryError::NotStaged(v) => write!(f, "version {v} is not staged"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Result of an atomic commit. `retired` is the version that stopped receiving
/// new traffic; a caller can use it to coordinate draining / in-flight abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitResult {
    pub committed: u64,
    pub retired: Option<u64>,
}

/// Thread-safe registry of model versions. A single lock covers all state
/// transitions so a commit is atomic with respect to publishes, aborts and
/// routing lookups.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    models: Mutex<HashMap<String, ModelState>>,
}

fn valid_digest(digest: &str) -> bool {
    digest.len() == 64 && digest.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish a version into STAGED state. Versions must be strictly
    /// increasing per model, so a restored checkpoint cannot re-introduce an
    /// older version. Workers must be a non-empty set of addresses.
    pub fn publish(
        &self,
        model: &str,
        version: u64,
        digest: &str,
        workers: Vec<String>,
    ) -> Result<(), RegistryError> {
        if !valid_digest(digest) {
            return Err(RegistryError::BadDigest);
        }
        if workers.is_empty() {
            return Err(RegistryError::NoWorkers);
        }
        let mut models = self.models.lock().unwrap();
        let state = models.entry(model.to_string()).or_insert_with(|| ModelState {
            versions: BTreeMap::new(),
            active: None,
            highest_seen: 0,
        });
        if version < state.highest_seen {
            return Err(RegistryError::StaleVersion {
                got: version,
                highest: state.highest_seen,
            });
        }
        if state.versions.contains_key(&version) {
            return Err(RegistryError::DuplicateVersion(version));
        }
        state.versions.insert(
            version,
            VersionEntry {
                version,
                digest: digest.to_string(),
                workers,
                state: VersionState::Staged,
                published_at_ms: now_ms(),
            },
        );
        state.highest_seen = version;
        Ok(())
    }

    /// Atomically promote a staged version to active. The previously active
    /// version (if any) becomes RETIRED and is returned for drain coordination.
    pub fn commit(&self, model: &str, version: u64) -> Result<CommitResult, RegistryError> {
        let mut models = self.models.lock().unwrap();
        let state = models.get_mut(model).ok_or(RegistryError::UnknownModel)?;
        let target_state = state
            .versions
            .get(&version)
            .ok_or(RegistryError::UnknownVersion(version))?
            .state;
        if target_state != VersionState::Staged {
            return Err(RegistryError::NotStaged(version));
        }
        let retired = state.active.replace(version);
        if let Some(old) = retired {
            if let Some(old_entry) = state.versions.get_mut(&old) {
                old_entry.state = VersionState::Retired;
            }
        }
        if let Some(entry) = state.versions.get_mut(&version) {
            entry.state = VersionState::Committed;
        }
        Ok(CommitResult {
            committed: version,
            retired,
        })
    }

    /// Abort a staged publish (e.g. a failed rollout). Nothing is ever routed
    /// to a staged version, so the active version is left untouched — this is
    /// the in-flight-safe abort path.
    pub fn abort(&self, model: &str, version: u64) -> Result<(), RegistryError> {
        let mut models = self.models.lock().unwrap();
        let state = models.get_mut(model).ok_or(RegistryError::UnknownModel)?;
        let entry = state
            .versions
            .get_mut(&version)
            .ok_or(RegistryError::UnknownVersion(version))?;
        if entry.state != VersionState::Staged {
            return Err(RegistryError::NotStaged(version));
        }
        entry.state = VersionState::Aborted;
        Ok(())
    }

    pub fn active_version(&self, model: &str) -> Option<u64> {
        self.models
            .lock()
            .unwrap()
            .get(model)
            .and_then(|s| s.active)
    }

    /// Workers of the active version, for routing new traffic. Returns `None`
    /// when there is no committed version (caller then uses the full pool).
    pub fn active_workers(&self, model: &str) -> Option<Vec<String>> {
        let models = self.models.lock().unwrap();
        let state = models.get(model)?;
        let active = state.active?;
        state.versions.get(&active).map(|e| e.workers.clone())
    }

    /// JSON snapshot of a single model, including version history and active
    /// version. `None` if the model is unknown.
    pub fn snapshot(&self, model: &str) -> Option<Value> {
        let models = self.models.lock().unwrap();
        let state = models.get(model)?;
        let versions = state
            .versions
            .values()
            .map(|e| {
                json!({
                    "version": e.version,
                    "digest": e.digest,
                    "workers": e.workers,
                    "state": e.state.as_str(),
                    "published_at_ms": e.published_at_ms,
                })
            })
            .collect::<Vec<_>>();
        Some(json!({
            "model": model,
            "active": state.active,
            "highest_seen": state.highest_seen,
            "versions": versions,
        }))
    }

    /// JSON snapshot of every model, sorted by name for stable output.
    /// Builds each snapshot inline while holding the single lock — calling
    /// `snapshot()` here would try to re-lock a non-reentrant `Mutex`.
    pub fn list(&self) -> Value {
        let models = self.models.lock().unwrap();
        let mut names: Vec<_> = models.keys().cloned().collect();
        names.sort();
        let snapshots = names
            .iter()
            .filter_map(|name| {
                let state = models.get(name)?;
                let versions = state
                    .versions
                    .values()
                    .map(|e| {
                        json!({
                            "version": e.version,
                            "digest": e.digest,
                            "workers": e.workers,
                            "state": e.state.as_str(),
                            "published_at_ms": e.published_at_ms,
                        })
                    })
                    .collect::<Vec<_>>();
                Some(json!({
                    "model": name,
                    "active": state.active,
                    "highest_seen": state.highest_seen,
                    "versions": versions,
                }))
            })
            .collect::<Vec<_>>();
        Value::Array(snapshots)
    }

    /// Drop terminal (RETIRED / ABORTED) versions beyond the `keep` newest ones,
    /// keeping STAGED and the active COMMITTED version untouched. Returns the
    /// number of entries removed. This bounds registry memory over long runs of
    /// publishes without disturbing traffic.
    pub fn gc(&self, model: &str, keep: usize) -> usize {
        let mut models = self.models.lock().unwrap();
        let Some(state) = models.get_mut(model) else {
            return 0;
        };
        let terminal: Vec<u64> = state
            .versions
            .values()
            .filter(|e| matches!(e.state, VersionState::Retired | VersionState::Aborted))
            .map(|e| e.version)
            .collect();
        if terminal.len() <= keep {
            return 0;
        }
        // Keep the `keep` highest-numbered terminal versions.
        let drop_count = terminal.len() - keep;
        let to_drop: Vec<u64> = terminal.into_iter().take(drop_count).collect();
        for v in to_drop {
            state.versions.remove(&v);
        }
        drop_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(n: u8) -> String {
        format!("{:02x}", n).repeat(32)
    }

    #[test]
    fn publish_commit_keeps_monotonic_active_version() {
        let reg = ModelRegistry::new();
        assert_eq!(reg.active_version("policy"), None);
        reg.publish("policy", 3, &digest(1), vec!["w1".into(), "w2".into()])
            .unwrap();
        assert_eq!(reg.active_version("policy"), None);
        assert_eq!(
            reg.commit("policy", 3).unwrap(),
            CommitResult {
                committed: 3,
                retired: None
            }
        );
        assert_eq!(reg.active_version("policy"), Some(3));
        assert_eq!(reg.active_workers("policy").unwrap(), vec!["w1", "w2"]);

        reg.publish("policy", 4, &digest(2), vec!["w2".into()])
            .unwrap();
        let result = reg.commit("policy", 4).unwrap();
        assert_eq!(result, CommitResult { committed: 4, retired: Some(3) });
        assert_eq!(reg.active_workers("policy").unwrap(), vec!["w2"]);
    }

    #[test]
    fn abort_does_not_disturb_active_version() {
        let reg = ModelRegistry::new();
        reg.publish("p", 1, &digest(1), vec!["a".into()]).unwrap();
        reg.commit("p", 1).unwrap();
        reg.publish("p", 2, &digest(2), vec!["b".into()]).unwrap();
        assert_eq!(reg.abort("p", 2), Ok(()));
        assert_eq!(reg.active_version("p"), Some(1));
        assert_eq!(reg.active_workers("p").unwrap(), vec!["a"]);
        // A committed or aborted version cannot be committed again.
        assert_eq!(reg.commit("p", 2), Err(RegistryError::NotStaged(2)));
        assert_eq!(reg.commit("p", 1), Err(RegistryError::NotStaged(1)));
    }

    #[test]
    fn stale_and_duplicate_publishes_are_rejected() {
        let reg = ModelRegistry::new();
        reg.publish("p", 5, &digest(1), vec!["a".into()]).unwrap();
        assert_eq!(
            reg.publish("p", 4, &digest(2), vec!["b".into()]),
            Err(RegistryError::StaleVersion { got: 4, highest: 5 })
        );
        assert_eq!(
            reg.publish("p", 5, &digest(3), vec!["c".into()]),
            Err(RegistryError::DuplicateVersion(5))
        );
        assert_eq!(reg.active_workers("p"), None); // never committed
    }

    #[test]
    fn invalid_publishes_are_rejected() {
        let reg = ModelRegistry::new();
        assert_eq!(
            reg.publish("p", 1, "not-a-sha256", vec!["a".into()]),
            Err(RegistryError::BadDigest)
        );
        assert_eq!(
            reg.publish("p", 1, &digest(1), vec![]),
            Err(RegistryError::NoWorkers)
        );
    }

    #[test]
    fn list_does_not_deadlock_and_reports_all_models() {
        let reg = ModelRegistry::new();
        reg.publish("a", 1, &digest(1), vec!["w1".into()]).unwrap();
        reg.commit("a", 1).unwrap();
        reg.publish("b", 1, &digest(2), vec!["w2".into()]).unwrap();
        let all = reg.list();
        let arr = all.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["model"], "a");
        assert_eq!(arr[1]["model"], "b");
    }

    #[test]
    fn snapshot_reports_lifecycle_and_gc_bounds_history() {
        let reg = ModelRegistry::new();
        for v in 1..=4 {
            reg.publish("p", v, &digest(v as u8), vec![format!("w{v}")])
                .unwrap();
            if v != 2 {
                reg.commit("p", v).unwrap();
            } else {
                reg.abort("p", v).unwrap();
            }
        }
        let snap = reg.snapshot("p").unwrap();
        assert_eq!(snap["active"], 4);
        let states: Vec<_> = snap["versions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["state"].as_str().unwrap())
            .collect();
        assert_eq!(states, vec!["retired", "aborted", "retired", "committed"]);
        // keep=1 terminal version; there are 3 terminal entries -> drop 2.
        assert_eq!(reg.gc("p", 1), 2);
        let snap = reg.snapshot("p").unwrap();
        assert_eq!(snap["versions"].as_array().unwrap().len(), 2); // v3 retired, v4 active
    }
}
