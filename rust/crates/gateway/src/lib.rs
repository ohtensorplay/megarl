//! Rust rollout gateway.
//!
//! The HTTP frontend owns parsing, validation, per-token logprob transport,
//! routing and weight-update coordination. GPU work stays in Python workers
//! behind the versioned MGRL frame protocol (the same split megaLM uses for
//! its axum server + Python worker). No GIL, no tensor serialization through
//! the gateway.

use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{routing::{get, post}, Json, Router};
use megarl_control::TokenBucketRateLimiter;
use megarl_protocol::{
    decode, encode, validate_payload, Kind, LifecycleAction, LifecycleAck, LifecycleCommand,
    LifecyclePhase, WeightUpdateAck,
};
use megarl_registry::{ModelRegistry, RegistryError};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tensorplay_observability::render_prometheus_metrics;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

#[derive(Debug, Default)]
pub struct Metrics {
    pub generates: AtomicU64,
    pub weight_reloads: AtomicU64,
    pub errors: AtomicU64,
    pub retries: AtomicU64,
    pub auth_rejected: AtomicU64,
    pub in_flight: AtomicU64,
    pub max_frame_bytes: AtomicU64,
    /// Sessions moved off their sticky worker because load became skewed.
    pub rebalances: AtomicU64,
    /// Requests rejected by the edge rate limiter.
    pub rate_rejected: AtomicU64,
    /// End-to-end weight-update forward time, measured at the gateway.
    pub weight_reload_ms: AtomicU64,
    pub weight_reload_total_ms: AtomicU64,
    pub weight_reload_max_ms: AtomicU64,
    /// Last worker-reported weight-reload phase breakdown (Q6).
    pub weight_reload_route_ms: AtomicU64,
    pub weight_reload_pause_ms: AtomicU64,
    pub weight_reload_barrier_ms: AtomicU64,
    pub weight_reload_transfer_ms: AtomicU64,
    pub weight_reload_apply_ms: AtomicU64,
    pub weight_reload_phase_samples: AtomicU64,
    /// Model-version registry lifecycle transitions.
    pub model_publishes: AtomicU64,
    pub model_commits: AtomicU64,
    pub model_aborts: AtomicU64,
    pub model_rejected: AtomicU64,
    /// Rollout engine pause/resume orchestration.
    pub lifecycle_pauses: AtomicU64,
    pub lifecycle_resumes: AtomicU64,
    pub lifecycle_errors: AtomicU64,
    pub lifecycle_pause_ms: AtomicU64,
    pub lifecycle_resume_ms: AtomicU64,
}

/// Active liveness state of one Python worker, maintained by `monitor`.
#[derive(Clone, Debug, Default)]
pub struct WorkerHealth {
    pub healthy: bool,
    pub failures: u64,
    pub successes: u64,
    pub last_error: Option<String>,
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Colocate lifecycle state of one worker. Only `running` workers are eligible
/// for generation traffic; transitional phases fence admission while the
/// engine drains and releases / restores GPU memory.
#[derive(Clone, Debug)]
pub struct WorkerLifecycle {
    pub phase: LifecyclePhase,
    pub epoch: u64,
    pub error: Option<String>,
    pub changed_at_ms: u128,
}

impl Default for WorkerLifecycle {
    fn default() -> Self {
        Self {
            phase: LifecyclePhase::Running,
            epoch: 0,
            error: None,
            changed_at_ms: now_ms(),
        }
    }
}

/// Pool-wide lifecycle state. `epoch` guards acks so a delayed reply from an
/// earlier transition cannot confirm a newer one.
#[derive(Debug, Default)]
pub struct LifecycleState {
    pub epoch: u64,
    pub paused: bool,
    pub offloaded: bool,
    pub workers: HashMap<String, WorkerLifecycle>,
}

#[derive(Clone, Debug)]
pub struct AppState {
    /// Python GPU workers. The list is mutable so workers can join or leave a
    /// running gateway; requests sharing a session_id are sticky so radix/KV
    /// locality is preserved.
    pub workers: Arc<RwLock<Vec<String>>>,
    pub metrics: Arc<Metrics>,
    pub auth_token: Option<String>,
    pub worker_timeout: Duration,
    pub max_frame_bytes: usize,
    next_worker: Arc<AtomicU64>,
    /// Monotonic nonce for health probes; a stale reply is rejected.
    health_nonce: Arc<AtomicU64>,
    health_interval: Duration,
    health_timeout: Duration,
    health: Arc<Mutex<HashMap<String, WorkerHealth>>>,
    /// Routing counts per worker, used to detect sticky-session skew.
    usage: Arc<Mutex<HashMap<String, u64>>>,
    /// Rebalance a sticky session when its worker exceeds this multiple of the
    /// least-used worker. Disabled by setting it <= 1.0.
    skew_ratio: f64,
    /// ...and only after at least this many extra assignments, so short-lived
    /// sessions keep their KV locality.
    skew_min_gap: u64,
    /// Optional edge admission cap. Independent of per-worker concurrency: it
    /// bounds requests/sec before any worker is touched.
    rate_limiter: Arc<Mutex<Option<TokenBucketRateLimiter>>>,
    /// Model-version registry: publish -> commit / in-flight abort state
    /// machine. New traffic is constrained to the active version's workers.
    pub registry: Arc<ModelRegistry>,
    /// Colocate engine lifecycle. Guarded by one mutex so a transition is
    /// atomic w.r.t. routing and observability; the separate async mutex below
    /// serializes transitions across requests (never held with a std guard).
    pub lifecycle: Arc<Mutex<LifecycleState>>,
    lifecycle_transition: Arc<tokio::sync::Mutex<()>>,
}

impl AppState {
    pub fn new(workers: Vec<String>) -> Self {
        let token = std::env::var("MEGARL_GATEWAY_TOKEN").ok().filter(|s| !s.is_empty());
        Self::with_token(workers, token)
    }

    pub fn with_token(workers: Vec<String>, auth_token: Option<String>) -> Self {
        assert!(!workers.is_empty(), "gateway needs at least one worker");
        let health = Arc::new(Mutex::new(
            workers
                .iter()
                .map(|addr| {
                    (
                        addr.clone(),
                        WorkerHealth {
                            healthy: true,
                            ..WorkerHealth::default()
                        },
                    )
                })
                .collect::<HashMap<_, _>>(),
        ));
        let usage = Arc::new(Mutex::new(
            workers.iter().map(|addr| (addr.clone(), 0u64)).collect::<HashMap<_, _>>(),
        ));
        let rate_limiter = Arc::new(Mutex::new(
            std::env::var("MEGARL_GATEWAY_RATE_PER_SEC")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(|rate| {
                    let burst = std::env::var("MEGARL_GATEWAY_RATE_BURST")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(64);
                    TokenBucketRateLimiter::new(rate, burst)
                }),
        ));
        let lifecycle_workers: HashMap<String, WorkerLifecycle> = workers
            .iter()
            .map(|addr| (addr.clone(), WorkerLifecycle::default()))
            .collect();
        Self {
            workers: Arc::new(RwLock::new(workers)),
            metrics: Arc::new(Metrics::default()),
            auth_token,
            worker_timeout: Duration::from_secs(
                std::env::var("MEGARL_WORKER_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(120),
            ),
            max_frame_bytes: std::env::var("MEGARL_MAX_FRAME_BYTES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64 * 1024 * 1024),
            next_worker: Arc::new(AtomicU64::new(0)),
            health_nonce: Arc::new(AtomicU64::new(1)),
            health_interval: Duration::from_millis(
                std::env::var("MEGARL_HEALTH_INTERVAL_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(5_000),
            ),
            health_timeout: Duration::from_millis(
                std::env::var("MEGARL_HEALTH_TIMEOUT_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2_000),
            ),
            health,
            usage,
            skew_ratio: std::env::var("MEGARL_SKEW_RATIO")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2.0),
            skew_min_gap: std::env::var("MEGARL_SKEW_MIN_GAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            rate_limiter,
            registry: Arc::new(ModelRegistry::new()),
            lifecycle: Arc::new(Mutex::new(LifecycleState {
                workers: lifecycle_workers,
                ..LifecycleState::default()
            })),
            lifecycle_transition: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// Admits one request through the edge token bucket.
    /// Returns the retry delay in ms when the bucket is empty.
    pub fn admit(&self) -> Result<(), u64> {
        let mut guard = self.rate_limiter.lock().unwrap();
        let Some(limiter) = guard.as_mut() else {
            return Ok(());
        };
        if limiter.try_acquire(1) {
            Ok(())
        } else {
            Err(limiter.retry_after_ms(1).max(1))
        }
    }

    /// Sticky by session, round-robin otherwise. Unhealthy workers are skipped
    /// unless the entire pool is down (then we still attempt a request).
    ///
    /// A sticky session is moved to the least-loaded healthy worker once its
    /// pinned worker is badly skewed. KV locality is worth preserving for
    /// normal traffic, so rebalancing needs BOTH a ratio and an absolute gap
    /// of extra assignments to fire.
    pub fn worker_for(&self, session: Option<u64>) -> String {
        self.worker_for_in(session, None)
            .expect("gateway has no routable worker")
    }

    /// Route new traffic for a model: restrict the pool to workers that hold
    /// the active (committed) version when one is registered. If the model has
    /// no committed version, fall back to the full worker pool.
    pub fn worker_for_model(&self, session: Option<u64>, model: &str) -> String {
        let allowed = self.registry.active_workers(model);
        self.worker_for_in(session, allowed)
            .expect("gateway has no routable worker")
    }

    /// Fallible routing used by HTTP handlers so a paused pool yields 503
    /// instead of a connection error to an empty address.
    pub fn try_route(&self, session: Option<u64>, model: Option<&str>) -> Option<String> {
        let allowed = model.and_then(|name| self.registry.active_workers(name));
        self.worker_for_in(session, allowed)
    }

    fn worker_for_in(&self, session: Option<u64>, allowed: Option<Vec<String>>) -> Option<String> {
        let workers = self.workers.read().unwrap();
        assert!(!workers.is_empty(), "gateway has no workers");
        let health = self.health.lock().unwrap();
        let lifecycle = self.lifecycle.lock().unwrap();
        let routable = |addr: &String| {
            // A pool-wide pause fences every worker, including ones that were
            // registered before the lifecycle map existed for them.
            !lifecycle.paused
                && lifecycle
                    .workers
                    .get(addr)
                    .map(|w| w.phase == LifecyclePhase::Running)
                    .unwrap_or(true)
                && allowed.as_ref().map(|a| a.contains(addr)).unwrap_or(true)
        };
        let alive: Vec<&String> = workers
            .iter()
            .filter(|addr| {
                routable(addr) && health.get(*addr).map(|h| h.healthy).unwrap_or(true)
            })
            .collect();
        let pool = if alive.is_empty() {
            // Only fall back among workers that are still lifecycle-routable;
            // never send generation traffic to a paused/offloaded worker.
            let fallback: Vec<&String> = workers.iter().filter(|addr| routable(addr)).collect();
            if fallback.is_empty() {
                return None;
            }
            fallback
        } else {
            alive
        };
        let mut index = match session {
            Some(id) => id % pool.len() as u64,
            None => self.next_worker.fetch_add(1, Ordering::Relaxed) % pool.len() as u64,
        };
        let mut usage = self.usage.lock().unwrap();
        if session.is_some() && self.skew_ratio > 1.0 && pool.len() > 1 {
            let least_index = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, addr)| usage.get(addr.as_str()).copied().unwrap_or(0))
                .map(|(i, _)| i)
                .unwrap();
            let pinned_usage = usage.get(pool[index as usize].as_str()).copied().unwrap_or(0);
            let least_usage = usage
                .get(pool[least_index].as_str())
                .copied()
                .unwrap_or(0);
            let skewed = least_usage >= 1
                && pinned_usage.saturating_sub(least_usage) >= self.skew_min_gap
                && (pinned_usage as f64) > (least_usage as f64) * self.skew_ratio;
            if skewed {
                index = least_index as u64;
                self.metrics.rebalances.fetch_add(1, Ordering::Relaxed);
            }
        }
        let chosen = pool[index as usize].clone();
        *usage.entry(chosen.clone()).or_insert(0) += 1;
        Some(chosen)
    }

    /// Per-worker assignment counts, for observability and skew debugging.
    pub fn usage_json(&self) -> Value {
        // Lock order is always workers -> health -> usage; never invert it.
        let workers = self.workers.read().unwrap();
        let usage = self.usage.lock().unwrap();
        let counts: Vec<Value> = workers
            .iter()
            .map(|addr| {
                json!({
                    "address": addr,
                    "assigned": usage.get(addr.as_str()).copied().unwrap_or(0),
                })
            })
            .collect();
        json!({
            "workers": counts,
            "rebalances": self.metrics.rebalances.load(Ordering::Relaxed),
        })
    }

    /// A healthy worker other than `primary`, for failover.
    pub fn fallback_for(&self, primary: &str) -> Option<String> {
        let workers = self.workers.read().unwrap();
        let health = self.health.lock().unwrap();
        workers
            .iter()
            .find(|addr| addr.as_str() != primary && health.get(*addr).map(|h| h.healthy).unwrap_or(true))
            .cloned()
    }

    pub fn mark_failure(&self, addr: &str, error: impl Into<String>) {
        if let Some(entry) = self.health.lock().unwrap().get_mut(addr) {
            entry.healthy = false;
            entry.failures += 1;
            entry.last_error = Some(error.into());
        }
    }

    pub fn mark_success(&self, addr: &str) {
        if let Some(entry) = self.health.lock().unwrap().get_mut(addr) {
            entry.healthy = true;
            entry.successes += 1;
            entry.last_error = None;
        }
    }

    /// Adds a worker at runtime. Returns false if it is already registered.
    /// Health and usage state start fresh so the probe monitor can vet it
    /// before it receives traffic.
    pub fn register(&self, addr: String) -> bool {
        if addr.trim().is_empty() {
            return false;
        }
        let mut workers = self.workers.write().unwrap();
        if workers.contains(&addr) {
            return false;
        }
        workers.push(addr.clone());
        drop(workers);
        self.health.lock().unwrap().entry(addr.clone()).or_insert_with(|| WorkerHealth {
            healthy: true,
            ..WorkerHealth::default()
        });
        // A worker joining while the pool is paused inherits the paused phase;
        // otherwise it would immediately receive traffic from a fenced pool.
        {
            let mut lifecycle = self.lifecycle.lock().unwrap();
            let phase = if lifecycle.offloaded {
                LifecyclePhase::Offloaded
            } else if lifecycle.paused {
                LifecyclePhase::Paused
            } else {
                LifecyclePhase::Running
            };
            let epoch = lifecycle.epoch;
            lifecycle.workers.entry(addr.clone()).or_insert_with(|| WorkerLifecycle {
                phase,
                epoch,
                ..WorkerLifecycle::default()
            });
        }
        self.usage.lock().unwrap().entry(addr).or_insert(0);
        true
    }

    /// Removes a worker at runtime, dropping its health and usage state.
    pub fn deregister(&self, addr: &str) -> bool {
        let mut workers = self.workers.write().unwrap();
        // Routing needs at least one worker; refuse to empty the pool.
        if workers.len() <= 1 {
            return false;
        }
        let before = workers.len();
        workers.retain(|existing| existing != addr);
        let removed = workers.len() != before;
        drop(workers);
        if removed {
            self.health.lock().unwrap().remove(addr);
            self.lifecycle.lock().unwrap().workers.remove(addr);
            self.usage.lock().unwrap().remove(addr);
        }
        removed
    }

    pub fn worker_list(&self) -> Vec<String> {
        self.workers.read().unwrap().clone()
    }

    pub fn health_json(&self) -> Value {
        let workers_guard = self.workers.read().unwrap();
        let health = self.health.lock().unwrap();
        let lifecycle = self.lifecycle.lock().unwrap();
        let workers: Vec<Value> = workers_guard
            .iter()
            .map(|addr| {
                let entry = health.get(addr);
                json!({
                    "address": addr,
                    "healthy": entry.map(|h| h.healthy).unwrap_or(true),
                    "failures": entry.map(|h| h.failures).unwrap_or(0),
                    "successes": entry.map(|h| h.successes).unwrap_or(0),
                    "last_error": entry.and_then(|h| h.last_error.clone()),
                    "phase": lifecycle.workers.get(addr)
                        .map(|w| w.phase.as_str()).unwrap_or("running"),
                })
            })
            .collect();
        let healthy = workers.iter().filter(|w| w["healthy"] == true).count();
        json!({
            "healthy": healthy,
            "total": workers.len(),
            "workers": workers,
        })
    }

    /// No worker is eligible for generation (every worker is paused, pausing,
    /// offloaded or resuming).
    pub fn all_paused(&self) -> bool {
        let lifecycle = self.lifecycle.lock().unwrap();
        lifecycle
            .workers
            .values()
            .all(|worker| worker.phase != LifecyclePhase::Running)
    }

    /// Snapshot of the pool-wide lifecycle state for `/rollout/lifecycle` and
    /// `/metrics`.
    pub fn lifecycle_json(&self) -> Value {
        let lifecycle = self.lifecycle.lock().unwrap();
        let mut workers: Vec<Value> = lifecycle
            .workers
            .iter()
            .map(|(addr, worker)| {
                json!({
                    "address": addr,
                    "phase": worker.phase.as_str(),
                    "epoch": worker.epoch,
                    "error": worker.error,
                    "changed_at_ms": worker.changed_at_ms,
                })
            })
            .collect();
        workers.sort_by(|a, b| a["address"].as_str().cmp(&b["address"].as_str()));
        json!({
            "epoch": lifecycle.epoch,
            "paused": lifecycle.paused,
            "offloaded": lifecycle.offloaded,
            "routable": lifecycle.workers.values()
                .filter(|w| w.phase == LifecyclePhase::Running).count(),
            "workers": workers,
        })
    }

    fn authorized(&self, headers: &HeaderMap) -> bool {
        match &self.auth_token {
            None => true,
            Some(expected) => headers
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .map(|v| v.strip_prefix("Bearer ").unwrap_or(v) == expected)
                .unwrap_or(false),
        }
    }
}

fn bad_request(message: impl Into<String>) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": message.into()}))).into_response()
}

/// Send one MGRL frame to a Python rollout worker and collect reply frames.
/// Replies are length-prefixed; a zero length terminates the response.
async fn forward(
    worker_addr: &str,
    frame: Vec<u8>,
    limit: Duration,
    max_frame: usize,
) -> Result<Vec<Vec<u8>>, String> {
    let work = async {
        let mut stream = TcpStream::connect(worker_addr)
            .await
            .map_err(|e| format!("worker connect failed: {e}"))?;
        let len = frame.len() as u32;
        stream
            .write_all(&len.to_be_bytes())
            .await
            .map_err(|e| format!("worker write failed: {e}"))?;
        stream
            .write_all(&frame)
            .await
            .map_err(|e| format!("worker write failed: {e}"))?;

        let mut replies = Vec::new();
        loop {
            let mut buf = [0u8; 4];
            stream
                .read_exact(&mut buf)
                .await
                .map_err(|e| format!("worker read failed: {e}"))?;
            let size = u32::from_be_bytes(buf) as usize;
            if size == 0 {
                break;
            }
            if size > max_frame {
                return Err(format!("worker frame too large: {size} > {max_frame}"));
            }
            let mut payload = vec![0u8; size];
            stream
                .read_exact(&mut payload)
                .await
                .map_err(|e| format!("worker read failed: {e}"))?;
            replies.push(payload);
        }
        Ok(replies)
    };
    timeout(limit, work)
        .await
        .map_err(|_| format!("worker timeout after {:?}", limit))?
}

/// Send a nonce-bound liveness frame and require an echo from the worker.
/// A TCP connect alone is insufficient: a wedged event loop still accepts.
async fn probe_worker(
    addr: &str,
    nonce: u64,
    limit: Duration,
    max_frame: usize,
) -> Result<(), String> {
    let work = async {
        let mut stream = TcpStream::connect(addr)
            .await
            .map_err(|e| format!("connect failed: {e}"))?;
        let frame = encode(
            Kind::HealthCheck,
            &json!({"probe": nonce, "ok": true}),
        )
        .map_err(|e| format!("encode failed: {e}"))?;
        stream
            .write_all(&(frame.len() as u32).to_be_bytes())
            .await
            .map_err(|e| format!("write failed: {e}"))?;
        stream
            .write_all(&frame)
            .await
            .map_err(|e| format!("write failed: {e}"))?;

        let mut buf = [0u8; 4];
        stream
            .read_exact(&mut buf)
            .await
            .map_err(|e| format!("read failed: {e}"))?;
        let size = u32::from_be_bytes(buf) as usize;
        if size == 0 {
            return Err("worker closed without a health reply".into());
        }
        if size > max_frame {
            return Err(format!("health frame too large: {size}"));
        }
        let mut payload = vec![0u8; size];
        stream
            .read_exact(&mut payload)
            .await
            .map_err(|e| format!("read failed: {e}"))?;
        let (kind, value) = decode(&payload).map_err(|e| format!("health frame: {e}"))?;
        if kind != Kind::HealthCheck {
            return Err(format!("unexpected health reply kind: {}", kind.as_str()));
        }
        if value.get("probe").and_then(Value::as_u64) != Some(nonce) {
            return Err("health reply nonce mismatch".into());
        }
        if value.get("ok").and_then(Value::as_bool) != Some(true) {
            return Err("worker reported unhealthy".into());
        }
        Ok(())
    };
    timeout(limit, work)
        .await
        .map_err(|_| format!("health probe timeout after {limit:?}"))?
}

/// Background liveness monitor. Workers failing a probe are excluded from
/// routing until a later probe succeeds (automatic recovery).
pub async fn monitor(state: AppState) {
    let mut ticker = tokio::time::interval(state.health_interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        let nonce = state.health_nonce.fetch_add(1, Ordering::Relaxed);
        for addr in state.worker_list() {
            let state = state.clone();
            let addr = addr.clone();
            tokio::spawn(async move {
                match probe_worker(&addr, nonce, state.health_timeout, state.max_frame_bytes).await {
                    Ok(()) => state.mark_success(&addr),
                    Err(error) => {
                        state.mark_failure(&addr, error);
                        state.metrics.errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
    }
}

pub async fn health(State(state): State<AppState>) -> Response {
    let summary = state.health_json();
    let ok = summary["healthy"].as_u64().unwrap_or(0) > 0;
    Json(json!({
        "ok": ok,
        "healthy": summary["healthy"],
        "total": summary["total"],
        "workers": summary["workers"],
    }))
    .into_response()
}

/// The snapshot behind `/metrics`; health details stay on `/health`.
fn metrics_snapshot(state: &AppState) -> Value {
    let health = state.health_json();
    let usage = state.usage_json();
    json!({
        "generates": state.metrics.generates.load(Ordering::Relaxed),
        "weight_reloads": state.metrics.weight_reloads.load(Ordering::Relaxed),
        "errors": state.metrics.errors.load(Ordering::Relaxed),
        "retries": state.metrics.retries.load(Ordering::Relaxed),
        "auth_rejected": state.metrics.auth_rejected.load(Ordering::Relaxed),
        "in_flight": state.metrics.in_flight.load(Ordering::Relaxed),
        "rebalances": state.metrics.rebalances.load(Ordering::Relaxed),
        "rate_rejected": state.metrics.rate_rejected.load(Ordering::Relaxed),
        "weight_reload_ms": state.metrics.weight_reload_ms.load(Ordering::Relaxed),
        "weight_reload_total_ms": state.metrics.weight_reload_total_ms.load(Ordering::Relaxed),
        "weight_reload_max_ms": state.metrics.weight_reload_max_ms.load(Ordering::Relaxed),
        "weight_reload_route_ms": state.metrics.weight_reload_route_ms.load(Ordering::Relaxed),
        "weight_reload_pause_ms": state.metrics.weight_reload_pause_ms.load(Ordering::Relaxed),
        "weight_reload_barrier_ms": state.metrics.weight_reload_barrier_ms.load(Ordering::Relaxed),
        "weight_reload_transfer_ms": state.metrics.weight_reload_transfer_ms.load(Ordering::Relaxed),
        "weight_reload_apply_ms": state.metrics.weight_reload_apply_ms.load(Ordering::Relaxed),
        "weight_reload_phase_samples": state.metrics.weight_reload_phase_samples.load(Ordering::Relaxed),
        "model_publishes": state.metrics.model_publishes.load(Ordering::Relaxed),
        "model_commits": state.metrics.model_commits.load(Ordering::Relaxed),
        "model_aborts": state.metrics.model_aborts.load(Ordering::Relaxed),
        "model_rejected": state.metrics.model_rejected.load(Ordering::Relaxed),
        "lifecycle_pauses": state.metrics.lifecycle_pauses.load(Ordering::Relaxed),
        "lifecycle_resumes": state.metrics.lifecycle_resumes.load(Ordering::Relaxed),
        "lifecycle_errors": state.metrics.lifecycle_errors.load(Ordering::Relaxed),
        "lifecycle_pause_ms": state.metrics.lifecycle_pause_ms.load(Ordering::Relaxed),
        "lifecycle_resume_ms": state.metrics.lifecycle_resume_ms.load(Ordering::Relaxed),
        "lifecycle": state.lifecycle_json(),
        "models": state.registry.list(),
        "workers": state.worker_list().len(),
        "healthy_workers": health["healthy"],
        "worker_health": health["workers"],
        "worker_usage": usage["workers"],
    })
}

/// GET /metrics — Prometheus text exposition 0.0.4. Scrapers (Prometheus,
/// VictoriaMetrics, Grafana, ...) consume this directly; the renderer adds
/// HELP/TYPE, native labels and real histograms.
pub async fn prometheus_metrics(State(state): State<AppState>) -> Response {
    let snapshot = metrics_snapshot(&state).to_string();
    // A `serde_json::Value` produced by `json!` always serializes, so a failure
    // here would be a programming error rather than bad input.
    let text = render_prometheus_metrics(&snapshot, "megarl")
        .expect("metrics snapshot is valid JSON");
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        text,
    )
        .into_response()
}

pub async fn generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    if let Err(retry_ms) = state.admit() {
        state.metrics.rate_rejected.fetch_add(1, Ordering::Relaxed);
        let mut response = (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error": "rate limited", "retry_after_ms": retry_ms})),
        )
            .into_response();
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, retry_ms.to_string().parse().unwrap());
        return response;
    }
    let text = payload.to_string();
    if text.len() > state.max_frame_bytes {
        return bad_request("payload too large");
    }
    if let Err(e) = validate_payload(Kind::RolloutRequest, &text) {
        state.metrics.errors.fetch_add(1, Ordering::Relaxed);
        return bad_request(format!("schema: {e}"));
    }
    let frame = match encode(Kind::RolloutRequest, &payload) {
        Ok(frame) => frame,
        Err(e) => return bad_request(format!("frame: {e}")),
    };
    let session = payload.get("session_id").and_then(Value::as_u64);
    state.metrics.in_flight.fetch_add(1, Ordering::Relaxed);
    // Model-aware routing: when a model is named, new traffic is constrained to
    // workers holding its active version. In-flight requests already captured
    // their worker and are unaffected by a later commit.
    // A paused / offloaded pool has no routable worker -> 503 rather than a
    // dead socket. Model routing still constrains to the active version.
    let mut used = match state.try_route(
        session,
        payload.get("model").and_then(Value::as_str),
    ) {
        Some(worker) => worker,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "rollout paused; no routable worker"})),
            )
                .into_response();
        }
    };
    let mut outcome = forward(&used, frame.clone(), state.worker_timeout, state.max_frame_bytes).await;
    if let Err(error) = &outcome {
        state.mark_failure(&used, error.clone());
        if let Some(fallback) = state.fallback_for(&used) {
            state.metrics.retries.fetch_add(1, Ordering::Relaxed);
            used = fallback;
            outcome = forward(&used, frame, state.worker_timeout, state.max_frame_bytes).await;
        }
    }
    match &outcome {
        Ok(_) => state.mark_success(&used),
        Err(error) => state.mark_failure(&used, error.clone()),
    }
    state.metrics.in_flight.fetch_sub(1, Ordering::Relaxed);
    let replies = match outcome {
        Ok(replies) => replies,
        Err(e) => {
            state.metrics.errors.fetch_add(1, Ordering::Relaxed);
            return bad_request(e);
        }
    };
    let mut records = Vec::new();
    for reply in replies {
        match decode(&reply) {
            Ok((kind, value)) if kind == Kind::TrajectoryRecord => records.push(value),
            Ok((kind, _)) => {
                state.metrics.errors.fetch_add(1, Ordering::Relaxed);
                return bad_request(format!("unexpected reply kind: {}", kind.as_str()));
            }
            Err(e) => {
                state.metrics.errors.fetch_add(1, Ordering::Relaxed);
                return bad_request(format!("reply frame: {e}"));
            }
        }
    }
    state.metrics.generates.fetch_add(1, Ordering::Relaxed);
    Json(json!({"records": records})).into_response()
}

pub async fn update_weights(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    let route_started = std::time::Instant::now();
    let path = match payload.get("path").and_then(Value::as_str) {
        Some(path) => path.to_string(),
        None => return bad_request("missing path"),
    };
    let message = json!({
        "from_version": 0,
        "to_version": 0,
        "transport": "fs",
        "digest": payload.get("expected_digest").and_then(Value::as_str).unwrap_or(""),
        "tensors": [],
        "abort_inflight": payload.get("abort_inflight").and_then(Value::as_bool).unwrap_or(false),
        "path": path,
    });
    // The Python worker handles the cold path today; the weight-update
    // coordinator will speak tensor/collective sources here.
    let frame = match encode(Kind::WeightUpdate, &message) {
        Ok(frame) => frame,
        Err(e) => return bad_request(format!("frame: {e}")),
    };
    // `deny_unknown_fields` rejects the extra `path` key, so validate a
    // projection before encoding.
    let projection = json!({
        "from_version": 0, "to_version": 0, "transport": "fs",
        "digest": message["digest"], "tensors": [],
        "abort_inflight": message["abort_inflight"],
    });
    if let Err(e) = validate_payload(Kind::WeightUpdate, &projection.to_string()) {
        return bad_request(format!("schema: {e}"));
    }
    let worker = state.worker_for(None);
    let route_ms = route_started.elapsed().as_millis() as u64;
    let started = std::time::Instant::now();
    let outcome = forward(&worker, frame, state.worker_timeout, state.max_frame_bytes).await;
    let wire_ms = started.elapsed().as_millis() as u64;
    state.metrics.weight_reload_ms.store(wire_ms, Ordering::Relaxed);
    state.metrics
        .weight_reload_total_ms
        .fetch_add(wire_ms, Ordering::Relaxed);
    state.metrics
        .weight_reload_max_ms
        .fetch_max(wire_ms, Ordering::Relaxed);
    state.metrics
        .weight_reload_route_ms
        .store(route_ms, Ordering::Relaxed);

    // The worker echoes a `WeightUpdateAck` before the terminating zero frame.
    // Absence is fine (older workers): all phase counters then stay at zero.
    let ack = outcome.as_ref().ok().and_then(|replies| {
        replies.iter().rev().find_map(|frame| {
            let (kind, value) = decode(frame).ok()?;
            if kind == Kind::WeightUpdateAck {
                serde_json::from_value::<WeightUpdateAck>(value).ok()
            } else {
                None
            }
        })
    });
    let phases = if let Some(ack) = &ack {
        let transfer_ms = wire_ms.saturating_sub(ack.apply_ms.saturating_add(ack.barrier_ms));
        state.metrics
            .weight_reload_pause_ms
            .store(ack.pause_ms, Ordering::Relaxed);
        state.metrics
            .weight_reload_barrier_ms
            .store(ack.barrier_ms, Ordering::Relaxed);
        state.metrics
            .weight_reload_transfer_ms
            .store(transfer_ms, Ordering::Relaxed);
        state.metrics
            .weight_reload_apply_ms
            .store(ack.apply_ms, Ordering::Relaxed);
        state.metrics
            .weight_reload_phase_samples
            .fetch_add(1, Ordering::Relaxed);
        json!({
            "route_ms": route_ms,
            "pause_ms": ack.pause_ms,
            "barrier_ms": ack.barrier_ms,
            "transfer_ms": transfer_ms,
            "apply_ms": ack.apply_ms,
            "worker_total_ms": ack.total_ms,
            "wire_ms": wire_ms,
        })
    } else {
        json!({"route_ms": route_ms, "wire_ms": wire_ms})
    };
    match outcome {
        Ok(_) => {
            state.mark_success(&worker);
            state.metrics.weight_reloads.fetch_add(1, Ordering::Relaxed);
            Json(json!({"ok": true, "phases": phases})).into_response()
        }
        Err(e) => {
            state.mark_failure(&worker, e.clone());
            state.metrics.errors.fetch_add(1, Ordering::Relaxed);
            bad_request(e)
        }
    }
}

/// Runtime worker registration. Authenticated: an untrusted client must not
/// be able to redirect rollout traffic to an arbitrary address.
pub async fn register_worker(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    let address = match payload.get("address").and_then(Value::as_str) {
        Some(addr) if !addr.trim().is_empty() => addr.trim().to_string(),
        _ => return bad_request("address is required"),
    };
    if !address.contains(':') {
        return bad_request("address must be host:port");
    }
    let added = state.register(address.clone());
    Json(json!({
        "ok": added,
        "address": address,
        "workers": state.worker_list().len(),
    }))
    .into_response()
}

pub async fn deregister_worker(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    let address = match payload.get("address").and_then(Value::as_str) {
        Some(addr) => addr.to_string(),
        None => return bad_request("address is required"),
    };
    let removed = state.deregister(&address);
    Json(json!({
        "ok": removed,
        "address": address,
        "workers": state.worker_list().len(),
    }))
    .into_response()
}

/// Send one lifecycle command to a worker and require a matching epoch/action
/// ack. Stale acks from a prior transition are rejected rather than accepted.
async fn send_lifecycle(
    addr: &str,
    command: &LifecycleCommand,
    limit: Duration,
    max_frame: usize,
) -> Result<LifecycleAck, String> {
    let frame = encode(Kind::LifecycleCommand, command)
        .map_err(|e| format!("encode lifecycle command: {e}"))?;
    let replies = forward(addr, frame, limit, max_frame).await?;
    for reply in replies {
        let (kind, value) = decode(&reply).map_err(|e| format!("lifecycle reply frame: {e}"))?;
        if kind != Kind::LifecycleAck {
            continue;
        }
        let ack: LifecycleAck = serde_json::from_value(value)
            .map_err(|e| format!("lifecycle ack schema: {e}"))?;
        if ack.matches(command) {
            return Ok(ack);
        }
    }
    Err(format!(
        "worker {addr} did not ack {} epoch {}",
        command.action.as_str(),
        command.epoch
    ))
}

/// Orchestrate a lifecycle transition across the whole worker pool. A
/// transition is serialized (no two pauses/resumes overlap), epoch-stamped,
/// and rolled back if any worker fails — so the pool is never left half
/// offloaded while generation is still being admitted.
async fn orchestrate_lifecycle(
    state: AppState,
    action: LifecycleAction,
    offload: bool,
    drain_ms: u64,
    timeout_ms: u64,
) -> Response {
    let _transition = state.lifecycle_transition.lock().await;
    let workers = state.worker_list();
    let started = std::time::Instant::now();

    // Stamp the transitional phase atomically, before any worker is touched.
    // Idempotent repeats return the current state without re-offloading. The
    // lock must be released before `lifecycle_json()` re-locks it.
    let already = {
        let mut lifecycle = state.lifecycle.lock().unwrap();
        let already = match action {
            LifecycleAction::Pause => lifecycle.paused,
            LifecycleAction::Resume => !lifecycle.paused,
        };
        if !already {
            lifecycle.epoch += 1;
            let epoch = lifecycle.epoch;
            let phase = match action {
                LifecycleAction::Pause => LifecyclePhase::Pausing,
                LifecycleAction::Resume => LifecyclePhase::Resuming,
            };
            for addr in &workers {
                let entry = lifecycle
                    .workers
                    .entry(addr.clone())
                    .or_insert_with(WorkerLifecycle::default);
                entry.phase = phase;
                entry.epoch = epoch;
                entry.error = None;
                entry.changed_at_ms = now_ms();
            }
        }
        already
    };
    if already {
        return Json(json!({
            "ok": true, "already": true, "state": state.lifecycle_json(),
        }))
        .into_response();
    }
    let epoch = state.lifecycle.lock().unwrap().epoch;
    let command = LifecycleCommand {
        request_id: epoch,
        epoch,
        action,
        offload,
        drain_ms,
    };
    let limit = Duration::from_millis(timeout_ms.max(1));
    let mut tasks = Vec::with_capacity(workers.len());
    for addr in &workers {
        let (addr, command, state) = (addr.clone(), command.clone(), state.clone());
        tasks.push(tokio::spawn(async move {
            let result =
                send_lifecycle(&addr, &command, limit, state.max_frame_bytes).await;
            (addr, result)
        }));
    }
    let mut outcomes: Vec<(String, Result<LifecycleAck, String>)> = Vec::new();
    for (addr, task) in workers.into_iter().zip(tasks) {
        match task.await {
            Ok((task_addr, result)) => outcomes.push((task_addr, result)),
            Err(e) => outcomes.push((addr, Err(format!("lifecycle task failed: {e}")))),
        }
    }
    let failures: Vec<(String, String)> = outcomes
        .iter()
        .filter_map(|(addr, result)| match result {
            Ok(ack) if ack.ok => None,
            Ok(ack) => Some((
                addr.clone(),
                ack.error
                    .clone()
                    .unwrap_or_else(|| format!("worker reported {} failed", action.as_str())),
            )),
            Err(error) => Some((addr.clone(), error.clone())),
        })
        .collect();

    let elapsed_ms = started.elapsed().as_millis() as u64;
    if !failures.is_empty() {
        // Compensating rollback: restore every worker to running, advance the
        // epoch so a late ack from the failed attempt can never be accepted.
        let rollback = LifecycleAction::Resume;
        let rollback_command = LifecycleCommand {
            request_id: epoch + 1,
            epoch: epoch + 1,
            action: rollback,
            offload: false,
            drain_ms: 0,
        };
        for addr in state.worker_list() {
            let (addr, rollback_command, state) =
                (addr, rollback_command.clone(), state.clone());
            tokio::spawn(async move {
                let _ = send_lifecycle(
                    &addr,
                    &rollback_command,
                    limit,
                    state.max_frame_bytes,
                )
                .await;
            });
        }
        {
            let mut lifecycle = state.lifecycle.lock().unwrap();
            lifecycle.epoch = epoch + 1;
            lifecycle.paused = false;
            lifecycle.offloaded = false;
            for worker in lifecycle.workers.values_mut() {
                worker.phase = LifecyclePhase::Running;
                worker.epoch = epoch + 1;
                worker.error = Some("lifecycle transition failed; rolled back".into());
                worker.changed_at_ms = now_ms();
            }
        }
        state.metrics.lifecycle_errors.fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "ok": false,
                "error": format!("{} failed on {} worker(s)", action.as_str(), failures.len()),
                "failures": failures,
                "state": state.lifecycle_json(),
            })),
        )
            .into_response();
    }

    let final_phase = match action {
        LifecycleAction::Pause => {
            if offload {
                LifecyclePhase::Offloaded
            } else {
                LifecyclePhase::Paused
            }
        }
        LifecycleAction::Resume => LifecyclePhase::Running,
    };
    let drained: u64 = outcomes
        .iter()
        .filter_map(|(_, result)| result.as_ref().ok())
        .map(|ack| ack.drained as u64)
        .sum();
    {
        let mut lifecycle = state.lifecycle.lock().unwrap();
        for (addr, _) in &outcomes {
            if let Some(worker) = lifecycle.workers.get_mut(addr) {
                worker.phase = final_phase;
                worker.error = None;
                worker.changed_at_ms = now_ms();
            }
        }
        lifecycle.paused = action == LifecycleAction::Pause;
        lifecycle.offloaded = action == LifecycleAction::Pause && offload;
    }
    let counter = match action {
        LifecycleAction::Pause => &state.metrics.lifecycle_pauses,
        LifecycleAction::Resume => &state.metrics.lifecycle_resumes,
    };
    counter.fetch_add(1, Ordering::Relaxed);
    let timer = match action {
        LifecycleAction::Pause => &state.metrics.lifecycle_pause_ms,
        LifecycleAction::Resume => &state.metrics.lifecycle_resume_ms,
    };
    timer.store(elapsed_ms, Ordering::Relaxed);
    Json(json!({
        "ok": true,
        "epoch": epoch,
        "action": action.as_str(),
        "drained": drained,
        "elapsed_ms": elapsed_ms,
        "state": state.lifecycle_json(),
    }))
    .into_response()
}

/// POST /rollout/pause — fence new generation, drain in-flight work and
/// optionally release GPU memory before the trainer takes the device.
pub async fn pause_rollout(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    let offload = payload.get("offload").and_then(Value::as_bool).unwrap_or(true);
    let drain_ms = payload
        .get("drain_ms")
        .and_then(Value::as_u64)
        .unwrap_or(30_000);
    let timeout_ms = payload
        .get("timeout_ms")
        .and_then(Value::as_u64)
        .unwrap_or(state.worker_timeout.as_millis() as u64);
    orchestrate_lifecycle(state, LifecycleAction::Pause, offload, drain_ms, timeout_ms).await
}

/// POST /rollout/resume — restore GPU memory and continue generation.
pub async fn resume_rollout(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    let timeout_ms = payload
        .get("timeout_ms")
        .and_then(Value::as_u64)
        .unwrap_or(state.worker_timeout.as_millis() as u64);
    orchestrate_lifecycle(state, LifecycleAction::Resume, false, 0, timeout_ms).await
}

/// GET /rollout/lifecycle — live lifecycle state (public, like /health).
pub async fn rollout_lifecycle(State(state): State<AppState>) -> Response {
    Json(state.lifecycle_json()).into_response()
}

/// Map a registry error to an HTTP status + message.
fn registry_error(e: RegistryError) -> Response {
    use RegistryError::*;
    let status = match e {
        StaleVersion { .. } | DuplicateVersion(_) | NotStaged(_) => StatusCode::CONFLICT,
        UnknownModel | UnknownVersion(_) => StatusCode::NOT_FOUND,
        BadDigest | NoWorkers => StatusCode::BAD_REQUEST,
    };
    (status, Json(json!({"error": e.to_string()}))).into_response()
}

/// GET /models — list every model with its version history and active version.
pub async fn list_models(State(state): State<AppState>) -> Response {
    Json(state.registry.list()).into_response()
}

/// GET /models/:name — snapshot one model.
pub async fn get_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Response {
    match state.registry.snapshot(&name) {
        Some(snapshot) => Json(snapshot).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "unknown model"})),
        )
            .into_response(),
    }
}

/// POST /models/:name/versions — publish a staged version. Body:
/// `{ "version": u64, "digest": "<sha256 hex>", "workers": ["addr", ...] }`.
pub async fn publish_version(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(payload): Json<Value>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    let version = match payload.get("version").and_then(Value::as_u64) {
        Some(v) => v,
        None => return bad_request("version is required"),
    };
    let digest = match payload.get("digest").and_then(Value::as_str) {
        Some(d) => d,
        None => return bad_request("digest is required"),
    };
    let workers: Vec<String> = match payload.get("workers").and_then(Value::as_array) {
        Some(arr) => arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        None => return bad_request("workers is required"),
    };
    match state.registry.publish(&name, version, digest, workers) {
        Ok(()) => {
            state.metrics.model_publishes.fetch_add(1, Ordering::Relaxed);
            Json(json!({"ok": true, "model": name, "version": version, "state": "staged"}))
                .into_response()
        }
        Err(e) => {
            state.metrics.model_rejected.fetch_add(1, Ordering::Relaxed);
            registry_error(e)
        }
    }
}

/// POST /models/:name/versions/:version/commit — atomically promote a staged
/// version to active; the previous active version is retired (drains).
pub async fn commit_version(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((name, version)): Path<(String, u64)>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    match state.registry.commit(&name, version) {
        Ok(result) => {
            state.metrics.model_commits.fetch_add(1, Ordering::Relaxed);
            Json(json!({
                "ok": true,
                "model": name,
                "committed": result.committed,
                "retired": result.retired,
            }))
            .into_response()
        }
        Err(e) => {
            state.metrics.model_rejected.fetch_add(1, Ordering::Relaxed);
            registry_error(e)
        }
    }
}

/// POST /models/:name/versions/:version/abort — discard a failed staged
/// publish without touching the active version.
pub async fn abort_version(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((name, version)): Path<(String, u64)>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    match state.registry.abort(&name, version) {
        Ok(()) => {
            state.metrics.model_aborts.fetch_add(1, Ordering::Relaxed);
            Json(json!({"ok": true, "model": name, "version": version, "state": "aborted"}))
                .into_response()
        }
        Err(e) => {
            state.metrics.model_rejected.fetch_add(1, Ordering::Relaxed);
            registry_error(e)
        }
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(prometheus_metrics))
        .route("/rollout/generate", post(generate))
        .route("/rollout/weights", post(update_weights))
        .route("/rollout/pause", post(pause_rollout))
        .route("/rollout/resume", post(resume_rollout))
        .route("/rollout/lifecycle", get(rollout_lifecycle))
        .route("/workers/register", post(register_worker))
        .route("/workers/deregister", post(deregister_worker))
        .route("/models", get(list_models))
        .route("/models/{name}", get(get_model))
        .route("/models/{name}/versions", post(publish_version))
        .route("/models/{name}/versions/{version}/commit", post(commit_version))
        .route("/models/{name}/versions/{version}/abort", post(abort_version))
        .with_state(state)
}

pub async fn serve(
    listener: TcpListener,
    workers: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState::new(workers);
    tokio::spawn(monitor(state.clone()));
    axum::serve(listener, router(state)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use megarl_protocol::TrajectoryRecord;
    use tower::ServiceExt;

    async fn fake_worker() -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4];
            socket.read_exact(&mut buf).await.unwrap();
            let len = u32::from_be_bytes(buf) as usize;
            let mut frame = vec![0u8; len];
            socket.read_exact(&mut frame).await.unwrap();
            let (kind, _) = decode(&frame).unwrap();
            assert_eq!(kind, Kind::RolloutRequest);

            let reply = TrajectoryRecord {
                request_id: 1,
                prompt_len: 2,
                token_ids: vec![7],
                logprobs: vec![-0.25],
                top_logprobs: vec![],
                policy_versions: vec![3],
                reward: None,
                finish_reason: 0,
                payload_handle: None,
                sampling_logprobs: vec![],
            };
            let frame = encode(Kind::TrajectoryRecord, &reply).unwrap();
            socket.write_all(&(frame.len() as u32).to_be_bytes()).await.unwrap();
            socket.write_all(&frame).await.unwrap();
            socket.write_all(&0u32.to_be_bytes()).await.unwrap();
        });
        (addr, handle)
    }

    /// Worker that answers a weight update with a phase-timing ack.
    async fn weight_ack_worker() -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4];
            socket.read_exact(&mut buf).await.unwrap();
            let len = u32::from_be_bytes(buf) as usize;
            let mut frame = vec![0u8; len];
            socket.read_exact(&mut frame).await.unwrap();
            assert_eq!(decode(&frame).unwrap().0, Kind::WeightUpdate);
            tokio::time::sleep(Duration::from_millis(5)).await;
            let ack = WeightUpdateAck {
                request_id: 7,
                ok: true,
                pause_ms: 1,
                barrier_ms: 2,
                transfer_ms: 3,
                apply_ms: 4,
                total_ms: 10,
                error: None,
            };
            let reply = encode(Kind::WeightUpdateAck, &ack).unwrap();
            socket.write_all(&(reply.len() as u32).to_be_bytes()).await.unwrap();
            socket.write_all(&reply).await.unwrap();
            socket.write_all(&0u32.to_be_bytes()).await.unwrap();
        });
        (addr, handle)
    }

    #[tokio::test]
    async fn weight_update_records_phase_breakdown_from_worker_ack() {
        let (addr, worker) = weight_ack_worker().await;
        let state = AppState::new(vec![addr]);
        let response = update_weights(
            State(state.clone()),
            HeaderMap::new(),
            Json(json!({"path": "/tmp/weights.mega"})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["ok"], true);
        assert_eq!(body["phases"]["pause_ms"], 1);
        assert_eq!(body["phases"]["barrier_ms"], 2);
        assert_eq!(body["phases"]["apply_ms"], 4);
        // Gateway transfer is wire time minus the worker-reported barrier and
        // apply time (floored at zero); it must not double-count either phase.
        let wire = body["phases"]["wire_ms"].as_u64().unwrap();
        let transfer = body["phases"]["transfer_ms"].as_u64().unwrap();
        assert_eq!(transfer, wire.saturating_sub(4 + 2));
        let snapshot = metrics_snapshot(&state);
        assert_eq!(snapshot["weight_reload_apply_ms"], 4);
        assert_eq!(snapshot["weight_reload_pause_ms"], 1);
        assert_eq!(snapshot["weight_reload_phase_samples"], 1);
        worker.abort();
    }

    #[test]
    fn worker_selection_is_sticky_by_session_and_round_robin_otherwise() {
        let state = AppState::new(vec!["a".into(), "b".into(), "c".into()]);
        // 5 % 3 == 2 -> worker "c", stable across calls.
        assert_eq!(state.worker_for(Some(5)), "c");
        assert_eq!(state.worker_for(Some(5)), "c");
        assert_eq!(state.worker_for(Some(7)), "b");
        let first = state.worker_for(None).to_string();
        let second = state.worker_for(None).to_string();
        assert_ne!(first, second);
    }

    #[test]
    fn sticky_sessions_are_rebalanced_when_usage_skews() {
        let mut state = AppState::new(vec!["a".into(), "b".into()]);
        state.skew_ratio = 1.5;
        state.skew_min_gap = 2;
        // Prime both workers so the least-used count is non-zero.
        assert_eq!(state.worker_for(None), "a");
        assert_eq!(state.worker_for(None), "b");
        // Session 0 pins to "a"; it stays there until the skew threshold is hit.
        assert_eq!(state.worker_for(Some(0)), "a");
        assert_eq!(state.worker_for(Some(0)), "a");
        assert_eq!(state.worker_for(Some(0)), "b"); // rebalanced
        assert_eq!(state.metrics.rebalances.load(Ordering::Relaxed), 1);
        let usage = state.usage_json();
        assert_eq!(usage["workers"][0]["assigned"], 3);
        assert_eq!(usage["workers"][1]["assigned"], 2);
        assert_eq!(usage["rebalances"], 1);
    }

    #[test]
    fn concurrent_routing_and_observability_do_not_deadlock() {
        let state = AppState::new(vec!["a".into(), "b".into(), "c".into()]);
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let s = state.clone();
                std::thread::spawn(move || {
                    for j in 0..200 {
                        match j % 4 {
                            0 => { s.worker_for(Some(i)); }
                            1 => { s.health_json(); }
                            2 => { s.usage_json(); }
                            _ => { s.fallback_for("a"); }
                        }
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[tokio::test]
    async fn runtime_registry_adds_and_removes_workers_with_auth() {
        let state = AppState::with_token(vec!["a:1".into()], Some("t".into()));
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer t".parse().unwrap());

        let denied = register_worker(
            State(state.clone()),
            HeaderMap::new(),
            Json(json!({"address": "b:2"})),
        )
        .await;
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(state.worker_list().len(), 1);

        let added = register_worker(
            State(state.clone()),
            headers.clone(),
            Json(json!({"address": "b:2"})),
        )
        .await;
        assert_eq!(added.status(), StatusCode::OK);
        assert_eq!(state.worker_list(), vec!["a:1", "b:2"]);
        assert_eq!(state.health_json()["total"], 2);

        let duplicate = register_worker(
            State(state.clone()),
            headers.clone(),
            Json(json!({"address": "b:2"})),
        )
        .await;
        assert_eq!(duplicate.status(), StatusCode::OK);
        assert_eq!(state.worker_list().len(), 2);

        let removed = deregister_worker(
            State(state.clone()),
            headers.clone(),
            Json(json!({"address": "a:1"})),
        )
        .await;
        assert_eq!(removed.status(), StatusCode::OK);
        assert_eq!(state.worker_list(), vec!["b:2"]);
        assert_eq!(state.health_json()["total"], 1);

        // The last worker cannot be removed, so routing never panics.
        let last = deregister_worker(
            State(state.clone()),
            headers,
            Json(json!({"address": "b:2"})),
        )
        .await;
        assert_eq!(last.status(), StatusCode::OK);
        assert_eq!(state.worker_list().len(), 1);
    }

    #[tokio::test]
    async fn health_probe_requires_a_matching_nonce_echo() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let responder = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4];
            socket.read_exact(&mut buf).await.unwrap();
            let mut frame = vec![0u8; u32::from_be_bytes(buf) as usize];
            socket.read_exact(&mut frame).await.unwrap();
            let (kind, value) = decode(&frame).unwrap();
            assert_eq!(kind, Kind::HealthCheck);
            let reply = encode(Kind::HealthCheck, &json!({"probe": value["probe"], "ok": true})).unwrap();
            socket.write_all(&(reply.len() as u32).to_be_bytes()).await.unwrap();
            socket.write_all(&reply).await.unwrap();
        });
        probe_worker(&addr, 99, Duration::from_secs(2), 1 << 20)
            .await
            .unwrap();
        // A wrong nonce must be rejected even if the worker answers.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let stale_addr = listener.local_addr().unwrap().to_string();
        let stale = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4];
            socket.read_exact(&mut buf).await.unwrap();
            let mut frame = vec![0u8; u32::from_be_bytes(buf) as usize];
            socket.read_exact(&mut frame).await.unwrap();
            let reply = encode(Kind::HealthCheck, &json!({"probe": 1, "ok": true})).unwrap();
            socket.write_all(&(reply.len() as u32).to_be_bytes()).await.unwrap();
            socket.write_all(&reply).await.unwrap();
        });
        assert!(probe_worker(&stale_addr, 2, Duration::from_secs(2), 1 << 20)
            .await
            .is_err());
        responder.await.unwrap();
        stale.await.unwrap();
    }

    #[tokio::test]
    async fn unhealthy_workers_are_excluded_and_health_endpoint_reports_them() {
        let state = AppState::new(vec!["a".into(), "b".into()]);
        assert_eq!(state.worker_for(Some(0)), "a");
        state.mark_failure("a", "probe timeout");
        assert_eq!(state.worker_for(Some(0)), "b");
        assert_eq!(state.fallback_for("b"), None); // only healthy "b" remains
        let response = health(State(state.clone())).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["ok"], true);
        assert_eq!(value["healthy"], 1);
        assert_eq!(value["workers"][0]["last_error"], "probe timeout");
        // A later successful probe restores the worker to the pool.
        state.mark_success("a");
        assert_eq!(state.worker_for(Some(0)), "a");
    }

    #[tokio::test]
    async fn bearer_auth_rejects_missing_and_wrong_tokens() {
        let state = AppState::with_token(
            vec!["127.0.0.1:1".into()],
            Some("secret".into()),
        );
        let payload = json!({
            "request_id": 1, "policy_version": 3, "prompt_token_ids": [1, 2],
            "max_new_tokens": 8, "n": 1, "session_id": null,
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                         "seed": null, "logprobs": true, "top_logprobs": 0}
        });
        let denied = generate(State(state.clone()), HeaderMap::new(), Json(payload.clone())).await;
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer wrong".parse().unwrap());
        let denied = generate(State(state.clone()), headers, Json(payload.clone())).await;
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(state.metrics.auth_rejected.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn edge_rate_limiter_shapes_admission() {
        let state = AppState::new(vec!["a".into()]);
        assert!(state.admit().is_ok()); // disabled by default
        *state.rate_limiter.lock().unwrap() = Some(TokenBucketRateLimiter::new(0.1, 1));
        assert!(state.admit().is_ok()); // burst token
        let retry = state.admit().unwrap_err();
        assert!((9_000..=10_000).contains(&retry));
    }

    #[tokio::test]
    async fn generate_rejects_rate_limited_requests_with_retry_after() {
        let state = AppState::new(vec!["127.0.0.1:1".into()]);
        *state.rate_limiter.lock().unwrap() = Some(TokenBucketRateLimiter::new(0.1, 1));
        let payload = json!({
            "request_id": 1, "policy_version": 3, "prompt_token_ids": [1, 2],
            "max_new_tokens": 8, "n": 1, "session_id": null,
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                         "seed": null, "logprobs": true, "top_logprobs": 0}
        });
        // First request consumes the burst token (the dead worker returns 400).
        let first = generate(State(state.clone()), HeaderMap::new(), Json(payload.clone())).await;
        assert_ne!(first.status(), StatusCode::TOO_MANY_REQUESTS);
        let second = generate(State(state.clone()), HeaderMap::new(), Json(payload)).await;
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(second.headers().contains_key(header::RETRY_AFTER));
        assert_eq!(state.metrics.rate_rejected.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn generate_forwards_frames_and_exposes_logprobs() {
        let (worker_addr, worker) = fake_worker().await;
        let state = AppState::new(vec![worker_addr]);
        let payload = json!({
            "request_id": 1, "policy_version": 3, "prompt_token_ids": [1, 2],
            "max_new_tokens": 8, "n": 1, "session_id": null,
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                         "seed": null, "logprobs": true, "top_logprobs": 0}
        });
        let response = generate(State(state.clone()), HeaderMap::new(), Json(payload)).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(state.metrics.generates.load(Ordering::Relaxed), 1);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn rejects_bad_schema_without_touching_worker() {
        let state = AppState::new(vec!["127.0.0.1:1".to_string()]);
        let response = generate(
            State(state.clone()),
            HeaderMap::new(),
            Json(json!({"request_id": 1, "unexpected": true})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(state.metrics.errors.load(Ordering::Relaxed), 1);
    }

    async fn body_json(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn digest(n: u8) -> String {
        format!("{:02x}", n).repeat(32)
    }

    #[tokio::test]
    async fn model_registry_publish_commit_and_abort_over_http() {
        let state = AppState::new(vec!["w1".into(), "w2".into()]);
        let publish = |v: u64, workers: Vec<&str>| {
            Json(json!({"version": v, "digest": digest(v as u8), "workers": workers}))
        };
        // Publish v1 and commit it.
        let r = publish_version(
            State(state.clone()),
            HeaderMap::new(),
            Path("policy".to_string()),
            publish(1, vec!["w1", "w2"]),
        )
        .await;
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(body_json(r).await["state"], "staged");
        let r = commit_version(
            State(state.clone()),
            HeaderMap::new(),
            Path(("policy".to_string(), 1)),
        )
        .await;
        assert_eq!(body_json(r).await["retired"], Value::Null);

        // Publish v2 and abort it: the active version must stay at v1.
        let r = publish_version(
            State(state.clone()),
            HeaderMap::new(),
            Path("policy".to_string()),
            publish(2, vec!["w2"]),
        )
        .await;
        assert_eq!(r.status(), StatusCode::OK);
        let r = abort_version(
            State(state.clone()),
            HeaderMap::new(),
            Path(("policy".to_string(), 2)),
        )
        .await;
        assert_eq!(body_json(r).await["state"], "aborted");
        assert_eq!(state.registry.active_version("policy"), Some(1));

        // The model registry reports the full lifecycle; /metrics only exposes numeric counters.
        let snap = body_json(get_model(State(state.clone()), Path("policy".to_string())).await).await;
        assert_eq!(snap["active"], 1);
        let states: Vec<_> = snap["versions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["state"].as_str().unwrap())
            .collect();
        assert_eq!(states, vec!["committed", "aborted"]);
        let metrics = metrics_snapshot(&state);
        assert_eq!(metrics["model_publishes"], 2);
        assert_eq!(metrics["model_commits"], 1);
        assert_eq!(metrics["model_aborts"], 1);
        assert!(metrics["models"].as_array().unwrap().len() >= 1);
    }

    #[tokio::test]
    async fn model_registry_endpoints_require_auth_and_reject_bad_input() {
        // Mutating endpoints are authenticated when a token is configured.
        let secure = AppState::with_token(vec!["w1".into()], Some("s3cret".into()));
        let r = publish_version(
            State(secure.clone()),
            HeaderMap::new(),
            Path("p".to_string()),
            Json(json!({"version": 1, "digest": digest(1), "workers": ["w1"]})),
        )
        .await;
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED);
        let r = commit_version(State(secure), HeaderMap::new(), Path(("p".to_string(), 1))).await;
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED);

        // Validation errors map to meaningful statuses.
        let state = AppState::new(vec!["w1".into(), "w2".into()]);
        let r = publish_version(
            State(state.clone()),
            HeaderMap::new(),
            Path("p".to_string()),
            Json(json!({"version": 1, "digest": "bad", "workers": ["w1"]})),
        )
        .await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        let r = publish_version(
            State(state.clone()),
            HeaderMap::new(),
            Path("p".to_string()),
            Json(json!({"version": 5, "digest": digest(1), "workers": ["w1"]})),
        )
        .await;
        assert_eq!(r.status(), StatusCode::OK);
        let r = publish_version(
            State(state.clone()),
            HeaderMap::new(),
            Path("p".to_string()),
            Json(json!({"version": 4, "digest": digest(2), "workers": ["w2"]})),
        )
        .await;
        assert_eq!(r.status(), StatusCode::CONFLICT); // stale version
    }

    #[test]
    fn model_aware_routing_constrains_pool_to_active_version_workers() {
        let state = AppState::new(vec!["w1".into(), "w2".into()]);
        // Without a committed version, session 0 pins to w1 (0 % 2).
        assert_eq!(state.worker_for(Some(0)), "w1");
        state
            .registry
            .publish("p", 1, &digest(1), vec!["w2".into()])
            .unwrap();
        state.registry.commit("p", 1).unwrap();
        // With an active version served only by w2, the same session routes there.
        assert_eq!(state.worker_for_model(Some(0), "p"), "w2");
        // Unknown models keep the full pool.
        assert_eq!(state.worker_for_model(Some(0), "missing"), "w1");
    }

    #[test]
    fn prometheus_renderer_renders_small_snapshot_with_megarl_prefix() {
        let rendered = render_prometheus_metrics(r#"{"generates":3}"#, "megarl").unwrap();
        assert!(rendered.contains("megarl_generates 3\n"), "{rendered}");
        assert!(rendered.contains("# TYPE megarl_generates counter\n"), "{rendered}");
    }

    #[tokio::test]
    async fn metrics_route_serves_prometheus_text() {
        let state = AppState::new(vec!["a".into()]);
        state.metrics.generates.store(3, Ordering::Relaxed);
        let app = router(state);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/plain; version=0.0.4; charset=utf-8"
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("megarl_generates 3\n"), "{text}");
        assert!(text.contains("# TYPE"), "{text}");
    }

    /// Fake worker that answers lifecycle commands. `fail_pause` makes a pause
    /// ack report failure so the gateway's compensating rollback can be tested.
    async fn lifecycle_worker(fail_pause: bool) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                loop {
                    let mut buf = [0u8; 4];
                    if socket.read_exact(&mut buf).await.is_err() {
                        break;
                    }
                    let len = u32::from_be_bytes(buf) as usize;
                    if len == 0 {
                        break;
                    }
                    let mut frame = vec![0u8; len];
                    if socket.read_exact(&mut frame).await.is_err() {
                        break;
                    }
                    let Ok((kind, value)) = decode(&frame) else {
                        break;
                    };
                    if kind != Kind::LifecycleCommand {
                        continue;
                    }
                    let command: LifecycleCommand = serde_json::from_value(value).unwrap();
                    let ok = !(fail_pause && command.action == LifecycleAction::Pause);
                    let phase = match command.action {
                        LifecycleAction::Pause => LifecyclePhase::Offloaded,
                        LifecycleAction::Resume => LifecyclePhase::Running,
                    };
                    let error = if ok {
                        None
                    } else {
                        Some("worker refused to offload".to_string())
                    };
                    let ack = LifecycleAck {
                        request_id: command.request_id,
                        epoch: command.epoch,
                        action: command.action,
                        ok,
                        phase,
                        drained: 1,
                        error,
                        elapsed_ms: 1,
                    };
                    let reply = encode(Kind::LifecycleAck, &ack).unwrap();
                    socket.write_all(&(reply.len() as u32).to_be_bytes()).await.unwrap();
                    socket.write_all(&reply).await.unwrap();
                    socket.write_all(&0u32.to_be_bytes()).await.unwrap();
                }
            }
        });
        (addr, handle)
    }

    #[tokio::test]
    async fn pause_fences_routing_and_resume_restores_it() {
        let (addr, worker) = lifecycle_worker(false).await;
        let state = AppState::new(vec![addr.clone()]);
        assert_eq!(state.try_route(Some(0), None), Some(addr.clone()));

        let response = pause_rollout(
            State(state.clone()),
            HeaderMap::new(),
            Json(json!({"offload": true, "drain_ms": 1_000, "timeout_ms": 5_000})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["ok"], true);
        assert_eq!(body["state"]["paused"], true);
        assert_eq!(body["state"]["offloaded"], true);
        assert_eq!(body["drained"], 1);
        assert_eq!(state.try_route(Some(0), None), None);
        assert!(state.all_paused());

        // A repeated pause is idempotent, not a second offload.
        let response = pause_rollout(
            State(state.clone()),
            HeaderMap::new(),
            Json(json!({"offload": true})),
        )
        .await;
        assert_eq!(body_json(response).await["already"], true);
        assert_eq!(state.metrics.lifecycle_pauses.load(Ordering::Relaxed), 1);

        let response =
            resume_rollout(State(state.clone()), HeaderMap::new(), Json(json!({}))).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(state.try_route(Some(0), None), Some(addr));
        assert!(!state.all_paused());
        assert_eq!(state.metrics.lifecycle_resumes.load(Ordering::Relaxed), 1);
        worker.abort();
    }

    #[tokio::test]
    async fn failed_pause_rolls_back_to_a_routable_pool() {
        let (addr, worker) = lifecycle_worker(true).await;
        let state = AppState::new(vec![addr.clone()]);
        let response = pause_rollout(
            State(state.clone()),
            HeaderMap::new(),
            Json(json!({"offload": true, "drain_ms": 1_000, "timeout_ms": 5_000})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = body_json(response).await;
        assert_eq!(body["ok"], false);
        assert_eq!(body["state"]["paused"], false);
        assert_eq!(state.try_route(Some(0), None), Some(addr));
        assert_eq!(state.metrics.lifecycle_errors.load(Ordering::Relaxed), 1);
        worker.abort();
    }

    #[tokio::test]
    async fn lifecycle_endpoints_require_auth() {
        let state = AppState::with_token(vec!["w1".into()], Some("secret".into()));
        let pause = pause_rollout(
            State(state.clone()),
            HeaderMap::new(),
            Json(json!({"offload": true})),
        )
        .await;
        assert_eq!(pause.status(), StatusCode::UNAUTHORIZED);
        let resume =
            resume_rollout(State(state), HeaderMap::new(), Json(json!({}))).await;
        assert_eq!(resume.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn registering_a_worker_while_paused_inherits_the_fenced_phase() {
        let state = AppState::new(vec!["w1".into()]);
        {
            let mut lifecycle = state.lifecycle.lock().unwrap();
            lifecycle.paused = true;
            lifecycle.offloaded = true;
            lifecycle.epoch = 7;
            if let Some(worker) = lifecycle.workers.get_mut("w1") {
                worker.phase = LifecyclePhase::Offloaded;
                worker.epoch = 7;
            }
        }
        assert!(state.register("w2".into()));
        let snapshot = state.lifecycle_json();
        let joined = snapshot["workers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|w| w["address"] == "w2")
            .unwrap();
        assert_eq!(joined["phase"], "offloaded");
        assert_eq!(joined["epoch"], 7);
        assert_eq!(state.try_route(Some(0), None), None);
    }
}
