//! HTTP front end for the Rust reward pool.
//!
//! Verifiers are CPU/process heavy and should not sit inside the rollout
//! gateway or the training process. This service owns the bounded retrying
//! pool and exposes it behind a version-neutral JSON API with bearer auth,
//! health and metrics.

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use megarl_reward::seccomp::{find_firejail, sandbox_label};
use megarl_reward::{grade_benchmark_json, verify_batch_json, SandboxConfig};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub verified: AtomicU64,
    pub graded: AtomicU64,
    pub errors: AtomicU64,
    pub auth_rejected: AtomicU64,
}

#[derive(Clone, Debug)]
pub struct RewardState {
    pub token: Option<String>,
    pub workers: usize,
    pub max_retries: u32,
    pub max_batch: usize,
    pub metrics: Arc<Metrics>,
}

impl Default for RewardState {
    fn default() -> Self {
        Self::new(None, 4, 1)
    }
}

impl RewardState {
    pub fn new(token: Option<String>, workers: usize, max_retries: u32) -> Self {
        assert!(workers >= 1, "workers must be positive");
        Self {
            token,
            workers,
            max_retries,
            max_batch: std::env::var("MEGARL_REWARD_MAX_BATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096),
            metrics: Arc::new(Metrics::default()),
        }
    }

    fn authorized(&self, headers: &HeaderMap) -> bool {
        match &self.token {
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

pub async fn health() -> Response {
    Json(json!({"ok": true, "service": "megarl-reward"})).into_response()
}

/// Active syscall-level sandbox profile, for operator diagnostics.
pub fn sandbox_profile_json() -> Value {
    let defaults = SandboxConfig::default();
    let available = find_firejail(None).is_some();
    json!({
        "seccomp": defaults.seccomp.as_str(),
        "firejail": defaults.firejail.as_str(),
        "firejail_available": available,
        "effective_label": sandbox_label(available, defaults.seccomp),
    })
}

pub async fn sandbox_profile() -> Response {
    Json(sandbox_profile_json()).into_response()
}

pub async fn metrics(State(state): State<RewardState>) -> Response {
    Json(json!({
        "sandbox": sandbox_profile_json(),
        "requests": state.metrics.requests.load(Ordering::Relaxed),
        "verified": state.metrics.verified.load(Ordering::Relaxed),
        "graded": state.metrics.graded.load(Ordering::Relaxed),
        "errors": state.metrics.errors.load(Ordering::Relaxed),
        "auth_rejected": state.metrics.auth_rejected.load(Ordering::Relaxed),
        "workers": state.workers,
        "max_retries": state.max_retries,
    }))
    .into_response()
}

/// `POST /verify` accepts either `{"requests":[...], "workers":N,
/// "max_retries":N}` or a bare array of verify requests.
pub async fn verify(
    State(state): State<RewardState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    let requests = if payload.is_array() {
        payload.clone()
    } else {
        match payload.get("requests") {
            Some(value) if value.is_array() => value.clone(),
            _ => return bad_request("requests must be an array"),
        }
    };
    let count = requests.as_array().map(Vec::len).unwrap_or(0);
    if count == 0 || count > state.max_batch {
        return bad_request(format!("batch size must be 1..={}", state.max_batch));
    }
    let workers = payload
        .get("workers")
        .and_then(Value::as_u64)
        .unwrap_or(state.workers as u64)
        .clamp(1, 64) as usize;
    let retries = payload
        .get("max_retries")
        .and_then(Value::as_u64)
        .unwrap_or(state.max_retries as u64)
        .min(10) as u32;

    let request_json = requests.to_string();
    let (outcomes, stats) = match verify_batch_json(&request_json, workers, retries) {
        Ok(result) => result,
        Err(e) => {
            state.metrics.errors.fetch_add(1, Ordering::Relaxed);
            return bad_request(format!("invalid request: {e}"));
        }
    };
    let outcomes: Value = match serde_json::from_str(&outcomes) {
        Ok(value) => value,
        Err(e) => {
            state.metrics.errors.fetch_add(1, Ordering::Relaxed);
            return bad_request(format!("outcome serialization failed: {e}"));
        }
    };
    let stats: Value = serde_json::from_str(&stats).unwrap_or_else(|_| json!({}));
    state.metrics.verified.fetch_add(count as u64, Ordering::Relaxed);
    Json(json!({"outcomes": outcomes, "stats": stats})).into_response()
}

/// `POST /bench/verify` grades one candidate against a benchmark spec. The
/// subprocess-backed grader runs in `spawn_blocking` so it never stalls the
/// async service. Body: `{"spec": {...}, "candidate": "..."}`.
pub async fn bench_verify(
    State(state): State<RewardState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    if !state.authorized(&headers) {
        state.metrics.auth_rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    let spec = match payload.get("spec") {
        Some(value) => value.to_string(),
        None => return bad_request("spec is required"),
    };
    let candidate = match payload.get("candidate").and_then(Value::as_str) {
        Some(text) => text.to_string(),
        None => return bad_request("candidate is required"),
    };
    let graded = tokio::task::spawn_blocking(move || grade_benchmark_json(&spec, &candidate)).await;
    let result = match graded {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            state.metrics.errors.fetch_add(1, Ordering::Relaxed);
            return bad_request(format!("invalid spec: {e}"));
        }
        Err(e) => {
            state.metrics.errors.fetch_add(1, Ordering::Relaxed);
            return bad_request(format!("grader task failed: {e}"));
        }
    };
    let result: Value = match serde_json::from_str(&result) {
        Ok(value) => value,
        Err(e) => {
            state.metrics.errors.fetch_add(1, Ordering::Relaxed);
            return bad_request(format!("grader serialization failed: {e}"));
        }
    };
    state.metrics.graded.fetch_add(1, Ordering::Relaxed);
    Json(json!({"result": result})).into_response()
}

pub fn router(state: RewardState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/sandbox/profile", get(sandbox_profile))
        .route("/metrics", get(metrics))
        .route("/verify", post(verify))
        .route("/bench/verify", post(bench_verify))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn body(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn verifies_a_batch_and_counts_metrics() {
        let state = RewardState::default();
        let payload = json!({"requests": [
            {"request_id": 1, "domain": "math", "response": "0.5", "expected": "1/2"},
            {"request_id": 2, "domain": "generic", "response": "yes", "expected": "yes"}
        ]});
        let response = verify(State(state.clone()), HeaderMap::new(), Json(payload)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let value = body(response).await;
        assert_eq!(value["outcomes"][0]["reward"], 1.0);
        assert_eq!(value["outcomes"][1]["reward"], 1.0);
        assert_eq!(state.metrics.verified.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn rejects_missing_token_and_bad_batches() {
        let state = RewardState::new(Some("secret".into()), 2, 1);
        let payload = json!([{"request_id": 1, "domain": "generic", "response": "a", "expected": "a"}]);
        let denied = verify(State(state.clone()), HeaderMap::new(), Json(payload.clone())).await;
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(state.metrics.auth_rejected.load(Ordering::Relaxed), 1);

        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer secret".parse().unwrap());
        let ok = verify(State(state.clone()), headers.clone(), Json(payload)).await;
        assert_eq!(ok.status(), StatusCode::OK);

        let empty = verify(State(state), headers, Json(json!([]))).await;
        assert_eq!(empty.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn health_and_metrics_are_public() {
        let state = RewardState::default();
        assert_eq!(body(health().await).await["ok"], true);
        let value = body(metrics(State(state)).await).await;
        assert_eq!(value["workers"], 4);
        assert_eq!(value["graded"], 0);
        assert!(value["sandbox"]["effective_label"].is_string());
    }

    #[tokio::test]
    async fn sandbox_profile_reports_the_active_layer() {
        let value = body(sandbox_profile().await).await;
        assert!(value["seccomp"].is_string());
        assert!(value["firejail"].is_string());
        assert!(value["firejail_available"].is_boolean());
        assert!(value["effective_label"].is_string());
    }

    #[tokio::test]
    async fn bench_verifier_reports_the_applied_sandbox_layer() {
        let state = RewardState::default();
        let spec = json!({
            "benchmark": "human_eval",
            "entry_point": "add",
            "tests": [{"args": [1, 2], "expected": "3"}],
            "seccomp": "blacklist",
            "firejail": "off"
        });
        let payload = json!({"spec": spec, "candidate": "def add(a, b): return a + b"});
        let response = bench_verify(State(state), HeaderMap::new(), Json(payload)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let value = body(response).await;
        assert_eq!(value["result"]["reward"], 1.0);
        assert_eq!(value["result"]["sandbox"], "seccomp_blacklist");
    }

    #[tokio::test]
    async fn bench_endpoint_grades_a_candidate_and_counts_metrics() {
        let state = RewardState::default();
        let spec = json!({
            "benchmark": "human_eval",
            "entry_point": "add",
            "tests": [
                {"args": [1, 2], "expected": "3"},
                {"args": [10, 5], "expected": "15"}
            ]
        });
        let payload = json!({"spec": spec, "candidate": "def add(a, b):\n    return a + b\n"});
        let response = bench_verify(State(state.clone()), HeaderMap::new(), Json(payload)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let value = body(response).await;
        assert_eq!(value["result"]["reward"], 1.0);
        assert_eq!(value["result"]["benchmark"], "human_eval");
        assert_eq!(state.metrics.graded.load(Ordering::Relaxed), 1);
        assert_eq!(body(metrics(State(state)).await).await["graded"], 1);
    }
}
