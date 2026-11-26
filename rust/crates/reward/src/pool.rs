//! Domain routing and a bounded, retrying verification pool.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::math::verify_math;
use crate::sandbox::{run_sandboxed, SandboxConfig};
use crate::seccomp::{FirejailMode, SeccompMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Math,
    Code,
    Generic,
}

impl Domain {
    pub fn from_str(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "math" | "math_rule" | "gsm8k" | "math_verify" => Domain::Math,
            "code" | "human_eval" | "program" | "python" => Domain::Code,
            _ => Domain::Generic,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Domain::Math => "math",
            Domain::Code => "code",
            Domain::Generic => "generic",
        }
    }
}

pub fn route_domain(name: &str) -> Domain {
    Domain::from_str(name)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyRequest {
    pub request_id: u64,
    pub domain: String,
    pub response: String,
    pub expected: String,
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    /// Executable used by the code domain (e.g. `python3`).
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_memory_mb")]
    pub memory_mb: u64,
    /// Isolation layer for the code domain (same shape as `BenchmarkSpec`).
    #[serde(default)]
    pub seccomp: SeccompMode,
    #[serde(default)]
    pub firejail: FirejailMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firejail_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firejail_bin: Option<String>,
}

fn default_tolerance() -> f64 {
    1e-5
}
fn default_timeout_ms() -> u64 {
    5_000
}
fn default_memory_mb() -> u64 {
    512
}

impl Default for VerifyRequest {
    fn default() -> Self {
        Self {
            request_id: 0,
            domain: "generic".into(),
            response: String::new(),
            expected: String::new(),
            tolerance: default_tolerance(),
            command: String::new(),
            args: Vec::new(),
            timeout_ms: default_timeout_ms(),
            memory_mb: default_memory_mb(),
            seccomp: SeccompMode::Blacklist,
            firejail: FirejailMode::Auto,
            firejail_profile: None,
            firejail_bin: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerifyOutcome {
    pub request_id: u64,
    pub reward: f32,
    pub domain: String,
    pub verifier: String,
    pub elapsed_ms: u64,
    #[serde(default)]
    pub retries: u32,
    #[serde(default)]
    pub timed_out: bool,
    /// Isolation layer actually applied (code domain only).
    #[serde(default)]
    pub sandbox: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Default)]
pub struct PoolStats {
    completed: AtomicU64,
    failed: AtomicU64,
    retries: AtomicU64,
    timeouts: AtomicU64,
    elapsed_ms: AtomicU64,
}

impl PoolStats {
    pub fn values(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.completed.load(Ordering::Relaxed),
            self.failed.load(Ordering::Relaxed),
            self.retries.load(Ordering::Relaxed),
            self.timeouts.load(Ordering::Relaxed),
            self.elapsed_ms.load(Ordering::Relaxed),
        )
    }
}

pub struct VerifyPool {
    workers: usize,
    max_retries: u32,
    backoff_ms: u64,
    stats: Arc<PoolStats>,
}

impl Default for VerifyPool {
    fn default() -> Self {
        Self::new(1, 0)
    }
}

impl VerifyPool {
    pub fn new(workers: usize, max_retries: u32) -> Self {
        Self {
            workers: workers.max(1),
            max_retries,
            backoff_ms: 5,
            stats: Arc::new(PoolStats::default()),
        }
    }

    pub fn stats(&self) -> Arc<PoolStats> {
        Arc::clone(&self.stats)
    }

    pub fn stats_json(&self) -> String {
        let (completed, failed, retries, timeouts, elapsed_ms) = self.stats.values();
        serde_json::json!({
            "completed": completed,
            "failed": failed,
            "retries": retries,
            "timeouts": timeouts,
            "elapsed_ms": elapsed_ms,
        })
        .to_string()
    }

    /// Verify all requests concurrently, returning outcomes in input order.
    pub fn verify_batch(&self, requests: Vec<VerifyRequest>) -> Vec<VerifyOutcome> {
        let count = requests.len();
        if count == 0 {
            return Vec::new();
        }
        let queue: Mutex<VecDeque<(usize, VerifyRequest)>> =
            Mutex::new(requests.into_iter().enumerate().collect());
        let results: Mutex<Vec<Option<VerifyOutcome>>> =
            Mutex::new((0..count).map(|_| None).collect());
        let worker_count = self.workers.min(count);
        let retries_allowed = self.max_retries;
        let backoff = Duration::from_millis(self.backoff_ms);
        let stats = &self.stats;

        thread::scope(|scope| {
            for _ in 0..worker_count {
                let queue = &queue;
                let results = &results;
                let stats = stats;
                scope.spawn(move || loop {
                    let job = queue.lock().unwrap().pop_front();
                    let (index, request) = match job {
                        Some(job) => job,
                        None => break,
                    };
                    let started = Instant::now();
                    let mut outcome = verify_one(&request);
                    let mut attempts = 0;
                    // Only transient verifier errors are retried. A wrong
                    // answer is a legitimate score of zero, not a failure.
                    while outcome.error.is_some() && attempts < retries_allowed {
                        attempts += 1;
                        stats.retries.fetch_add(1, Ordering::Relaxed);
                        thread::sleep(backoff);
                        outcome = verify_one(&request);
                    }
                    outcome.retries = attempts;
                    outcome.elapsed_ms = started.elapsed().as_millis() as u64;
                    if outcome.error.is_some() {
                        stats.failed.fetch_add(1, Ordering::Relaxed);
                    } else {
                        stats.completed.fetch_add(1, Ordering::Relaxed);
                    }
                    if outcome.timed_out {
                        stats.timeouts.fetch_add(1, Ordering::Relaxed);
                    }
                    stats
                        .elapsed_ms
                        .fetch_add(outcome.elapsed_ms as u64, Ordering::Relaxed);
                    results.lock().unwrap()[index] = Some(outcome);
                });
            }
        });

        results
            .into_inner()
            .unwrap()
            .into_iter()
            .map(|outcome| outcome.expect("every job must produce an outcome"))
            .collect()
    }
}

pub fn verify_one(request: &VerifyRequest) -> VerifyOutcome {
    let domain = route_domain(&request.domain);
    let mut outcome = VerifyOutcome {
        request_id: request.request_id,
        domain: domain.as_str().to_string(),
        ..VerifyOutcome::default()
    };
    match domain {
        Domain::Math => {
            outcome.verifier = "math_equivalence".into();
            outcome.reward = verify_math(&request.response, &request.expected, request.tolerance);
        }
        Domain::Generic => {
            outcome.verifier = "exact_match".into();
            outcome.reward = if request.response.trim() == request.expected.trim() {
                1.0
            } else {
                0.0
            };
        }
        Domain::Code => {
            outcome.verifier = "sandboxed_exec".into();
            if request.command.is_empty() {
                outcome.error = Some("code domain requires a command".into());
                return outcome;
            }
            let config = SandboxConfig {
                timeout_ms: request.timeout_ms,
                memory_mb: request.memory_mb,
                seccomp: request.seccomp,
                firejail: request.firejail,
                firejail_profile: request.firejail_profile.clone(),
                firejail_bin: request.firejail_bin.clone(),
                ..SandboxConfig::default()
            };
            match run_sandboxed(&request.command, &request.args, request.response.as_bytes(), &config) {
                Ok(result) => {
                    outcome.sandbox = result.sandbox.clone();
                    outcome.timed_out = result.timed_out;
                    if result.timed_out {
                        outcome.error = Some("code verifier timed out".into());
                    }
                    outcome.reward = if result.succeeded() { 1.0 } else { 0.0 };
                }
                Err(e) => outcome.error = Some(e.to_string()),
            }
        }
    }
    outcome
}

/// JSON-in/JSON-out helper used by the PyO3 surface and external services.
pub fn verify_batch_json(
    requests_json: &str,
    workers: usize,
    max_retries: u32,
) -> Result<(String, String), serde_json::Error> {
    let requests: Vec<VerifyRequest> = serde_json::from_str(requests_json)?;
    let pool = VerifyPool::new(workers, max_retries);
    let outcomes = pool.verify_batch(requests);
    Ok((serde_json::to_string(&outcomes)?, pool.stats_json()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_domains() {
        assert_eq!(route_domain("GSM8K"), Domain::Math);
        assert_eq!(route_domain("human_eval"), Domain::Code);
        assert_eq!(route_domain("something_else"), Domain::Generic);
    }

    #[test]
    fn verifies_math_and_generic() {
        let math = VerifyRequest {
            request_id: 1,
            domain: "math".into(),
            response: "work\\boxed{0.5}".into(),
            expected: "1/2".into(),
            ..VerifyRequest::default()
        };
        assert_eq!(verify_one(&math).reward, 1.0);
        let generic = VerifyRequest {
            request_id: 2,
            response: "yes ".into(),
            expected: "yes".into(),
            ..VerifyRequest::default()
        };
        assert_eq!(verify_one(&generic).reward, 1.0);
    }

    #[test]
    fn code_domain_uses_sandbox() {
        let ok = VerifyRequest {
            domain: "code".into(),
            command: "/bin/sh".into(),
            args: vec!["-c".into(), "read line; test \"$line\" = pass".into()],
            response: "pass\n".into(),
            timeout_ms: 2_000,
            ..VerifyRequest::default()
        };
        let outcome = verify_one(&ok);
        assert_eq!(outcome.reward, 1.0);
        assert!(outcome.error.is_none());

        let fail = VerifyRequest { response: "nope\n".into(), ..ok };
        assert_eq!(verify_one(&fail).reward, 0.0);
    }

    #[test]
    fn pool_preserves_order_and_reports_stats() {
        let requests: Vec<VerifyRequest> = (0..12)
            .map(|i| VerifyRequest {
                request_id: i,
                domain: "math".into(),
                response: format!("{i}"),
                expected: format!("{i}"),
                ..VerifyRequest::default()
            })
            .collect();
        let pool = VerifyPool::new(4, 1);
        let outcomes = pool.verify_batch(requests);
        assert_eq!(outcomes.len(), 12);
        for (i, outcome) in outcomes.iter().enumerate() {
            assert_eq!(outcome.request_id, i as u64);
            assert_eq!(outcome.reward, 1.0);
        }
        let (completed, failed, retries, timeouts, _) = pool.stats().values();
        assert_eq!(completed, 12);
        assert_eq!(failed, 0);
        assert_eq!(retries, 0);
        assert_eq!(timeouts, 0);
    }

    #[test]
    fn json_batch_roundtrip() {
        let requests = r#"[{"request_id":7,"domain":"math","response":"0.5","expected":"1/2"}]"#;
        let (outcomes, stats) = verify_batch_json(requests, 2, 1).unwrap();
        assert!(outcomes.contains("\"reward\":1.0"));
        assert!(stats.contains("\"completed\":1"));
    }
}
