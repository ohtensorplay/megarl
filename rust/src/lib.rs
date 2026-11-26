//! PyO3 surface of megaRL. Kept deliberately narrow: the Python layer owns all
//! algorithmic semantics; Rust owns scheduling, framing and lease bookkeeping.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::BTreeMap;

use megarl_optimizer::{
    adamw_step_f32 as rb_adamw_step, bytes_to_f32 as rb_bytes_to_f32,
    cast_f32_to_bytes as rb_cast_f32_to_bytes, fingerprint as rb_fingerprint,
    invert_adamw_f32 as rb_invert_adamw, invert_and_diff_bytes as rb_invert_and_diff,
    AdamwInversionDetector as RbInversionDetector,
    AdamwParams as RbAdamwParams, StorageDtype as RbStorageDtype,
};
use megarl_control::{
    AsyncPacer as RbAsyncPacer, LedgerError, SlotLease as RbSlotLease, StalenessBudget as RbStalenessBudget,
    TokenBucketRateLimiter as RbRateLimiter, TokenSlotSemaphore as RbTokenSlotSemaphore,
    VersionLedger as RbVersionLedger,
};
use megarl_protocol::{validate_payload, Kind, ProtocolError, VERSION};
use megarl_transport::{
    map_range_to_shards, plan_tp_shards, write_shard, LeaseTable as RbLeaseTable,
    LocalRdmaProvider, MemoryRegion, RdmaProvider, TensorShard,
};
use std::time::Duration;

use megarl_reward::{
    extract_answer as rb_extract_answer, grade_benchmark_json as rb_grade_benchmark_json,
    math_equivalent as rb_math_equivalent, normalize_answer as rb_normalize_answer,
    run_sandboxed as rb_run_sandboxed, verify_batch_json as rb_verify_batch_json,
    verify_math as rb_verify_math, FirejailMode, SandboxConfig, SeccompMode,
};
use megarl_reward::seccomp::{find_firejail, sandbox_label};
use megarl_agent::{
    parse_tool_calls as rb_parse_tool_calls, validate_arguments as rb_validate_arguments,
};
use megarl_nettools::{http_fetch as rb_http_fetch, FetchRequest as RbFetchRequest};
use std::sync::OnceLock;

fn protocol_err(e: ProtocolError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pyfunction]
fn protocol_version() -> u16 {
    VERSION
}

#[pyfunction]
fn kind_ids() -> BTreeMap<String, u8> {
    [
        Kind::RolloutRequest,
        Kind::TrajectoryRecord,
        Kind::WeightUpdate,
        Kind::RewardOutcome,
        Kind::MultimodalTrajectory,
        Kind::HealthCheck,
        Kind::AgentTrajectory,
        Kind::LifecycleCommand,
        Kind::LifecycleAck,
        Kind::WeightUpdateAck,
    ]
    .iter()
    .map(|k| (k.as_str().to_string(), *k as u8))
    .collect()
}

/// Encode a JSON payload into a versioned MGRL frame.
#[pyfunction]
fn encode<'py>(py: Python<'py>, kind: &str, payload_json: &str) -> PyResult<Bound<'py, PyBytes>> {
    let kind = Kind::from_str(kind)
        .ok_or_else(|| PyValueError::new_err(format!("unknown message kind: {kind}")))?;
    let value: serde_json::Value = serde_json::from_str(payload_json)
        .map_err(|e| PyValueError::new_err(format!("invalid payload JSON: {e}")))?;
    let frame = megarl_protocol::encode(kind, &value).map_err(protocol_err)?;
    Ok(PyBytes::new_bound(py, &frame))
}

/// Decode a frame into (kind name, JSON payload string).
#[pyfunction]
fn decode(frame: &[u8]) -> PyResult<(String, String)> {
    let (kind, value) = megarl_protocol::decode(frame).map_err(protocol_err)?;
    let json = serde_json::to_string(&value)
        .map_err(|e| PyValueError::new_err(format!("payload serialization failed: {e}")))?;
    Ok((kind.as_str().to_string(), json))
}

/// Catch field drift between the Python mirror and the authoritative Rust struct.
#[pyfunction]
fn validate(kind: &str, payload_json: &str) -> PyResult<()> {
    let kind = Kind::from_str(kind)
        .ok_or_else(|| PyValueError::new_err(format!("unknown message kind: {kind}")))?;
    validate_payload(kind, payload_json).map_err(protocol_err)
}

#[pyclass(frozen)]
#[derive(Clone)]
struct StalenessBudget {
    inner: RbStalenessBudget,
}

#[pymethods]
impl StalenessBudget {
    #[new]
    fn new(max_staleness: u64, consumer_batch_size: u32, concurrency: u32) -> Self {
        Self {
            inner: RbStalenessBudget {
                max_staleness,
                consumer_batch_size,
                concurrency,
            },
        }
    }

    fn capacity(&self, current_version: u64, accepted: u64, running: u64) -> u32 {
        self.inner.capacity(current_version, accepted, running)
    }

    fn is_stale(&self, current_version: u64, token_version: u64) -> bool {
        self.inner.is_stale(current_version, token_version)
    }

    /// Admitted-token count to seed after restoring a checkpoint at `version`.
    fn rebase_after_recovery(&self, version: u64) -> u64 {
        self.inner.rebase_after_recovery(version)
    }
}

#[pyclass]
struct VersionLedger {
    inner: RbVersionLedger,
}

#[pymethods]
impl VersionLedger {
    #[new]
    fn new() -> Self {
        Self {
            inner: RbVersionLedger::new(),
        }
    }

    fn record(
        &mut self,
        request_id: u64,
        token_count: usize,
        versions: Vec<u64>,
    ) -> PyResult<()> {
        self.inner
            .record(request_id, token_count, versions)
            .map_err(|e: LedgerError| PyValueError::new_err(format!("{e:?}")))
    }

    fn forget(&mut self, request_id: u64) -> bool {
        self.inner.forget(request_id)
    }

    fn stale_tokens(&self, budget: &StalenessBudget, current_version: u64) -> usize {
        self.inner.stale_tokens(&budget.inner, current_version)
    }

    fn token_count(&self) -> usize {
        self.inner.token_count()
    }

    fn versions(&self, request_id: u64) -> Option<Vec<u64>> {
        self.inner.versions(request_id).map(|v| v.to_vec())
    }
}

#[pyclass]
struct LeaseTable {
    inner: RbLeaseTable,
}

#[pymethods]
impl LeaseTable {
    #[new]
    fn new() -> Self {
        Self {
            inner: RbLeaseTable::new(),
        }
    }

    fn acquire(&mut self, handle: u64, ttl_secs: f64) -> u64 {
        self.inner.acquire(handle, Duration::from_secs_f64(ttl_secs))
    }

    fn renew(&mut self, id: u64, ttl_secs: f64) -> bool {
        self.inner.renew(id, Duration::from_secs_f64(ttl_secs))
    }

    fn release(&mut self, id: u64) -> Option<u64> {
        self.inner.release(id)
    }

    fn reclaim_expired(&mut self) -> Vec<u64> {
        self.inner.reclaim_expired()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass]
struct RateLimiter {
    inner: RbRateLimiter,
}

#[pymethods]
impl RateLimiter {
    #[new]
    fn new(rate_per_sec: f64, burst: u64) -> Self {
        Self {
            inner: RbRateLimiter::new(rate_per_sec, burst),
        }
    }

    /// Consume `count` tokens if available; never consumes on failure.
    fn try_acquire(&mut self, count: u64) -> bool {
        self.inner.try_acquire(count)
    }

    fn retry_after_ms(&mut self, count: u64) -> u64 {
        self.inner.retry_after_ms(count)
    }

    fn available(&mut self) -> f64 {
        self.inner.available()
    }
}

#[pyclass]
struct AsyncPacer {
    inner: RbAsyncPacer,
}

#[pymethods]
impl AsyncPacer {
    #[new]
    fn new(rate_per_sec: f64, batch_size: u32) -> Self {
        Self {
            inner: RbAsyncPacer::new(rate_per_sec, batch_size),
        }
    }

    /// How many of `waiting` requests may be woken now (0 when not enough
    /// credits have accrued). Call repeatedly on each tick.
    fn wake(&mut self, waiting: u32) -> u32 {
        self.inner.wake(waiting)
    }

    fn retry_after_ms(&mut self) -> u64 {
        self.inner.retry_after_ms()
    }
}

#[pyclass]
struct TokenSlotSemaphore {
    inner: RbTokenSlotSemaphore,
}

#[pymethods]
impl TokenSlotSemaphore {
    #[new]
    fn new(capacity: usize, ttl_secs: f64) -> Self {
        Self {
            inner: RbTokenSlotSemaphore::new(capacity, ttl_secs),
        }
    }

    /// Returns ``(index, token)`` on success, ``None`` when every slot is held.
    fn try_acquire(&mut self) -> Option<(usize, u64)> {
        self.inner.try_acquire().map(|lease| (lease.index, lease.token))
    }

    /// Releases a slot only when the lease token still matches.
    fn release(&mut self, index: usize, token: u64) -> bool {
        self.inner.release(RbSlotLease { index, token })
    }

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn available(&mut self) -> usize {
        // The Python surface uses wall-clock time; Rust reaps expired leases.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        self.inner.available(now)
    }

    fn in_use(&self) -> usize {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        self.inner.in_use(now)
    }
}

#[pyclass]
struct TpLayout {
    shards: Vec<TensorShard>,
}

#[pymethods]
impl TpLayout {
    #[new]
    fn new(name: String, shape: Vec<u64>, tp_size: u32, shard_dim: usize) -> Self {
        Self {
            shards: plan_tp_shards(&name, &shape, tp_size, shard_dim),
        }
    }

    /// Rank, global start, local offset, length for each touched rank.
    fn map_range(&self, offset: u64, length: u64) -> Vec<(u32, u64, u64, u64)> {
        map_range_to_shards(&self.shards, offset, length)
    }

    fn shards_json(&self) -> String {
        serde_json::json!(
            self.shards
                .iter()
                .map(|s| serde_json::json!({
                    "name": s.name,
                    "rank": s.rank,
                    "tp_size": s.tp_size,
                    "shard_dim": s.shard_dim,
                    "global_offset": s.global_offset,
                    "length": s.length,
                }))
                .collect::<Vec<_>>()
        )
        .to_string()
    }
}

#[pyclass]
struct RdmaBuffer {
    provider: LocalRdmaProvider,
    region: MemoryRegion,
}

#[pymethods]
impl RdmaBuffer {
    #[new]
    fn new(size: usize) -> PyResult<Self> {
        let mut provider = LocalRdmaProvider::default();
        let region = provider
            .register(&vec![0u8; size])
            .map_err(PyValueError::new_err)?;
        Ok(Self { provider, region })
    }

    /// One-sided remote write into the registered region.
    fn write(&mut self, offset: u64, data: &[u8]) -> PyResult<()> {
        write_shard(&mut self.provider, &self.region, offset, data)
            .map_err(PyValueError::new_err)
    }

    fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = self
            .provider
            .read(&self.region)
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new_bound(py, &data))
    }
}

// ---------------------------------------------------------------------------
// Reward verification (implemented entirely in megarl-reward)
// ---------------------------------------------------------------------------

#[pyfunction]
fn extract_answer(text: &str) -> String {
    rb_extract_answer(text)
}

#[pyfunction]
fn normalize_math(text: &str) -> String {
    rb_normalize_answer(text)
}

#[pyfunction]
fn math_equivalent(predicted: &str, expected: &str, tolerance: f64) -> bool {
    rb_math_equivalent(predicted, expected, tolerance)
}

#[pyfunction]
fn verify_math(response: &str, expected: &str, tolerance: f64) -> f32 {
    rb_verify_math(response, expected, tolerance)
}

/// Verify a batch of domain-routed requests. Input and output are JSON so the
/// Python layer stays a thin transport adapter, not a second implementation.
#[pyfunction]
fn verify_batch_json(requests_json: &str, workers: usize, max_retries: u32) -> PyResult<(String, String)> {
    rb_verify_batch_json(requests_json, workers, max_retries)
        .map_err(|e| PyValueError::new_err(format!("invalid verify batch: {e}")))
}

/// Grade a candidate solution against a benchmark spec (HumanEval / MBPP /
/// APPS / custom). Input and result are JSON: the spec carries test cases and
/// the result carries a fractional reward plus per-test outcomes.
#[pyfunction]
fn grade_benchmark_json(spec_json: &str, candidate: &str) -> PyResult<String> {
    rb_grade_benchmark_json(spec_json, candidate)
        .map_err(|e| PyValueError::new_err(format!("invalid benchmark spec: {e}")))
}

// ---------------------------------------------------------------------------
// Agentic tool parsing (implemented in megarl-agent)
// ---------------------------------------------------------------------------

/// Parse tool calls from a decoded assistant message. Returns a JSON array of
/// `{"id","name","arguments_json"}`; parsing is all-or-nothing, so a malformed
/// later call never lets a valid prefix execute.
#[pyfunction]
fn parse_tool_calls(text: &str, format: &str) -> PyResult<String> {
    let calls = rb_parse_tool_calls(text, format).map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&calls)
        .map_err(|e| PyValueError::new_err(format!("tool call serialization failed: {e}")))
}

/// Validate canonical tool arguments against the tool's JSON Schema subset.
#[pyfunction]
fn validate_tool_arguments(schema_json: &str, arguments_json: &str) -> PyResult<()> {
    rb_validate_arguments(schema_json, arguments_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn net_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("megarl-nettools")
            .build()
            .expect("nettools runtime")
    })
}

/// Hardened HTTP request for the network tool layer. The config and result are
/// JSON so the Python adapter stays thin; all policy checks live in Rust. The
/// GIL is released while the request is in flight.
#[pyfunction]
fn http_request_json(py: Python<'_>, config_json: &str) -> PyResult<String> {
    let request: RbFetchRequest = serde_json::from_str(config_json)
        .map_err(|e| PyValueError::new_err(format!("invalid HTTP config: {e}")))?;
    let outcome = py.allow_threads(|| net_runtime().block_on(rb_http_fetch(request)));
    let value = match outcome {
        Ok(response) => serde_json::json!({"ok": true, "response": response}),
        Err(error) => serde_json::json!({"ok": false, "error": error}),
    };
    serde_json::to_string(&value)
        .map_err(|e| PyValueError::new_err(format!("HTTP result serialization failed: {e}")))
}

#[derive(serde::Deserialize)]
struct PySandboxRequest {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    stdin: String,
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
    #[serde(default = "default_memory_mb")]
    memory_mb: u64,
    #[serde(default = "default_max_output_bytes")]
    max_output_bytes: usize,
    #[serde(default)]
    seccomp: SeccompMode,
    #[serde(default)]
    firejail: FirejailMode,
    #[serde(default)]
    firejail_profile: Option<String>,
    #[serde(default)]
    firejail_bin: Option<String>,
}

fn default_timeout_ms() -> u64 {
    5_000
}
fn default_memory_mb() -> u64 {
    512
}
fn default_max_output_bytes() -> usize {
    1 << 20
}

/// Run a command under RLIMIT_AS/CPU/FSIZE and a hard wall-clock deadline.
#[pyfunction]
fn run_sandboxed_json(config_json: &str) -> PyResult<String> {
    let request: PySandboxRequest = serde_json::from_str(config_json)
        .map_err(|e| PyValueError::new_err(format!("invalid sandbox config: {e}")))?;
    let config = SandboxConfig {
        timeout_ms: request.timeout_ms,
        memory_mb: request.memory_mb,
        max_output_bytes: request.max_output_bytes,
        seccomp: request.seccomp,
        firejail: request.firejail,
        firejail_profile: request.firejail_profile,
        firejail_bin: request.firejail_bin,
        ..SandboxConfig::default()
    };
    let result = rb_run_sandboxed(
        &request.command,
        &request.args,
        request.stdin.as_bytes(),
        &config,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let json = serde_json::json!({
        "exit_code": result.exit_code,
        "signal": result.signal,
        "stdout": String::from_utf8_lossy(&result.stdout),
        "stderr": String::from_utf8_lossy(&result.stderr),
        "timed_out": result.timed_out,
        "output_truncated": result.output_truncated,
        "elapsed_ms": result.elapsed_ms,
        "sandbox": result.sandbox,
        "succeeded": result.succeeded(),
    })
    .to_string();
    Ok(json)
}

/// Describe the sandbox profile active in this process, for operator checks.
#[pyfunction]
fn sandbox_profile() -> String {
    let defaults = SandboxConfig::default();
    serde_json::json!({
        "seccomp": defaults.seccomp.as_str(),
        "firejail": defaults.firejail.as_str(),
        "firejail_available": find_firejail(None).is_some(),
        "effective_label": sandbox_label(find_firejail(None).is_some(), defaults.seccomp),
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// AdamW inversion — no-baseline delta detection (implemented in megarl-optimizer)
// ---------------------------------------------------------------------------

fn dtype(name: &str) -> PyResult<RbStorageDtype> {
    RbStorageDtype::from_str(name).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn adamw_params(
    step: u64,
    lr: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
) -> RbAdamwParams {
    RbAdamwParams {
        step,
        lr,
        weight_decay,
        beta1,
        beta2,
        eps,
    }
}

/// Reconstruct pre-step weights from post-step weights and post-step Adam
/// moments. All tensors are little-endian storage bytes (`float32` or
/// `bfloat16`); reconstruction is always fp32 internally, then cast back.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn invert_adamw_bytes(
    theta: &[u8],
    exp_avg: &[u8],
    exp_avg_sq: &[u8],
    step: u64,
    lr: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    dtype_name: &str,
    moments_dtype: &str,
) -> PyResult<Vec<u8>> {
    let target = dtype(dtype_name)?;
    // Adam moments are fp32 master values even when the visible payload is
    // bf16; decoding them at the visible dtype would lose the very precision
    // the inversion relies on.
    let moments = dtype(moments_dtype)?;
    let theta_f32 = rb_bytes_to_f32(theta, target)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let m_f32 =
        rb_bytes_to_f32(exp_avg, moments).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let v_f32 =
        rb_bytes_to_f32(exp_avg_sq, moments).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let recovered = rb_invert_adamw(
        &theta_f32,
        &m_f32,
        &v_f32,
        adamw_params(step, lr, weight_decay, beta1, beta2, eps),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(rb_cast_f32_to_bytes(&recovered, target))
}

/// One AdamW step over storage bytes. Exposed as the reference forward
/// direction so adapters can validate an inversion against a known step.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn adamw_step_bytes(
    theta: &[u8],
    exp_avg: &[u8],
    exp_avg_sq: &[u8],
    grad: &[u8],
    step: u64,
    lr: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    dtype_name: &str,
) -> PyResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let target = dtype(dtype_name)?;
    let mut theta_f32 =
        rb_bytes_to_f32(theta, target).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut m_f32 =
        rb_bytes_to_f32(exp_avg, target).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut v_f32 =
        rb_bytes_to_f32(exp_avg_sq, target).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let grad_f32 =
        rb_bytes_to_f32(grad, target).map_err(|e| PyValueError::new_err(e.to_string()))?;
    rb_adamw_step(
        &mut theta_f32,
        &mut m_f32,
        &mut v_f32,
        &grad_f32,
        adamw_params(step, lr, weight_decay, beta1, beta2, eps),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        rb_cast_f32_to_bytes(&theta_f32, target),
        rb_cast_f32_to_bytes(&m_f32, target),
        rb_cast_f32_to_bytes(&v_f32, target),
    ))
}

/// Conservative element-granular diff against a reconstructed pre-step
/// baseline. Returns `(offset, chunk)` pairs that cover every changed byte.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn invert_and_diff_bytes(
    theta: &[u8],
    exp_avg: &[u8],
    exp_avg_sq: &[u8],
    step: u64,
    lr: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    dtype_name: &str,
    moments_dtype: &str,
) -> PyResult<Vec<(usize, Vec<u8>)>> {
    rb_invert_and_diff(
        theta,
        exp_avg,
        exp_avg_sq,
        adamw_params(step, lr, weight_decay, beta1, beta2, eps),
        dtype(dtype_name)?,
        dtype(moments_dtype)?,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compact value fingerprint of a visible weight payload (not a snapshot).
#[pyfunction]
fn weight_fingerprint(data: &[u8], dtype_name: &str) -> PyResult<u64> {
    Ok(rb_fingerprint(data, dtype(dtype_name)?))
}

#[pyclass]
struct AdamwInversionDetector {
    inner: RbInversionDetector,
}

#[pymethods]
impl AdamwInversionDetector {
    #[new]
    fn new() -> Self {
        Self {
            inner: RbInversionDetector::new(),
        }
    }

    /// True when the visible payload changed and the step advanced past the
    /// last successful sync — i.e. a fresh inversion is warranted.
    fn should_detect(&self, name: &str, visible: &[u8], dtype_name: &str, step: u64) -> PyResult<bool> {
        Ok(self.inner.should_detect(name, visible, dtype(dtype_name)?, step))
    }

    fn mark_synced(&mut self, name: &str, visible: &[u8], dtype_name: &str, step: u64) -> PyResult<()> {
        self.inner.mark_synced(name, visible, dtype(dtype_name)?, step);
        Ok(())
    }

    fn needs_sync(&self, name: &str) -> bool {
        self.inner.needs_sync(name)
    }
}

#[pymodule]
fn megarl_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(protocol_version, m)?)?;
    m.add_function(wrap_pyfunction!(kind_ids, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add("StalenessBudget", m.py().get_type_bound::<StalenessBudget>())?;
    m.add("VersionLedger", m.py().get_type_bound::<VersionLedger>())?;
    m.add("LeaseTable", m.py().get_type_bound::<LeaseTable>())?;
    m.add("RateLimiter", m.py().get_type_bound::<RateLimiter>())?;
    m.add("AsyncPacer", m.py().get_type_bound::<AsyncPacer>())?;
    m.add(
        "TokenSlotSemaphore",
        m.py().get_type_bound::<TokenSlotSemaphore>(),
    )?;
    m.add("TpLayout", m.py().get_type_bound::<TpLayout>())?;
    m.add("RdmaBuffer", m.py().get_type_bound::<RdmaBuffer>())?;
    m.add(
        "AdamwInversionDetector",
        m.py().get_type_bound::<AdamwInversionDetector>(),
    )?;
    m.add_function(wrap_pyfunction!(invert_adamw_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(invert_and_diff_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(adamw_step_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(weight_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(extract_answer, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_math, m)?)?;
    m.add_function(wrap_pyfunction!(math_equivalent, m)?)?;
    m.add_function(wrap_pyfunction!(verify_math, m)?)?;
    m.add_function(wrap_pyfunction!(verify_batch_json, m)?)?;
    m.add_function(wrap_pyfunction!(grade_benchmark_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_sandboxed_json, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox_profile, m)?)?;
    m.add_function(wrap_pyfunction!(parse_tool_calls, m)?)?;
    m.add_function(wrap_pyfunction!(validate_tool_arguments, m)?)?;
    m.add_function(wrap_pyfunction!(http_request_json, m)?)?;
    Ok(())
}
