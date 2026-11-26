//! Versioned wire protocol for megaRL.
//!
//! Design rule (see devdocs/architecture.md): a frame carries metadata and
//! out-of-band handles only. Tensor payloads never pass through the controller.
//! The Rust structs here are authoritative; `src/megarl/contracts.py` mirrors
//! them and `validate_payload` catches drift.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

pub const MAGIC: [u8; 4] = *b"MGRL";
pub const VERSION: u16 = 1;
const HEADER_LEN: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    RolloutRequest = 1,
    TrajectoryRecord = 2,
    WeightUpdate = 3,
    RewardOutcome = 4,
    MultimodalTrajectory = 5,
    /// Lightweight liveness probe; never triggers generation or weight reload.
    HealthCheck = 6,
    /// Multi-turn agent trajectory: flattened tokens with provenance masks.
    AgentTrajectory = 7,
    /// Colocate lifecycle command (pause/drain/offload or onload/resume).
    LifecycleCommand = 8,
    /// Worker acknowledgement of a lifecycle command, epoch-guarded.
    LifecycleAck = 9,
    /// Worker-reported weight-reload phase timings (pause/barrier/transfer/apply).
    WeightUpdateAck = 10,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::RolloutRequest => "RolloutRequest",
            Kind::TrajectoryRecord => "TrajectoryRecord",
            Kind::WeightUpdate => "WeightUpdate",
            Kind::RewardOutcome => "RewardOutcome",
            Kind::MultimodalTrajectory => "MultimodalTrajectory",
            Kind::HealthCheck => "HealthCheck",
            Kind::AgentTrajectory => "AgentTrajectory",
            Kind::LifecycleCommand => "LifecycleCommand",
            Kind::LifecycleAck => "LifecycleAck",
            Kind::WeightUpdateAck => "WeightUpdateAck",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "RolloutRequest" => Some(Kind::RolloutRequest),
            "TrajectoryRecord" => Some(Kind::TrajectoryRecord),
            "WeightUpdate" => Some(Kind::WeightUpdate),
            "RewardOutcome" => Some(Kind::RewardOutcome),
            "MultimodalTrajectory" => Some(Kind::MultimodalTrajectory),
            "HealthCheck" => Some(Kind::HealthCheck),
            "AgentTrajectory" => Some(Kind::AgentTrajectory),
            "LifecycleCommand" => Some(Kind::LifecycleCommand),
            "LifecycleAck" => Some(Kind::LifecycleAck),
            "WeightUpdateAck" => Some(Kind::WeightUpdateAck),
            _ => None,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Kind::RolloutRequest),
            2 => Some(Kind::TrajectoryRecord),
            3 => Some(Kind::WeightUpdate),
            4 => Some(Kind::RewardOutcome),
            5 => Some(Kind::MultimodalTrajectory),
            6 => Some(Kind::HealthCheck),
            7 => Some(Kind::AgentTrajectory),
            8 => Some(Kind::LifecycleCommand),
            9 => Some(Kind::LifecycleAck),
            10 => Some(Kind::WeightUpdateAck),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub seed: Option<u64>,
    pub logprobs: bool,
    pub top_logprobs: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutRequest {
    pub request_id: u64,
    pub policy_version: u64,
    pub prompt_token_ids: Vec<u32>,
    pub max_new_tokens: u32,
    pub n: u32,
    pub sampling: SamplingParams,
    pub session_id: Option<u64>,
    /// Optional model name. When present the gateway constrains routing to
    /// workers holding that model's active version. Backwards compatible:
    /// absent means "no model constraint".
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryRecord {
    pub request_id: u64,
    pub prompt_len: u32,
    pub token_ids: Vec<u32>,
    /// Processed logprobs from the actual generating policy, 1:1 with token_ids.
    pub logprobs: Vec<f32>,
    /// (token_id, logprob) pairs, aligned with token_ids in v1.
    pub top_logprobs: Vec<Vec<(u32, f32)>>,
    /// Per-token policy version. Partial rollout makes prefixes mixed-version.
    pub policy_versions: Vec<u64>,
    pub reward: Option<f32>,
    pub finish_reason: u8,
    /// Out-of-band shared-memory / RDMA handle; never an inline tensor.
    pub payload_handle: Option<u64>,
    /// Logprobs reported by the sampling engine (vLLM). Distinct from the
    /// trainer's recomputed logprobs; empty disables IS correction.
    #[serde(default)]
    pub sampling_logprobs: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultimodalTrajectory {
    pub request_id: u64,
    pub prompt_len: u32,
    pub token_ids: Vec<u32>,
    pub logprobs: Vec<f32>,
    pub policy_versions: Vec<u64>,
    /// Image token ids inserted into the prompt at `image_insert_at`.
    pub image_token_ids: Vec<u32>,
    pub image_insert_at: u32,
    /// (height, width) token grid for the visual encoder.
    pub image_grid: (u32, u32),
    /// 0 = text token, 1 = image token; aligned with the composed prompt.
    pub modality_mask: Vec<u8>,
    pub reward: Option<f32>,
    pub finish_reason: u8,
    pub payload_handle: Option<u64>,
}

/// One tool invocation emitted by an assistant turn.
///
/// `arguments_json` is a serialized JSON object validated against the tool's
/// schema before execution; keeping it as a string makes the frame stable
/// across tool argument types and avoids re-serialization in the controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentToolCall {
    /// Stable id used to pair the call with its observation.
    pub id: String,
    pub name: String,
    pub arguments_json: String,
    /// 0-based assistant turn that emitted the call.
    pub turn: u32,
    /// Whether the tool returned successfully.
    pub ok: bool,
    /// Wall-clock milliseconds spent executing the tool (0 if not executed).
    pub elapsed_ms: u64,
}

/// Multi-turn agent trajectory.
///
/// `token_ids` is the *full* flattened sequence (initial prompt + every
/// assistant turn + every tool observation). All response-aligned vectors
/// (`logprobs`, `policy_versions`, `sampling_logprobs`, `top_logprobs`) cover
/// `token_ids[prompt_len..]`; positions with no policy token (tool
/// observations) carry zeros and are excluded by `loss_mask`.
///
/// The mask is authoritative: it is the only way tool/environment tokens are
/// kept out of the policy gradient. The controller never re-derives it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentTrajectory {
    pub request_id: u64,
    /// Policy version at admission; later turns may be newer (partial rollout).
    pub policy_version: u64,
    pub prompt_len: u32,
    pub token_ids: Vec<u32>,
    /// Processed logprobs of the generating policy, response-aligned.
    pub logprobs: Vec<f32>,
    pub top_logprobs: Vec<Vec<(u32, f32)>>,
    /// Per-token policy versions, response-aligned.
    pub policy_versions: Vec<u64>,
    /// 0 = prompt, 1 = assistant, 2 = tool observation; aligned with token_ids.
    pub token_roles: Vec<u8>,
    /// 0-based turn index per token; aligned with token_ids.
    pub turn_ids: Vec<u32>,
    /// 1 = trainable assistant token, 0 = prompt/tool/pad; aligned with token_ids.
    pub loss_mask: Vec<u8>,
    /// Sampling-engine logprobs, response-aligned; empty disables IS correction.
    #[serde(default)]
    pub sampling_logprobs: Vec<f32>,
    pub tool_calls: Vec<AgentToolCall>,
    /// Terminal reward for the whole episode.
    pub reward: Option<f32>,
    /// Per-turn rewards for credit assignment.
    #[serde(default)]
    pub step_rewards: Vec<f32>,
    /// 0 = stop, 1 = max_turns, 2 = max_tokens, 3 = tool_error, 4 = timeout.
    pub finish_reason: u8,
    pub payload_handle: Option<u64>,
}

impl AgentTrajectory {
    /// Check the alignment invariants that the batching layer relies on.
    /// Called by `validate_payload`, so Python `contracts.validate` catches
    /// malformed agent frames before they reach a trainer.
    pub fn check_invariants(&self) -> Result<(), ProtocolError> {
        let n = self.token_ids.len();
        let prompt = self.prompt_len as usize;
        if prompt > n {
            return Err(ProtocolError::AgentInvariant(format!(
                "prompt_len {} exceeds token_ids length {}",
                prompt, n
            )));
        }
        let response = n - prompt;
        let aligned = |name: &str, len: usize| -> Result<(), ProtocolError> {
            if len != response {
                Err(ProtocolError::AgentInvariant(format!(
                    "{name} has {len} entries, expected {response} response tokens"
                )))
            } else {
                Ok(())
            }
        };
        if self.token_roles.len() != n {
            return Err(ProtocolError::AgentInvariant(format!(
                "token_roles has {} entries, expected {n}",
                self.token_roles.len()
            )));
        }
        if self.turn_ids.len() != n {
            return Err(ProtocolError::AgentInvariant(format!(
                "turn_ids has {} entries, expected {n}",
                self.turn_ids.len()
            )));
        }
        if self.loss_mask.len() != n {
            return Err(ProtocolError::AgentInvariant(format!(
                "loss_mask has {} entries, expected {n}",
                self.loss_mask.len()
            )));
        }
        aligned("logprobs", self.logprobs.len())?;
        aligned("policy_versions", self.policy_versions.len())?;
        if !self.top_logprobs.is_empty() {
            aligned("top_logprobs", self.top_logprobs.len())?;
        }
        if !self.sampling_logprobs.is_empty() {
            aligned("sampling_logprobs", self.sampling_logprobs.len())?;
        }
        for (i, (role, mask)) in self
            .token_roles
            .iter()
            .zip(self.loss_mask.iter())
            .enumerate()
        {
            if *role > 2 {
                return Err(ProtocolError::AgentInvariant(format!(
                    "token_roles[{i}] = {role} is not prompt/assistant/tool"
                )));
            }
            if *mask > 1 {
                return Err(ProtocolError::AgentInvariant(format!(
                    "loss_mask[{i}] = {mask} is not binary"
                )));
            }
            if i < prompt && *mask != 0 {
                return Err(ProtocolError::AgentInvariant(format!(
                    "prompt token {i} must not be trainable"
                )));
            }
            if *role == 2 && *mask != 0 {
                return Err(ProtocolError::AgentInvariant(format!(
                    "tool observation token {i} must not be trainable"
                )));
            }
        }
        for (i, turn) in self.turn_ids.iter().enumerate() {
            if i > 0 && *turn < self.turn_ids[i - 1] {
                return Err(ProtocolError::AgentInvariant(format!(
                    "turn_ids decrease at position {i}"
                )));
            }
        }
        for call in &self.tool_calls {
            if !self.turn_ids.contains(&call.turn) {
                return Err(ProtocolError::AgentInvariant(format!(
                    "tool call {} references turn {} with no tokens",
                    call.id, call.turn
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub shard_rank: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightUpdate {
    pub from_version: u64,
    pub to_version: u64,
    /// ipc | nccl | shm | fs | rdma
    pub transport: String,
    /// SHA-256 hex of the new weight set; must verify after apply.
    pub digest: String,
    pub tensors: Vec<TensorDescriptor>,
    pub abort_inflight: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardOutcome {
    pub request_id: u64,
    pub reward: f32,
    pub domain: String,
    pub metadata: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HealthCheck {
    /// Monotonic nonce; a reply must echo it or it is stale/spoofed.
    pub probe: u64,
    pub ok: bool,
}

/// Colocate lifecycle action. `Pause` fences new generation, drains in-flight
/// work and optionally releases GPU memory; `Resume` restores memory and
/// continues generation. Mirrors AReaL's pause/pause_generation/offload and
/// onload/continue_generation sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LifecycleAction {
    Pause,
    Resume,
}

impl LifecycleAction {
    pub fn as_str(self) -> &'static str {
        match self {
            LifecycleAction::Pause => "pause",
            LifecycleAction::Resume => "resume",
        }
    }
}

/// State reported by a worker after applying a lifecycle command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LifecyclePhase {
    Running,
    Pausing,
    Paused,
    Offloaded,
    Resuming,
}

impl LifecyclePhase {
    pub fn as_str(self) -> &'static str {
        match self {
            LifecyclePhase::Running => "running",
            LifecyclePhase::Pausing => "pausing",
            LifecyclePhase::Paused => "paused",
            LifecyclePhase::Offloaded => "offloaded",
            LifecyclePhase::Resuming => "resuming",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleCommand {
    pub request_id: u64,
    /// Monotonic epoch; acks from an older epoch are stale and rejected.
    pub epoch: u64,
    pub action: LifecycleAction,
    /// Pause only: release weights / KV cache after draining in-flight work.
    #[serde(default)]
    pub offload: bool,
    /// Pause only: deadline in ms for draining in-flight generation.
    #[serde(default = "default_drain_ms")]
    pub drain_ms: u64,
}

fn default_drain_ms() -> u64 {
    30_000
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleAck {
    pub request_id: u64,
    pub epoch: u64,
    pub action: LifecycleAction,
    pub ok: bool,
    pub phase: LifecyclePhase,
    #[serde(default)]
    pub drained: u32,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub elapsed_ms: u64,
}

impl LifecycleAck {
    /// An ack is applicable only if it answers this exact command.
    pub fn matches(&self, command: &LifecycleCommand) -> bool {
        self.request_id == command.request_id
            && self.epoch == command.epoch
            && self.action == command.action
    }
}

/// Phase breakdown of a weight reload, measured by the worker. `transfer_ms`
/// covers staging/reading the weights outside the engine call; `apply_ms` is
/// the time inside `reload_weights`. `total_ms` is the worker-side wall clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightUpdateAck {
    pub request_id: u64,
    pub ok: bool,
    #[serde(default)]
    pub pause_ms: u64,
    #[serde(default)]
    pub barrier_ms: u64,
    #[serde(default)]
    pub transfer_ms: u64,
    #[serde(default)]
    pub apply_ms: u64,
    #[serde(default)]
    pub total_ms: u64,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug)]
pub enum ProtocolError {
    TooShort { len: usize },
    BadMagic,
    UnsupportedVersion(u16),
    UnknownKind(u8),
    LengthMismatch { declared: usize, actual: usize },
    /// An agent trajectory violated its mask/alignment invariants.
    AgentInvariant(String),
    Json(serde_json::Error),
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtocolError::TooShort { len } => write!(f, "frame too short: {len} bytes"),
            ProtocolError::BadMagic => write!(f, "bad magic, expected MGRL"),
            ProtocolError::UnsupportedVersion(v) => write!(f, "unsupported protocol version {v}"),
            ProtocolError::UnknownKind(k) => write!(f, "unknown message kind {k}"),
            ProtocolError::LengthMismatch { declared, actual } => {
                write!(f, "payload length mismatch: declared {declared}, got {actual}")
            }
            ProtocolError::AgentInvariant(msg) => write!(f, "agent trajectory invariant: {msg}"),
            ProtocolError::Json(e) => write!(f, "payload JSON error: {e}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

impl From<serde_json::Error> for ProtocolError {
    fn from(e: serde_json::Error) -> Self {
        ProtocolError::Json(e)
    }
}

/// Frame = MAGIC[4] | VERSION u16 LE | KIND u8 | FLAGS u8 | LEN u32 LE | JSON.
pub fn encode<T: Serialize>(kind: Kind, value: &T) -> Result<Vec<u8>, ProtocolError> {
    let payload = serde_json::to_vec(value)?;
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.push(kind as u8);
    out.push(0);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

pub fn decode(frame: &[u8]) -> Result<(Kind, Value), ProtocolError> {
    if frame.len() < HEADER_LEN {
        return Err(ProtocolError::TooShort { len: frame.len() });
    }
    if frame[0..4] != MAGIC {
        return Err(ProtocolError::BadMagic);
    }
    let version = u16::from_le_bytes([frame[4], frame[5]]);
    if version != VERSION {
        return Err(ProtocolError::UnsupportedVersion(version));
    }
    let kind = Kind::from_u8(frame[6]).ok_or(ProtocolError::UnknownKind(frame[6]))?;
    let declared = u32::from_le_bytes([frame[8], frame[9], frame[10], frame[11]]) as usize;
    let payload = &frame[HEADER_LEN..];
    if payload.len() != declared {
        return Err(ProtocolError::LengthMismatch {
            declared,
            actual: payload.len(),
        });
    }
    Ok((kind, serde_json::from_slice(payload)?))
}

pub fn decode_typed<T: DeserializeOwned>(frame: &[u8]) -> Result<(Kind, T), ProtocolError> {
    let (kind, value) = decode(frame)?;
    let typed = serde_json::from_value(value)?;
    Ok((kind, typed))
}

/// Validate a JSON payload against the struct for `kind`. Used by Python tests
/// to catch field drift between the mirror and this authoritative definition.
pub fn validate_payload(kind: Kind, payload_json: &str) -> Result<(), ProtocolError> {
    match kind {
        Kind::RolloutRequest => {
            serde_json::from_str::<RolloutRequest>(payload_json)?;
        }
        Kind::TrajectoryRecord => {
            serde_json::from_str::<TrajectoryRecord>(payload_json)?;
        }
        Kind::WeightUpdate => {
            serde_json::from_str::<WeightUpdate>(payload_json)?;
        }
        Kind::RewardOutcome => {
            serde_json::from_str::<RewardOutcome>(payload_json)?;
        }
        Kind::MultimodalTrajectory => {
            serde_json::from_str::<MultimodalTrajectory>(payload_json)?;
        }
        Kind::HealthCheck => {
            serde_json::from_str::<HealthCheck>(payload_json)?;
        }
        Kind::AgentTrajectory => {
            let traj: AgentTrajectory = serde_json::from_str(payload_json)?;
            traj.check_invariants()?;
        }
        Kind::LifecycleCommand => {
            serde_json::from_str::<LifecycleCommand>(payload_json)?;
        }
        Kind::LifecycleAck => {
            serde_json::from_str::<LifecycleAck>(payload_json)?;
        }
        Kind::WeightUpdateAck => {
            serde_json::from_str::<WeightUpdateAck>(payload_json)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> RolloutRequest {
        RolloutRequest {
            request_id: 42,
            policy_version: 7,
            prompt_token_ids: vec![1, 2, 3],
            max_new_tokens: 64,
            n: 4,
            sampling: SamplingParams {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 0,
                seed: None,
                logprobs: true,
                top_logprobs: 0,
            },
            session_id: None,
            model: None,
        }
    }

    #[test]
    fn roundtrip() {
        let frame = encode(Kind::RolloutRequest, &request()).unwrap();
        let (kind, decoded) = decode_typed::<RolloutRequest>(&frame).unwrap();
        assert_eq!(kind, Kind::RolloutRequest);
        assert_eq!(decoded, request());
    }

    #[test]
    fn rejects_version_mismatch() {
        let mut frame = encode(Kind::RolloutRequest, &request()).unwrap();
        frame[4] = 9; // little-endian u16 low byte
        frame[5] = 0;
        assert!(matches!(
            decode(&frame),
            Err(ProtocolError::UnsupportedVersion(9))
        ));
    }

    #[test]
    fn rejects_truncated_frame() {
        let frame = encode(Kind::RolloutRequest, &request()).unwrap();
        assert!(matches!(
            decode(&frame[..HEADER_LEN + 2]),
            Err(ProtocolError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn rejects_unknown_field() {
        let json = r#"{"request_id":1,"policy_version":1,"prompt_token_ids":[],
            "max_new_tokens":1,"n":1,"session_id":null,
            "sampling":{"temperature":1.0,"top_p":1.0,"top_k":0,"seed":null,
            "logprobs":true,"top_logprobs":0},"surprise":true}"#;
        assert!(validate_payload(Kind::RolloutRequest, json).is_err());
    }

    #[test]
    fn multimodal_trajectory_roundtrip() {
        let record = MultimodalTrajectory {
            request_id: 5,
            prompt_len: 4,
            token_ids: vec![20, 21],
            logprobs: vec![-0.2, -0.3],
            policy_versions: vec![7, 7],
            image_token_ids: vec![100, 101],
            image_insert_at: 2,
            image_grid: (2, 1),
            modality_mask: vec![0, 0, 1, 1, 0, 0],
            reward: Some(1.0),
            finish_reason: 0,
            payload_handle: Some(0xABCD),
        };
        let frame = encode(Kind::MultimodalTrajectory, &record).unwrap();
        let (kind, decoded) = decode_typed::<MultimodalTrajectory>(&frame).unwrap();
        assert_eq!(kind, Kind::MultimodalTrajectory);
        assert_eq!(decoded, record);
    }

    #[test]
    fn health_check_roundtrip_and_nonce_validation() {
        let ping = HealthCheck { probe: 42, ok: true };
        let frame = encode(Kind::HealthCheck, &ping).unwrap();
        let (kind, decoded) = decode_typed::<HealthCheck>(&frame).unwrap();
        assert_eq!(kind, Kind::HealthCheck);
        assert_eq!(decoded, ping);
        // A missing nonce is a protocol error, not a default-able field.
        assert!(validate_payload(Kind::HealthCheck, r#"{"ok":true}"#).is_err());
    }

    #[test]
    fn lifecycle_command_and_epoch_guarded_ack_roundtrip() {
        let command = LifecycleCommand {
            request_id: 9,
            epoch: 3,
            action: LifecycleAction::Pause,
            offload: true,
            drain_ms: 1_000,
        };
        let frame = encode(Kind::LifecycleCommand, &command).unwrap();
        let (kind, decoded) = decode_typed::<LifecycleCommand>(&frame).unwrap();
        assert_eq!(kind, Kind::LifecycleCommand);
        assert_eq!(decoded, command);

        let ack = LifecycleAck {
            request_id: 9,
            epoch: 3,
            action: LifecycleAction::Pause,
            ok: true,
            phase: LifecyclePhase::Offloaded,
            drained: 2,
            error: None,
            elapsed_ms: 12,
        };
        assert!(ack.matches(&command));
        let stale = LifecycleAck { epoch: 2, ..ack.clone() };
        assert!(!stale.matches(&command));
        let wrong_action = LifecycleAck {
            action: LifecycleAction::Resume,
            ..ack.clone()
        };
        assert!(!wrong_action.matches(&command));
        let frame = encode(Kind::LifecycleAck, &ack).unwrap();
        let (kind, decoded) = decode_typed::<LifecycleAck>(&frame).unwrap();
        assert_eq!(kind, Kind::LifecycleAck);
        assert_eq!(decoded, ack);
        assert!(validate_payload(Kind::LifecycleAck, &serde_json::to_string(&ack).unwrap()).is_ok());
        assert!(validate_payload(
            Kind::LifecycleCommand,
            r#"{"request_id":1,"epoch":1,"action":"pause","surprise":true}"#
        )
        .is_err());
    }

    #[test]
    fn weight_update_ack_round_trips_with_phase_timings() {
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
        assert_eq!(Kind::from_str("WeightUpdateAck"), Some(Kind::WeightUpdateAck));
        assert_eq!(Kind::from_u8(10), Some(Kind::WeightUpdateAck));
        let frame = encode(Kind::WeightUpdateAck, &ack).unwrap();
        let (kind, decoded) = decode_typed::<WeightUpdateAck>(&frame).unwrap();
        assert_eq!(kind, Kind::WeightUpdateAck);
        assert_eq!(decoded, ack);
        let json = serde_json::to_string(&ack).unwrap();
        assert!(validate_payload(Kind::WeightUpdateAck, &json).is_ok());
        assert!(validate_payload(
            Kind::WeightUpdateAck,
            r#"{"request_id":1,"ok":true,"surprise":true}"#
        )
        .is_err());
    }

    fn agent_trajectory() -> AgentTrajectory {
        // prompt: [10, 11]; assistant turn 0: [20]; tool obs: [30];
        // assistant turn 1: [21, 22]. Every response-aligned vector has len 4.
        AgentTrajectory {
            request_id: 9,
            policy_version: 5,
            prompt_len: 2,
            token_ids: vec![10, 11, 20, 30, 21, 22],
            logprobs: vec![-0.1, 0.0, -0.2, -0.3],
            top_logprobs: vec![],
            policy_versions: vec![5, 5, 6, 6],
            token_roles: vec![0, 0, 1, 2, 1, 1],
            turn_ids: vec![0, 0, 0, 1, 1, 1],
            loss_mask: vec![0, 0, 1, 0, 1, 1],
            sampling_logprobs: vec![],
            tool_calls: vec![AgentToolCall {
                id: "c1".to_string(),
                name: "calculator".to_string(),
                arguments_json: r#"{"expr":"1+2"}"#.to_string(),
                turn: 0,
                ok: true,
                elapsed_ms: 7,
            }],
            reward: Some(1.0),
            step_rewards: vec![0.0, 1.0],
            finish_reason: 0,
            payload_handle: None,
        }
    }

    #[test]
    fn agent_trajectory_roundtrip_and_invariants() {
        let traj = agent_trajectory();
        traj.check_invariants().unwrap();
        let frame = encode(Kind::AgentTrajectory, &traj).unwrap();
        let (kind, decoded) = decode_typed::<AgentTrajectory>(&frame).unwrap();
        assert_eq!(kind, Kind::AgentTrajectory);
        assert_eq!(decoded, traj);
        let json = serde_json::to_string(&traj).unwrap();
        validate_payload(Kind::AgentTrajectory, &json).unwrap();
    }

    #[test]
    fn agent_trajectory_rejects_tool_tokens_in_loss() {
        let mut traj = agent_trajectory();
        traj.loss_mask[3] = 1; // tool observation must never carry gradient
        assert!(traj.check_invariants().is_err());
        let json = serde_json::to_string(&traj).unwrap();
        assert!(validate_payload(Kind::AgentTrajectory, &json).is_err());
    }

    #[test]
    fn agent_trajectory_rejects_misaligned_response_vectors() {
        let mut traj = agent_trajectory();
        traj.policy_versions.pop();
        assert!(traj.check_invariants().is_err());
        let json = serde_json::to_string(&traj).unwrap();
        assert!(validate_payload(Kind::AgentTrajectory, &json).is_err());
    }

    #[test]
    fn trajectory_keeps_per_token_versions() {
        let traj = TrajectoryRecord {
            request_id: 1,
            prompt_len: 10,
            token_ids: vec![11, 12],
            logprobs: vec![-0.1, -0.2],
            top_logprobs: vec![vec![(11, -0.1)], vec![(12, -0.2)]],
            policy_versions: vec![6, 7],
            reward: Some(1.0),
            finish_reason: 0,
            payload_handle: None,
            sampling_logprobs: vec![-0.1, -0.2],
        };
        let frame = encode(Kind::TrajectoryRecord, &traj).unwrap();
        let (_, decoded) = decode_typed::<TrajectoryRecord>(&frame).unwrap();
        assert_eq!(decoded, traj);
    }
}
