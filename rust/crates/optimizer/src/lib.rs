//! AdamW inversion for delta detection without a stored weight baseline.
//!
//! AReaL's DTE path avoids pinning a full fp32 copy of every weight just to
//! compute a delta. After an optimizer step it reconstructs the *pre-step*
//! master weights from the post-step weights and the post-step Adam moments:
//!
//! ```text
//!   m_t = beta1*m_(t-1) + (1-beta1)*g
//!   v_t = beta2*v_(t-1) + (1-beta2)*g^2
//!   theta_t = theta_(t-1) * (1 - lr*wd) - (lr/bc1) * m_t / (sqrt(v_t)/sqrt(bc2) + eps)
//!
//!   => theta_(t-1) = (theta_t + (lr/bc1) * m_t / (sqrt(v_t)/sqrt(bc2) + eps)) / (1 - lr*wd)
//! ```
//!
//! Only the LR **consumed by the completed step** may be used; a scheduler's
//! next-step LR silently corrupts the reconstruction. A small value
//! fingerprint (not a snapshot) plus the recorded sync step prevents replaying
//! moments for a tensor whose visible payload did not actually move.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InversionError {
    LengthMismatch { theta: usize, m: usize, v: usize },
    InvalidStep(u64),
    InvalidLr(f64),
    InvalidWeightDecay(f64),
    InvalidBeta { beta1: f64, beta2: f64 },
    InvalidEps(f64),
    NonPositiveDecayFactor(f64),
    UnsupportedDtype(&'static str),
}

impl std::fmt::Display for InversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InversionError::LengthMismatch { theta, m, v } => {
                write!(f, "theta/m/v length mismatch: {theta}/{m}/{v}")
            }
            InversionError::InvalidStep(s) => write!(f, "step must be >= 1, got {s}"),
            InversionError::InvalidLr(lr) => write!(f, "invalid learning rate: {lr}"),
            InversionError::InvalidWeightDecay(wd) => {
                write!(f, "invalid weight decay: {wd}")
            }
            InversionError::InvalidBeta { beta1, beta2 } => {
                write!(f, "betas must be in [0, 1): {beta1}/{beta2}")
            }
            InversionError::InvalidEps(eps) => write!(f, "eps must be > 0, got {eps}"),
            InversionError::NonPositiveDecayFactor(value) => {
                write!(f, "1 - lr*weight_decay must be positive, got {value}")
            }
            InversionError::UnsupportedDtype(name) => write!(f, "unsupported dtype: {name}"),
        }
    }
}

impl std::error::Error for InversionError {}

/// Hyperparameters of the **completed** optimizer step. `step` is the
/// post-step count (PyTorch's `state["step"]`), and `lr` is the LR actually
/// used for that step, recorded at step time rather than read from a schedule.
#[derive(Debug, Clone, Copy)]
pub struct AdamwParams {
    pub step: u64,
    pub lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl AdamwParams {
    pub fn validate(&self) -> Result<(), InversionError> {
        if self.step < 1 {
            return Err(InversionError::InvalidStep(self.step));
        }
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err(InversionError::InvalidLr(self.lr));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(InversionError::InvalidWeightDecay(self.weight_decay));
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            return Err(InversionError::InvalidBeta {
                beta1: self.beta1,
                beta2: self.beta2,
            });
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(InversionError::InvalidEps(self.eps));
        }
        Ok(())
    }

    fn bias_correction(&self) -> Result<(f64, f64), InversionError> {
        let bc1 = 1.0 - self.beta1.powf(self.step as f64);
        let bc2 = 1.0 - self.beta2.powf(self.step as f64);
        if bc1 <= 0.0 || bc2 <= 0.0 {
            return Err(InversionError::InvalidBeta {
                beta1: self.beta1,
                beta2: self.beta2,
            });
        }
        Ok((bc1, bc2))
    }

    fn decay_factor(&self) -> Result<f64, InversionError> {
        let factor = 1.0 - self.lr * self.weight_decay;
        if !factor.is_finite() || factor <= 0.0 {
            return Err(InversionError::NonPositiveDecayFactor(factor));
        }
        Ok(factor)
    }
}

/// Apply one AdamW step (matches PyTorch's decoupled AdamW). Kept as the
/// reference forward direction so inversion can be verified end to end.
pub fn adamw_step_f32(
    theta: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grad: &[f32],
    params: AdamwParams,
) -> Result<(), InversionError> {
    params.validate()?;
    if theta.len() != m.len() || theta.len() != v.len() || theta.len() != grad.len() {
        return Err(InversionError::LengthMismatch {
            theta: theta.len(),
            m: m.len(),
            v: v.len(),
        });
    }
    let (bc1, bc2) = params.bias_correction()?;
    let decay = params.decay_factor()?;
    let step_size = params.lr / bc1;
    let bc2_sqrt = bc2.sqrt();
    for i in 0..theta.len() {
        m[i] = (params.beta1 * m[i] as f64 + (1.0 - params.beta1) * grad[i] as f64) as f32;
        v[i] = (params.beta2 * v[i] as f64 + (1.0 - params.beta2) * grad[i] as f64 * grad[i] as f64)
            as f32;
        let denom = v[i].max(0.0).sqrt() as f64 / bc2_sqrt + params.eps;
        theta[i] = (theta[i] as f64 * decay - step_size * m[i] as f64 / denom) as f32;
    }
    Ok(())
}

/// Reconstruct pre-step fp32 master weights from post-step weights and the
/// **post-step** Adam moments. This is the no-baseline delta primitive.
pub fn invert_adamw_f32(
    theta: &[f32],
    m: &[f32],
    v: &[f32],
    params: AdamwParams,
) -> Result<Vec<f32>, InversionError> {
    params.validate()?;
    if theta.len() != m.len() || theta.len() != v.len() {
        return Err(InversionError::LengthMismatch {
            theta: theta.len(),
            m: m.len(),
            v: v.len(),
        });
    }
    let (bc1, bc2) = params.bias_correction()?;
    let inv_decay = 1.0 / params.decay_factor()?;
    let step_size = params.lr / bc1;
    let bc2_sqrt = bc2.sqrt();
    let mut out = Vec::with_capacity(theta.len());
    for i in 0..theta.len() {
        // eps > 0 guarantees denom > 0; clamp a negative sqrt from NaN states.
        let denom = v[i].max(0.0).sqrt() as f64 / bc2_sqrt + params.eps;
        let corrected = step_size * m[i] as f64 / denom;
        out.push(((theta[i] as f64 + corrected) * inv_decay) as f32);
    }
    Ok(out)
}

/// Weight storage dtype. Only lossless-from-fp32 targets are supported: fp32
/// master weights and bf16 visible payloads are the production path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDtype {
    Float32,
    BFloat16,
}

impl StorageDtype {
    pub fn from_str(name: &str) -> Result<Self, InversionError> {
        match name.trim().to_ascii_lowercase().replace('_', "").as_str() {
            "float32" | "f32" => Ok(StorageDtype::Float32),
            "bfloat16" | "bf16" => Ok(StorageDtype::BFloat16),
            other => Err(InversionError::UnsupportedDtype(Box::leak(
                other.to_string().into_boxed_str(),
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            StorageDtype::Float32 => "float32",
            StorageDtype::BFloat16 => "bfloat16",
        }
    }

    pub fn itemsize(self) -> usize {
        match self {
            StorageDtype::Float32 => 4,
            StorageDtype::BFloat16 => 2,
        }
    }
}

/// Cast fp32 values to the storage dtype's little-endian byte representation.
/// bf16 uses round-to-nearest-even, matching TensorPlay / PyTorch casts.
pub fn cast_f32_to_bytes(values: &[f32], dtype: StorageDtype) -> Vec<u8> {
    match dtype {
        StorageDtype::Float32 => {
            let mut out = Vec::with_capacity(values.len() * 4);
            for &value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out
        }
        StorageDtype::BFloat16 => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for &value in values {
                let bits = value.to_bits();
                let lsb = (bits >> 16) & 1;
                let rounded = bits.wrapping_add(0x7fff + lsb);
                out.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
            }
            out
        }
    }
}

/// Decode storage bytes into fp32 (used by adapters and tests).
pub fn bytes_to_f32(data: &[u8], dtype: StorageDtype) -> Result<Vec<f32>, InversionError> {
    let size = dtype.itemsize();
    if data.len() % size != 0 {
        return Err(InversionError::UnsupportedDtype("misaligned byte payload"));
    }
    Ok(match dtype {
        StorageDtype::Float32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        StorageDtype::BFloat16 => data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
    })
}

/// Reconstruct the pre-step weights and diff at **element** granularity.
///
/// Inversion is accurate to a few ulp but not bit-exact (the forward step was
/// rounded in fp32), so a byte-level diff can miss a high-order byte near a
/// power-of-two boundary. Emitting every byte of a changed element removes that
/// failure mode while keeping the delta sparse. Adjacent changed elements are
/// merged into a single range.
pub fn invert_and_diff_bytes(
    theta: &[u8],
    m: &[u8],
    v: &[u8],
    params: AdamwParams,
    visible: StorageDtype,
    moments: StorageDtype,
) -> Result<Vec<(usize, Vec<u8>)>, InversionError> {
    let current = bytes_to_f32(theta, visible)?;
    let m_f32 = bytes_to_f32(m, moments)?;
    let v_f32 = bytes_to_f32(v, moments)?;
    let pre = invert_adamw_f32(&current, &m_f32, &v_f32, params)?;
    let size = visible.itemsize();
    let mut ranges: Vec<(usize, Vec<u8>)> = Vec::new();
    for i in 0..current.len() {
        if pre[i].to_bits() == current[i].to_bits() {
            continue;
        }
        let start = i * size;
        let chunk = &theta[start..start + size];
        if let Some((last_start, last)) = ranges.last_mut() {
            if *last_start + last.len() == start {
                last.extend_from_slice(chunk);
                continue;
            }
        }
        ranges.push((start, chunk.to_vec()));
    }
    Ok(ranges)
}

/// Compact value fingerprint of a visible payload — not a stored snapshot.
/// Samples 256 bins so checking a large tensor stays cheap; the length and
/// dtype are mixed in so layout changes cannot collide silently.
pub fn fingerprint(data: &[u8], dtype: StorageDtype) -> u64 {
    let mut hash = 0xcbf29ce4_8422_2325u64 ^ (dtype as u64);
    hash = hash
        .wrapping_mul(0x100_0000_01b3)
        .wrapping_add(data.len() as u64);
    let bins = data.len().min(256);
    if bins == 0 {
        return hash;
    }
    let width = data.len() / bins;
    if width == 0 {
        for &byte in data {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    } else {
        for bin in 0..bins {
            let mut sum = 0u64;
            for &byte in &data[bin * width..(bin + 1) * width] {
                sum = sum.wrapping_add(byte as u64);
            }
            hash = (hash ^ sum).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

#[derive(Debug, Clone, Copy)]
struct SyncRecord {
    step: u64,
    fp: u64,
}

/// Prevents replaying an already-synced update. Mirrors AReaL's detector:
/// a tensor is a candidate only when the visible payload changed *and* the
/// step advanced past the last successful sync.
#[derive(Debug, Default)]
pub struct AdamwInversionDetector {
    synced: HashMap<String, SyncRecord>,
}

impl AdamwInversionDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn should_detect(
        &self,
        name: &str,
        visible: &[u8],
        dtype: StorageDtype,
        step: u64,
    ) -> bool {
        let fp = fingerprint(visible, dtype);
        match self.synced.get(name) {
            None => true,
            Some(record) => step > record.step && fp != record.fp,
        }
    }

    /// Record a successful sync so the same (step, payload) is not detected
    /// again on the next checkpoint/weight-update cycle.
    pub fn mark_synced(
        &mut self,
        name: &str,
        visible: &[u8],
        dtype: StorageDtype,
        step: u64,
    ) {
        self.synced.insert(
            name.to_string(),
            SyncRecord {
                step,
                fp: fingerprint(visible, dtype),
            },
        );
    }

    pub fn needs_sync(&self, name: &str) -> bool {
        !self.synced.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(step: u64) -> AdamwParams {
        AdamwParams {
            step,
            lr: 1e-3,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    #[test]
    fn inversion_recovers_pre_step_weights() {
        let n = 512;
        let prev: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.01).collect();
        let grad: Vec<f32> = (0..n).map(|i| ((i * 17) % 101) as f32 * 0.001 - 0.05).collect();
        let mut theta = prev.clone();
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let p = params(1);
        adamw_step_f32(&mut theta, &mut m, &mut v, &grad, p).unwrap();
        let recovered = invert_adamw_f32(&theta, &m, &v, p).unwrap();
        for (a, b) in prev.iter().zip(&recovered) {
            assert!((a - b).abs() <= 1e-4 + 1e-3 * a.abs(), "{a} != {b}");
        }
    }

    #[test]
    fn inversion_handles_later_steps_and_zero_gradient_decay() {
        let n = 128;
        let prev: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let grad = vec![0.0f32; n]; // only decoupled decay changes the weights
        let mut theta = prev.clone();
        let mut m = vec![0.1f32; n];
        let mut v = vec![0.01f32; n];
        let p = params(100);
        adamw_step_f32(&mut theta, &mut m, &mut v, &grad, p).unwrap();
        let recovered = invert_adamw_f32(&theta, &m, &v, p).unwrap();
        for (a, b) in prev.iter().zip(&recovered) {
            assert!((a - b).abs() <= 1e-3 + 1e-3 * a.abs(), "{a} != {b}");
        }
    }

    #[test]
    fn inversion_rejects_bad_hyperparameters_and_shapes() {
        let theta = [1.0f32];
        let m = [0.1f32];
        let v = [0.01f32];
        assert_eq!(
            invert_adamw_f32(&theta, &m, &v, params(0)),
            Err(InversionError::InvalidStep(0))
        );
        let mut bad = params(1);
        bad.lr = -0.1;
        assert!(matches!(
            invert_adamw_f32(&theta, &m, &v, bad),
            Err(InversionError::InvalidLr(_))
        ));
        let mut bad = params(1);
        bad.beta1 = 1.0;
        assert!(matches!(
            invert_adamw_f32(&theta, &m, &v, bad),
            Err(InversionError::InvalidBeta { .. })
        ));
        let mut bad = params(1);
        bad.lr = 200.0;
        bad.weight_decay = 0.01;
        assert!(matches!(
            invert_adamw_f32(&theta, &m, &v, bad),
            Err(InversionError::NonPositiveDecayFactor(_))
        ));
        assert!(matches!(
            invert_adamw_f32(&theta, &[], &v, params(1)),
            Err(InversionError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn bf16_cast_is_round_to_nearest_even() {
        let values = [1.0f32, 1.004, 1.005, -2.5, 0.0];
        let bytes = cast_f32_to_bytes(&values, StorageDtype::BFloat16);
        let decoded = bytes_to_f32(&bytes, StorageDtype::BFloat16).unwrap();
        // 1.005 sits near a bf16 boundary; the exact rounded value is checked
        // against a bit-level reference to avoid relying on a tolerance.
        for (value, round) in values.iter().zip(&decoded) {
            let bits = value.to_bits();
            let bf = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u32;
            let expected = f32::from_bits(bf << 16);
            assert_eq!(*round, expected);
        }
        assert_eq!(bytes.len(), values.len() * 2);
    }

    #[test]
    fn conservative_diff_covers_high_order_bytes_near_boundaries() {
        let n = 256;
        let prev: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.25).collect();
        let grad: Vec<f32> = (0..n).map(|i| ((i * 17) % 101) as f32 * 0.005 - 0.25).collect();
        let mut theta = prev.clone();
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let p = params(1);
        adamw_step_f32(&mut theta, &mut m, &mut v, &grad, p).unwrap();
        let theta_bytes = cast_f32_to_bytes(&theta, StorageDtype::Float32);
        let ranges = invert_and_diff_bytes(
            &theta_bytes,
            &cast_f32_to_bytes(&m, StorageDtype::Float32),
            &cast_f32_to_bytes(&v, StorageDtype::Float32),
            p,
            StorageDtype::Float32,
            StorageDtype::Float32,
        )
        .unwrap();
        assert!(!ranges.is_empty());
        // Applying the ranges to the true pre-step bytes must reproduce the
        // post-step bytes exactly, including boundary-nearing values.
        let mut restored = cast_f32_to_bytes(&prev, StorageDtype::Float32);
        for (offset, chunk) in &ranges {
            restored[*offset..*offset + chunk.len()].copy_from_slice(chunk);
        }
        assert_eq!(restored, theta_bytes);
    }

    #[test]
    fn detector_prevents_replay_of_an_unchanged_payload() {
        let mut detector = AdamwInversionDetector::new();
        let visible = [0u8; 64];
        assert!(detector.should_detect("w", &visible, StorageDtype::Float32, 1));
        detector.mark_synced("w", &visible, StorageDtype::Float32, 1);
        assert!(!detector.should_detect("w", &visible, StorageDtype::Float32, 1));
        assert!(!detector.should_detect("w", &visible, StorageDtype::Float32, 0));
        let changed = [1u8; 64];
        assert!(detector.should_detect("w", &changed, StorageDtype::Float32, 2));
        assert!(!detector.needs_sync("w"));
    }
}
