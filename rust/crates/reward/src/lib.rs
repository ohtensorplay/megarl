//! Production reward verification for megaRL.
//!
//! Math answers are checked by symbolic normalization plus numeric evaluation
//! at deterministic sample points; code is executed in a resource-limited
//! process group; both are dispatched through a bounded pool with retries.

pub mod graders;
pub mod math;
pub mod pool;
pub mod sandbox;
pub mod seccomp;

pub use graders::{
    grade_benchmark, grade_benchmark_json, Benchmark, BenchmarkSpec, CompareMode, GradeResult,
    TestCase,
};
pub use math::{extract_answer, math_equivalent, normalize_answer, numeric_value, verify_math};
pub use pool::{
    route_domain, verify_batch_json, verify_one, Domain, VerifyOutcome, VerifyPool, VerifyRequest,
    PoolStats,
};
pub use sandbox::{run_sandboxed, SandboxConfig, SandboxError, SandboxResult};
pub use seccomp::{FirejailMode, SeccompFilter, SeccompMode};
