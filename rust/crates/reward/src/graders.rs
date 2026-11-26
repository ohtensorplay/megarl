//! Domain-specific code graders.
//!
//! A benchmark ships a set of test cases (arguments + expected output). The
//! candidate solution is executed **inside the existing resource-limited
//! sandbox** against every case, and the reward is the fraction of cases that
//! pass. This replaces a single pass/fail exit-code check with the per-test
//! granularity HumanEval / MBPP / APPS-style benchmarks need.
//!
//! The harness is generated here, never supplied by the model. Candidate code
//! is fed on stdin and `exec`'d in a fresh interpreter with cleared
//! environment, rlimits, a process group and a hard wall-clock timeout (see
//! [`crate::sandbox`]); each test additionally has a `signal.alarm` deadline so
//! a single hanging case cannot consume the whole budget.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::sandbox::{run_sandboxed, SandboxConfig};
use crate::seccomp::{sandbox_label, FirejailMode, SeccompMode};

/// Well-known benchmark profiles. They differ only in how the observed result
/// is obtained: HumanEval/MBPP compare the function's return value, APPS
/// compares stdout, and Custom lets the spec decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Benchmark {
    HumanEval,
    Mbpp,
    Apps,
    Custom,
}

impl Benchmark {
    pub fn from_str(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
            "human_eval" | "humaneval" => Benchmark::HumanEval,
            "mbpp" => Benchmark::Mbpp,
            "apps" => Benchmark::Apps,
            _ => Benchmark::Custom,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Benchmark::HumanEval => "human_eval",
            Benchmark::Mbpp => "mbpp",
            Benchmark::Apps => "apps",
            Benchmark::Custom => "custom",
        }
    }

    fn default_mode(self) -> CompareMode {
        match self {
            Benchmark::HumanEval | Benchmark::Mbpp => CompareMode::Return,
            Benchmark::Apps | Benchmark::Custom => CompareMode::Stdout,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareMode {
    /// Compare the function's return value (JSON-serialized).
    Return,
    /// Compare what the function prints to stdout.
    Stdout,
}

/// One grading case. `args` is the argument list passed to the entry point;
/// `expected` is the expected return value (JSON) or stdout text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    #[serde(default)]
    pub args: Vec<Value>,
    pub expected: String,
}

/// A benchmark grading specification, normally stored as JSON next to the
/// dataset. `benchmark` selects the profile, `mode` optionally overrides its
/// comparison mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSpec {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub benchmark: String,
    #[serde(default)]
    pub mode: Option<CompareMode>,
    /// Name of the candidate function to call.
    #[serde(default)]
    pub entry_point: String,
    pub tests: Vec<TestCase>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_memory_mb")]
    pub memory_mb: u64,
    /// Syscall-level sandbox layer. Defaults to the seccomp blacklist.
    #[serde(default)]
    pub seccomp: SeccompMode,
    /// Optional Firejail wrapper. Defaults to auto-detect.
    #[serde(default)]
    pub firejail: FirejailMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firejail_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firejail_bin: Option<String>,
}

fn default_timeout_ms() -> u64 {
    5_000
}
fn default_memory_mb() -> u64 {
    512
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradeResult {
    pub reward: f32,
    pub passed: usize,
    pub total: usize,
    pub verifier: String,
    pub benchmark: String,
    #[serde(default)]
    pub per_test: Vec<bool>,
    /// Isolation layer actually applied to the grader subprocess.
    #[serde(default)]
    pub sandbox: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn harness(spec: &BenchmarkSpec) -> String {
    let profile = Benchmark::from_str(&spec.benchmark);
    let mode = spec.mode.unwrap_or_else(|| profile.default_mode());
    let embedded = serde_json::to_string(&(
        &spec.entry_point,
        &spec.tests,
        mode_as_str(mode),
        (spec.timeout_ms / 1000).max(1),
    ))
    .unwrap_or_else(|_| "[\"\",[],\"stdout\",1]".to_string());
    format!(
        r#"import sys, io, json, signal, contextlib
ENTRY, TESTS, MODE, PER_TEST_SECS = json.loads({embedded:?})
src = sys.stdin.read()
ns = {{}}
exec(compile(src, "<candidate>", "exec"), ns)
fn = ns.get(ENTRY)
results = []
err = None
if not callable(fn):
    err = "entry_point is not callable: " + str(ENTRY)
else:
    def _alarm(*_):
        raise TimeoutError("per-test timeout")
    signal.signal(signal.SIGALRM, _alarm)
    for case in TESTS:
        buf = io.StringIO()
        ok = False
        try:
            signal.setitimer(signal.ITIMER_REAL, PER_TEST_SECS)
            with contextlib.redirect_stdout(buf):
                value = fn(*case["args"])
            signal.setitimer(signal.ITIMER_REAL, 0)
            if MODE == "return":
                observed = json.dumps(value, sort_keys=True, default=str).strip()
            else:
                observed = buf.getvalue().strip()
            ok = observed == str(case["expected"]).strip()
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            ok = False
        results.append(ok)
sys.stdout.write("__GRADE__" + json.dumps({{"passed": results, "error": err}}))
"#
    )
}

fn mode_as_str(mode: CompareMode) -> &'static str {
    match mode {
        CompareMode::Return => "return",
        CompareMode::Stdout => "stdout",
    }
}

/// Grade a candidate solution against a benchmark spec. Returns a fractional
/// reward rather than a boolean; a malformed spec or a sandbox failure yields
/// a zero reward with a structured error.
pub fn grade_benchmark(spec: &BenchmarkSpec, candidate: &str) -> GradeResult {
    let total = spec.tests.len();
    let profile = Benchmark::from_str(&spec.benchmark);
    let mut result = GradeResult {
        reward: 0.0,
        passed: 0,
        total,
        verifier: "benchmark_sandbox".into(),
        benchmark: profile.as_str().to_string(),
        per_test: Vec::new(),
        sandbox: sandbox_label(false, spec.seccomp).to_string(),
        error: None,
    };
    if total == 0 {
        result.error = Some("benchmark needs at least one test case".into());
        return result;
    }
    if spec.entry_point.trim().is_empty() {
        result.error = Some("entry_point is required".into());
        return result;
    }
    let config = SandboxConfig {
        timeout_ms: spec.timeout_ms,
        memory_mb: spec.memory_mb,
        seccomp: spec.seccomp,
        firejail: spec.firejail,
        firejail_profile: spec.firejail_profile.clone(),
        firejail_bin: spec.firejail_bin.clone(),
        ..SandboxConfig::default()
    };
    let args = vec!["-I".to_string(), "-c".to_string(), harness(spec)];
    match run_sandboxed("python3", &args, candidate.as_bytes(), &config) {
        Ok(run) => {
            result.sandbox = run.sandbox.clone();
            if run.timed_out {
                result.error = Some("grader timed out".into());
                result.per_test = vec![false; total];
                return result;
            }
            let text = String::from_utf8_lossy(&run.stdout);
            let Some(line) = text.rsplit("__GRADE__").next().filter(|_| text.contains("__GRADE__")) else {
                result.error = Some("grader produced no result marker".into());
                result.per_test = vec![false; total];
                return result;
            };
            match serde_json::from_str::<Value>(line) {
                Ok(value) => {
                    let per_test: Vec<bool> = value["passed"]
                        .as_array()
                        .map(|a| a.iter().map(|v| v.as_bool().unwrap_or(false)).collect())
                        .unwrap_or_default();
                    result.error = value["error"].as_str().map(str::to_string);
                    result.passed = per_test.iter().filter(|&&p| p).count();
                    result.per_test = per_test;
                    result.reward = result.passed as f32 / total as f32;
                }
                Err(e) => {
                    result.error = Some(format!("malformed grader output: {e}"));
                    result.per_test = vec![false; total];
                }
            }
        }
        Err(e) => {
            result.error = Some(e.to_string());
            result.per_test = vec![false; total];
        }
    }
    result
}

/// JSON-in/JSON-out helper for PyO3 and the HTTP service.
pub fn grade_benchmark_json(spec_json: &str, candidate: &str) -> Result<String, serde_json::Error> {
    let spec: BenchmarkSpec = serde_json::from_str(spec_json)?;
    serde_json::to_string(&grade_benchmark(&spec, candidate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(benchmark: &str, mode: Option<CompareMode>, expected: &str) -> BenchmarkSpec {
        BenchmarkSpec {
            name: "unit".into(),
            benchmark: benchmark.into(),
            mode,
            entry_point: "add".into(),
            tests: vec![
                TestCase {
                    args: vec![json!(1), json!(2)],
                    expected: expected.into(),
                },
                TestCase {
                    args: vec![json!(10), json!(5)],
                    expected: expected.into(),
                },
            ],
            timeout_ms: 5_000,
            memory_mb: 512,
            seccomp: SeccompMode::Blacklist,
            firejail: FirejailMode::Off,
            firejail_profile: None,
            firejail_bin: None,
        }
    }

    #[test]
    fn profiles_map_to_comparison_modes() {
        assert_eq!(Benchmark::from_str("HumanEval"), Benchmark::HumanEval);
        assert_eq!(Benchmark::from_str("mbpp"), Benchmark::Mbpp);
        assert_eq!(Benchmark::from_str("APPS"), Benchmark::Apps);
        assert_eq!(Benchmark::from_str("other"), Benchmark::Custom);
        assert_eq!(Benchmark::HumanEval.default_mode(), CompareMode::Return);
        assert_eq!(Benchmark::Apps.default_mode(), CompareMode::Stdout);
    }

    #[test]
    fn grades_correct_and_incorrect_return_values() {
        let good = "def add(a, b):\n    return a + b\n";
        let bad = "def add(a, b):\n    return a - b\n";
        let mut correct = spec("human_eval", None, "3");
        correct.tests[1].expected = "15".into(); // 10 + 5 = 15
        let result = grade_benchmark(&correct, good);
        assert_eq!(result.reward, 1.0);
        assert_eq!(result.passed, 2);
        assert_eq!(result.per_test, vec![true, true]);
        let partial = grade_benchmark(&correct, bad);
        assert_eq!(partial.passed, 0);
        assert_eq!(partial.reward, 0.0);
    }

    #[test]
    fn stdout_mode_grades_printed_output() {
        let candidate = "def add(a, b):\n    print(a * b)\n";
        let result = grade_benchmark(&spec("apps", None, "2"), candidate);
        // 1*2=2 (pass), 10*5=50 (fail) -> half credit.
        assert_eq!(result.passed, 1);
        assert_eq!(result.per_test, vec![true, false]);
        assert_eq!(result.reward, 0.5);
    }

    #[test]
    fn invalid_spec_yields_zero_with_error() {
        let mut s = spec("custom", None, "3");
        s.tests.clear();
        let result = grade_benchmark(&s, "def add(a, b): return a + b");
        assert_eq!(result.reward, 0.0);
        assert!(result.error.is_some());
    }

    #[test]
    fn grader_handles_a_hanging_candidate() {
        let mut s = spec("custom", Some(CompareMode::Stdout), "3");
        s.timeout_ms = 1_000;
        let hang = "def add(a, b):\n    while True: pass\n";
        let result = grade_benchmark(&s, hang);
        assert_eq!(result.reward, 0.0);
        assert!(result.error.is_some() || result.passed == 0);
    }

    #[test]
    fn seccomp_blocks_network_from_candidate_code() {
        // Return-mode graders compare JSON-serialized values, so a string
        // result must carry its JSON quotes in `expected`.
        let mut s = spec("custom", Some(CompareMode::Return), "\"blocked\"");
        s.entry_point = "probe".into();
        s.tests = vec![TestCase {
            args: vec![],
            expected: "\"blocked\"".into(),
        }];
        let candidate = r#"import ctypes
def probe():
    libc = ctypes.CDLL(None, use_errno=True)
    return 'blocked' if libc.socket(2, 1, 0) < 0 and ctypes.get_errno() == 1 else 'allowed'
"#;
        let result = grade_benchmark(&s, candidate);
        assert_eq!(result.reward, 1.0, "{result:?}");
        assert_eq!(result.sandbox, "seccomp_blacklist");
    }

    #[test]
    fn required_firejail_failure_surfaces_as_a_grade_error() {
        let mut s = spec("custom", None, "3");
        s.firejail = FirejailMode::Required;
        s.firejail_bin = Some("/nonexistent/megarl-firejail".into());
        let result = grade_benchmark(&s, "def add(a, b): return a + b");
        assert_eq!(result.reward, 0.0);
        assert!(result.error.is_some());
    }
}
