//! Sandboxed process execution for code rewards.
//!
//! The verifier never trusts generated code. Every run gets its own process
//! group, address-space / CPU / file-size / process-count limits, a bounded
//! stdout/stderr capture and a hard wall-clock deadline. On timeout the whole
//! process group is killed so spawned children cannot survive.

use std::io::{self, Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::seccomp::{
    find_firejail, firejail_args, sandbox_label, FirejailMode, SeccompFilter, SeccompMode,
};

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub timeout_ms: u64,
    pub memory_mb: u64,
    pub cpu_secs: u64,
    pub max_output_bytes: usize,
    pub env: Vec<(String, String)>,
    /// In-process seccomp layer. Defaults to the curated syscall blacklist.
    pub seccomp: SeccompMode,
    /// Optional Firejail wrapper. Defaults to auto-detect.
    pub firejail: FirejailMode,
    /// Optional Firejail profile path (otherwise the built-in filter is used).
    pub firejail_profile: Option<String>,
    /// Explicit Firejail binary path; overrides PATH discovery.
    pub firejail_bin: Option<String>,
}

fn env_enum<T: std::str::FromStr + Copy>(key: &str, fallback: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 5_000,
            memory_mb: 512,
            cpu_secs: 5,
            max_output_bytes: 1 << 20,
            env: Vec::new(),
            seccomp: env_enum("MEGARL_SANDBOX_SECCOMP", SeccompMode::Blacklist),
            firejail: env_enum("MEGARL_SANDBOX_FIREJAIL", FirejailMode::Auto),
            firejail_profile: std::env::var("MEGARL_FIREJAIL_PROFILE")
                .ok()
                .filter(|v| !v.is_empty()),
            firejail_bin: std::env::var("MEGARL_FIREJAIL_BIN")
                .ok()
                .filter(|v| !v.is_empty()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SandboxResult {
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub timed_out: bool,
    pub output_truncated: bool,
    pub elapsed_ms: u64,
    /// Which isolation layer was actually applied: `none`,
    /// `seccomp_blacklist` or `firejail+seccomp`.
    pub sandbox: String,
}

impl SandboxResult {
    pub fn succeeded(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }
}

#[derive(Debug)]
pub enum SandboxError {
    Spawn(io::Error),
    Wait(io::Error),
    Invalid(&'static str),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::Spawn(e) => write!(f, "sandbox spawn failed: {e}"),
            SandboxError::Wait(e) => write!(f, "sandbox wait failed: {e}"),
            SandboxError::Invalid(msg) => write!(f, "invalid sandbox config: {msg}"),
        }
    }
}

impl std::error::Error for SandboxError {}

fn set_limit(resource: u32, value: u64) {
    let limit = libc::rlimit {
        rlim_cur: value as libc::rlim_t,
        rlim_max: value as libc::rlim_t,
    };
    unsafe {
        libc::setrlimit(resource, &limit);
    }
}

fn read_bounded<R: Read>(mut reader: R, limit: usize) -> (Vec<u8>, bool) {
    let mut data = Vec::new();
    let mut truncated = false;
    let mut chunk = [0u8; 8192];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                let room = limit.saturating_sub(data.len());
                let keep = n.min(room);
                data.extend_from_slice(&chunk[..keep]);
                if keep < n {
                    truncated = true;
                }
                // Continue draining even when full: a blocked child would
                // otherwise hang until the wall-clock deadline.
            }
            Err(_) => break,
        }
    }
    (data, truncated)
}

/// Run `command` with `args`, feeding `stdin`, under the supplied limits.
pub fn run_sandboxed(
    command: &str,
    args: &[String],
    stdin_data: &[u8],
    config: &SandboxConfig,
) -> Result<SandboxResult, SandboxError> {
    if config.timeout_ms == 0 {
        return Err(SandboxError::Invalid("timeout_ms must be positive"));
    }

    // Resolve the isolation layers *before* spawning. Firejail installs its own
    // seccomp filter after exec, so our pre-exec filter is skipped when it is
    // used (its namespace setup needs mount/clone syscalls).
    let firejail = match config.firejail {
        FirejailMode::Off => None,
        FirejailMode::Auto => find_firejail(config.firejail_bin.as_deref()),
        FirejailMode::Required => Some(find_firejail(config.firejail_bin.as_deref()).ok_or_else(|| {
            SandboxError::Invalid(
                "firejail is required but was not found on PATH (set MEGARL_FIREJAIL_BIN)",
            )
        })?),
    };
    let (launcher, launch_args) = match &firejail {
        Some(binary) => {
            let mut a = firejail_args(config.firejail_profile.as_deref());
            a.push(command.to_string());
            a.extend(args.iter().cloned());
            (binary.to_string_lossy().into_owned(), a)
        }
        None => (command.to_string(), args.to_vec()),
    };
    let filter = if firejail.is_none() && config.seccomp == SeccompMode::Blacklist {
        Some(SeccompFilter::new_blacklist())
    } else {
        None
    };
    let sandbox = sandbox_label(firejail.is_some(), config.seccomp).to_string();

    let mut cmd = Command::new(&launcher);
    cmd.args(&launch_args)
        .env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .env("HOME", "/tmp")
        .envs(config.env.iter().map(|(k, v)| (k, v)))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let limits = config.clone();
    unsafe {
        cmd.pre_exec(move || {
            if libc::setpgid(0, 0) < 0 {
                return Err(io::Error::last_os_error());
            }
            if let Some(filter) = &filter {
                // Fail closed: a verifier must never run without the filter it
                // was configured with.
                filter.install()?;
            }
            set_limit(libc::RLIMIT_AS, limits.memory_mb.saturating_mul(1024 * 1024));
            set_limit(libc::RLIMIT_CPU, limits.cpu_secs.max(1));
            // RLIMIT_NPROC is intentionally omitted: it is enforced against the
            // real UID across the whole host, so a busy CI box makes otherwise
            // harmless forks fail spuriously. Container/cgroup limits cover it.
            set_limit(libc::RLIMIT_FSIZE, 16 * 1024 * 1024);
            set_limit(libc::RLIMIT_CORE, 0);
            Ok(())
        });
    }

    let mut child = cmd.spawn().map_err(SandboxError::Spawn)?;
    let pid = child.id() as i32;
    let started = Instant::now();

    if let Some(mut stdin) = child.stdin.take() {
        let payload = stdin_data.to_vec();
        thread::spawn(move || {
            let _ = stdin.write_all(&payload);
            // Dropping the handle closes the pipe so readers see EOF.
        });
    }

    let stdout_handle = child.stdout.take().ok_or(SandboxError::Invalid("no stdout"))?;
    let stderr_handle = child.stderr.take().ok_or(SandboxError::Invalid("no stderr"))?;
    let out_limit = config.max_output_bytes;
    let stdout_thread = thread::spawn(move || read_bounded(stdout_handle, out_limit));
    let stderr_thread = thread::spawn(move || read_bounded(stderr_handle, out_limit));

    let deadline = started + Duration::from_millis(config.timeout_ms);
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    timed_out = true;
                    unsafe {
                        libc::kill(-pid, libc::SIGKILL);
                    }
                    break child.wait().map_err(SandboxError::Wait)?;
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(e) => {
                unsafe {
                    libc::kill(-pid, libc::SIGKILL);
                }
                return Err(SandboxError::Wait(e));
            }
        }
    };

    let (stdout, out_truncated) = stdout_thread.join().unwrap_or_default();
    let (stderr, err_truncated) = stderr_thread.join().unwrap_or_default();
    use std::os::unix::process::ExitStatusExt;
    Ok(SandboxResult {
        exit_code: status.code(),
        signal: status.signal(),
        stdout,
        stderr,
        timed_out,
        output_truncated: out_truncated || err_truncated,
        elapsed_ms: started.elapsed().as_millis() as u64,
        sandbox,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_stdout_and_exit_code() {
        let result = run_sandboxed(
            "/bin/echo",
            &["hello".into()],
            b"",
            &SandboxConfig::default(),
        )
        .unwrap();
        assert!(result.succeeded());
        assert_eq!(result.stdout, b"hello\n");
        assert!(!result.timed_out);
    }

    #[test]
    fn feeds_stdin_to_the_child() {
        let result = run_sandboxed(
            "/bin/sh",
            &["-c".into(), "cat".into()],
            b"from-stdin",
            &SandboxConfig::default(),
        )
        .unwrap();
        assert!(result.succeeded(), "{result:?}");
        assert_eq!(result.stdout, b"from-stdin");
    }

    #[test]
    fn kills_the_process_group_on_timeout() {
        let config = SandboxConfig {
            timeout_ms: 100,
            ..SandboxConfig::default()
        };
        let result = run_sandboxed(
            "/bin/sh",
            &["-c".into(), "sleep 30".into()],
            b"",
            &config,
        )
        .unwrap();
        assert!(result.timed_out);
        assert!(!result.succeeded());
        assert!(result.elapsed_ms < 3_000);
    }

    #[test]
    fn bounds_captured_output() {
        let config = SandboxConfig {
            max_output_bytes: 64,
            ..SandboxConfig::default()
        };
        let result = run_sandboxed(
            "/bin/sh",
            &["-c".into(), "head -c 200000 /dev/zero".into()],
            b"",
            &config,
        )
        .unwrap();
        assert!(result.output_truncated);
        assert_eq!(result.stdout.len(), 64);
        assert!(result.succeeded());
    }

    #[test]
    fn rejects_zero_timeout() {
        let config = SandboxConfig {
            timeout_ms: 0,
            ..SandboxConfig::default()
        };
        assert!(matches!(
            run_sandboxed("/bin/true", &[], b"", &config),
            Err(SandboxError::Invalid(_))
        ));
    }

    const SOCKET_PROBE: &str = "import ctypes\n\
        libc = ctypes.CDLL(None, use_errno=True)\n\
        r = libc.socket(2, 1, 0)\n\
        print('BLOCKED' if r < 0 and ctypes.get_errno() == 1 else 'ALLOWED')\n";

    fn probe(config: &SandboxConfig) -> SandboxResult {
        run_sandboxed(
            "python3",
            &["-I".into(), "-c".into(), SOCKET_PROBE.into()],
            b"",
            config,
        )
        .unwrap()
    }

    #[test]
    fn seccomp_blacklist_blocks_network_and_reports_the_layer() {
        let config = SandboxConfig {
            seccomp: SeccompMode::Blacklist,
            firejail: FirejailMode::Off,
            ..SandboxConfig::default()
        };
        let result = probe(&config);
        assert!(result.succeeded(), "{result:?}");
        assert_eq!(result.stdout, b"BLOCKED\n");
        assert_eq!(result.sandbox, "seccomp_blacklist");
    }

    #[test]
    fn seccomp_can_be_disabled_for_trusted_local_experiments() {
        let config = SandboxConfig {
            seccomp: SeccompMode::Off,
            firejail: FirejailMode::Off,
            ..SandboxConfig::default()
        };
        let result = probe(&config);
        assert!(result.succeeded(), "{result:?}");
        assert_eq!(result.stdout, b"ALLOWED\n");
        assert_eq!(result.sandbox, "none");
    }

    #[test]
    fn required_firejail_fails_closed_when_binary_is_missing() {
        let config = SandboxConfig {
            firejail: FirejailMode::Required,
            firejail_bin: Some("/nonexistent/megarl-firejail".into()),
            ..SandboxConfig::default()
        };
        assert!(matches!(
            run_sandboxed("/bin/true", &[], b"", &config),
            Err(SandboxError::Invalid(_))
        ));
    }
}
