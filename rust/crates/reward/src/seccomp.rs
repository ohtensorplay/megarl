//! Seccomp-BPF and Firejail hardening for code-reward subprocesses.
//!
//! The candidate program is untrusted. Resource limits bound *how much* it can
//! consume but not *what* it may ask the kernel to do: Python code can call
//! `socket`, `ptrace`, `mount`, `keyctl`, ... directly through `ctypes`. This
//! module adds a syscall-level layer:
//!
//! - a classic seccomp-BPF filter that denies a curated set of dangerous
//!   syscalls with `EPERM` while leaving ordinary compute/IO syscalls intact;
//! - an optional Firejail wrapper (network-less private namespace) as a
//!   stronger, namespace-isolated layer when its binary is available.
//!
//! The filter is **built before `fork()`** — allocating in a forked child is
//! not async-signal-safe — and installed inside `Command::pre_exec` using only
//! `prctl` and the `seccomp` syscall. Installation is fail-closed: if the
//! kernel refuses the filter, spawning the verifier fails rather than silently
//! running untrusted code without the intended restriction.
//!
//! Firejail installs its own seccomp filter *after* exec, so when it is used
//! our pre-exec filter is skipped (its rlimits would block Firejail's own
//! namespace setup).
//!
//! `fork`/`clone`/`execve` are deliberately left available: `execve` is needed
//! to start the verifier at all, and forked/exec'd children inherit the filter,
//! so they cannot regain the blocked syscalls. Fork storms are bounded by the
//! CPU/address-space rlimits and the timeout-driven process-group kill; the
//! optional Firejail layer adds namespace isolation on top.

use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// In-process seccomp layer. `Blacklist` is the default: it blocks escape,
/// networking and kernel-attack syscalls without breaking pure computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SeccompMode {
    /// No filter. Only for local experiments on trusted code.
    Off,
    /// Deny the curated syscall blacklist with `EPERM`.
    #[default]
    Blacklist,
}

impl SeccompMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SeccompMode::Off => "off",
            SeccompMode::Blacklist => "blacklist",
        }
    }
}

impl std::str::FromStr for SeccompMode {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(SeccompMode::Off),
            "blacklist" | "on" | "enabled" => Ok(SeccompMode::Blacklist),
            _ => Err("expected off or blacklist"),
        }
    }
}

/// Optional Firejail layer. `Auto` uses it when installed and falls back to
/// the in-process seccomp filter otherwise; `Required` fails closed when the
/// binary is missing or not executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FirejailMode {
    #[default]
    Auto,
    Off,
    Required,
}

impl FirejailMode {
    pub fn as_str(self) -> &'static str {
        match self {
            FirejailMode::Auto => "auto",
            FirejailMode::Off => "off",
            FirejailMode::Required => "required",
        }
    }
}

impl std::str::FromStr for FirejailMode {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "on" => Ok(FirejailMode::Auto),
            "off" | "none" | "disabled" => Ok(FirejailMode::Off),
            "required" | "require" => Ok(FirejailMode::Required),
            _ => Err("expected auto, off or required"),
        }
    }
}

// Classic BPF opcodes (see linux/filter.h).
const BPF_LD: u16 = 0x00;
const BPF_W: u16 = 0x00;
const BPF_ABS: u16 = 0x20;
const BPF_JMP: u16 = 0x05;
const BPF_JEQ: u16 = 0x10;
const BPF_K: u16 = 0x00;
const BPF_RET: u16 = 0x06;

// Seccomp return actions (see linux/seccomp.h).
const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;

/// `struct sock_filter` / `struct sock_fprog` in a shape the `seccomp(2)`
/// syscall accepts. Defined locally because libc does not expose them.
#[repr(C)]
#[derive(Clone, Copy)]
struct SockFilter {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}

#[repr(C)]
struct SockFprog {
    len: u16,
    filter: *const SockFilter,
}

/// Syscalls denied to an untrusted verifier. These are never needed by pure
/// scoring code but enable networking, debugging, namespace/mount escape,
/// kernel keyring abuse, kernel-module loading and cross-process memory access.
/// Numbers come from the native `libc` constants, so no architecture-specific
/// literals are embedded here.
fn denied_syscalls() -> &'static [libc::c_long] {
    &[
        // Network sockets (exfiltration / remote control).
        libc::SYS_socket,
        libc::SYS_connect,
        libc::SYS_accept,
        libc::SYS_sendto,
        libc::SYS_recvfrom,
        libc::SYS_sendmsg,
        libc::SYS_recvmsg,
        libc::SYS_shutdown,
        libc::SYS_bind,
        libc::SYS_listen,
        libc::SYS_socketpair,
        libc::SYS_setsockopt,
        libc::SYS_getsockopt,
        // Debugging and cross-process memory access.
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_userfaultfd,
        libc::SYS_perf_event_open,
        libc::SYS_bpf,
        // Mount / namespace escapes.
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_unshare,
        libc::SYS_setns,
        libc::SYS_fanotify_init,
        // Kernel keyring and module loading.
        libc::SYS_add_key,
        libc::SYS_request_key,
        libc::SYS_keyctl,
        libc::SYS_init_module,
        libc::SYS_delete_module,
        // Privileged host administration.
        libc::SYS_reboot,
        libc::SYS_swapon,
        libc::SYS_swapoff,
        libc::SYS_sethostname,
        libc::SYS_setdomainname,
        libc::SYS_iopl,
        libc::SYS_ioperm,
        libc::SYS_syslog,
        libc::SYS_acct,
        libc::SYS_quotactl,
        // clone3 carries namespace flags; glibc falls back to clone when it is
        // denied with EPERM, so ordinary threads/forks keep working.
        libc::SYS_clone3,
    ]
}

/// A ready-to-install blacklist filter. Owned so it can be moved into a
/// `pre_exec` closure without allocating in the child.
#[derive(Clone)]
pub struct SeccompFilter {
    program: Vec<SockFilter>,
    denied: Vec<u32>,
}

impl SeccompFilter {
    /// Build the `Blacklist` program. All allocation happens here, in the
    /// parent, before `fork()`.
    pub fn new_blacklist() -> Self {
        let mut denied: Vec<u32> = denied_syscalls()
            .iter()
            .map(|&nr| nr as u32)
            .collect();
        denied.sort_unstable();
        denied.dedup();

        let mut program = Vec::with_capacity(denied.len() + 2);
        // Load seccomp_data.nr (offset 0).
        program.push(SockFilter {
            code: BPF_LD | BPF_W | BPF_ABS,
            jt: 0,
            jf: 0,
            k: 0,
        });
        for nr in &denied {
            // jt=0 -> fall through to the RET ERRNO below; jf=1 -> skip it.
            program.push(SockFilter {
                code: BPF_JMP | BPF_JEQ | BPF_K,
                jt: 0,
                jf: 1,
                k: *nr,
            });
            program.push(SockFilter {
                code: BPF_RET | BPF_K,
                jt: 0,
                jf: 0,
                k: SECCOMP_RET_ERRNO | libc::EPERM as u32,
            });
        }
        program.push(SockFilter {
            code: BPF_RET | BPF_K,
            jt: 0,
            jf: 0,
            k: SECCOMP_RET_ALLOW,
        });
        Self { program, denied }
    }

    /// Number of BPF instructions; exposed for tests/diagnostics.
    pub fn len(&self) -> usize {
        self.program.len()
    }

    pub fn is_empty(&self) -> bool {
        self.program.is_empty()
    }

    /// Syscall numbers this filter denies; exposed for tests/diagnostics.
    pub fn denied(&self) -> &[u32] {
        &self.denied
    }

    /// Install the filter in the calling process. Uses only `prctl` and the
    /// `seccomp` syscall so it is safe from a `pre_exec` hook.
    pub fn install(&self) -> io::Result<()> {
        if self.program.len() > u16::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seccomp program exceeds sock_fprog::len",
            ));
        }
        unsafe {
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0 {
                return Err(io::Error::last_os_error());
            }
            let prog = SockFprog {
                len: self.program.len() as u16,
                filter: self.program.as_ptr(),
            };
            if libc::syscall(
                libc::SYS_seccomp,
                libc::SECCOMP_SET_MODE_FILTER,
                0,
                &prog,
            ) < 0
            {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }
}

/// Resolve the Firejail binary. An explicit override (config field or
/// `MEGARL_FIREJAIL_BIN`) takes precedence over PATH discovery; this also makes
/// the `Required` path testable on hosts without Firejail.
pub fn find_firejail(override_path: Option<&str>) -> Option<PathBuf> {
    let explicit = override_path
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("MEGARL_FIREJAIL_BIN").map(PathBuf::from));
    if let Some(path) = explicit {
        return is_executable(&path).then_some(path);
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join("firejail");
        if is_executable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn is_executable(path: &Path) -> bool {
    path.is_file()
        && std::fs::metadata(path)
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
}

/// Firejail options for a verifier. Network is disabled and the filesystem is
/// private; the candidate itself arrives on stdin, so no host paths are needed.
pub fn firejail_args(profile: Option<&str>) -> Vec<String> {
    let mut args = vec![
        "--quiet".to_string(),
        "--net=none".to_string(),
        "--private".to_string(),
        "--nosound".to_string(),
        "--nodbus".to_string(),
        "--disable-mnt".to_string(),
        "--seccomp".to_string(),
    ];
    match profile {
        Some(path) if !path.is_empty() => args.push(format!("--profile={path}")),
        // No profile means the built-in default filter; `--noprofile` keeps
        // per-user ~/.config/firejail files from changing the policy silently.
        _ => args.push("--noprofile".to_string()),
    }
    args
}

/// Human-readable layer description stored in grade results and metrics.
pub fn sandbox_label(firejail: bool, seccomp: SeccompMode) -> &'static str {
    match (firejail, seccomp) {
        (true, _) => "firejail+seccomp",
        (false, SeccompMode::Blacklist) => "seccomp_blacklist",
        (false, SeccompMode::Off) => "none",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blacklist_program_blocks_dangerous_syscalls_and_allows_the_rest() {
        let filter = SeccompFilter::new_blacklist();
        // One load + two instructions per syscall + one final allow.
        assert_eq!(filter.len(), filter.denied().len() * 2 + 2);
        let denied = filter.denied();
        for name in ["socket", "connect", "ptrace", "mount", "unshare", "bpf", "keyctl"] {
            let nr = syscall_nr(name);
            assert!(denied.contains(&(nr as u32)), "{name} is not denied");
        }
        // Ordinary Python syscalls must stay available.
        for name in ["read", "write", "openat", "mmap", "futex", "execve", "exit_group"] {
            let nr = syscall_nr(name);
            assert!(!denied.contains(&(nr as u32)), "{name} must stay allowed");
        }
        assert!(filter.len() > 8);
    }

    fn syscall_nr(name: &str) -> libc::c_long {
        match name {
            "socket" => libc::SYS_socket,
            "connect" => libc::SYS_connect,
            "ptrace" => libc::SYS_ptrace,
            "mount" => libc::SYS_mount,
            "unshare" => libc::SYS_unshare,
            "bpf" => libc::SYS_bpf,
            "keyctl" => libc::SYS_keyctl,
            "read" => libc::SYS_read,
            "write" => libc::SYS_write,
            "openat" => libc::SYS_openat,
            "mmap" => libc::SYS_mmap,
            "futex" => libc::SYS_futex,
            "execve" => libc::SYS_execve,
            "exit_group" => libc::SYS_exit_group,
            _ => panic!("unknown syscall in test: {name}"),
        }
    }

    #[test]
    fn firejail_options_disable_network_and_user_profiles() {
        let args = firejail_args(None);
        assert!(args.iter().any(|a| a == "--net=none"));
        assert!(args.iter().any(|a| a == "--private"));
        assert!(args.iter().any(|a| a == "--seccomp"));
        assert!(args.iter().any(|a| a == "--noprofile"));
        let with_profile = firejail_args(Some("/etc/megarl/grader.profile"));
        assert!(!with_profile.iter().any(|a| a == "--noprofile"));
        assert!(with_profile
            .iter()
            .any(|a| a == "--profile=/etc/megarl/grader.profile"));
    }

    #[test]
    fn labels_identify_the_active_layer() {
        assert_eq!(sandbox_label(false, SeccompMode::Off), "none");
        assert_eq!(
            sandbox_label(false, SeccompMode::Blacklist),
            "seccomp_blacklist"
        );
        assert_eq!(sandbox_label(true, SeccompMode::Blacklist), "firejail+seccomp");
    }
}
