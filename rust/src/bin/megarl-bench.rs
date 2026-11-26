//! Local throughput harness for the production reward plane.
//!
//! Usage: megarl-bench [requests] [workers]
//! Prints one JSON object so CI can track regressions.

use std::time::Instant;

use megarl_reward::{run_sandboxed, SandboxConfig, VerifyPool, VerifyRequest};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let count: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4_096);
    let workers: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let requests: Vec<VerifyRequest> = (0..count)
        .map(|i| VerifyRequest {
            request_id: i as u64,
            domain: if i % 4 == 0 {
                "math".into()
            } else if i % 4 == 1 {
                "code".into()
            } else {
                "generic".into()
            },
            response: if i % 4 == 1 {
                "pass\n".into()
            } else {
                format!("{}", i % 7)
            },
            expected: format!("{}", i % 7),
            command: if i % 4 == 1 { "/bin/sh".into() } else { String::new() },
            args: if i % 4 == 1 {
                vec!["-c".into(), "read line; test \"$line\" = pass".into()]
            } else {
                Vec::new()
            },
            timeout_ms: 2_000,
            ..VerifyRequest::default()
        })
        .collect();

    let pool = VerifyPool::new(workers, 0);
    let started = Instant::now();
    let outcomes = pool.verify_batch(requests);
    let elapsed = started.elapsed();
    let correct = outcomes.iter().filter(|o| o.reward == 1.0).count();
    let (completed, failed, retries, timeouts, _) = pool.stats().values();
    let seconds = elapsed.as_secs_f64().max(1e-9);

    // Measure sandbox startup cost without verification work on top.
    let sandbox_runs = 32;
    let sandbox_started = Instant::now();
    for _ in 0..sandbox_runs {
        let result = run_sandboxed("/bin/true", &[], b"", &SandboxConfig::default());
        assert!(result.map(|r| r.succeeded()).unwrap_or(false));
    }
    let sandbox_ms = sandbox_started.elapsed().as_millis() as u64;

    println!(
        "{{\"requests\":{},\"workers\":{},\"elapsed_ms\":{},\"throughput_per_sec\":{:.0},\
         \"correct\":{},\"completed\":{},\"failed\":{},\"retries\":{},\"timeouts\":{},\
         \"sandbox_startup_ms\":{}}}",
        count,
        workers,
        elapsed.as_millis(),
        count as f64 / seconds,
        correct,
        completed,
        failed,
        retries,
        timeouts,
        sandbox_ms / sandbox_runs,
    );
}
