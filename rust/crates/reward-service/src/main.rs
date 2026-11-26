//! Standalone reward verifier service.
//!
//! Usage: megarl-reward-router [listen_addr]
//! Env: MEGARL_REWARD_TOKEN, MEGARL_REWARD_WORKERS,
//!      MEGARL_REWARD_MAX_RETRIES, MEGARL_REWARD_MAX_BATCH.

use megarl_reward_service::{router, RewardState};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listen = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:8081".to_string());
    let workers = std::env::var("MEGARL_REWARD_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let retries = std::env::var("MEGARL_REWARD_MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let token = std::env::var("MEGARL_REWARD_TOKEN").ok().filter(|s| !s.is_empty());
    let listener = TcpListener::bind(&listen).await?;
    println!("{}", listener.local_addr()?);
    axum::serve(
        listener,
        router(RewardState::new(token, workers, retries)),
    )
    .await?;
    Ok(())
}
