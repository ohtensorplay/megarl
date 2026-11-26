//! Rust rollout gateway binary.
//!
//! Usage: megarl-gateway [listen_addr] [python_worker_addr,another_addr,...]
//! Prints the bound address on stdout so a supervisor can discover the port.
//! Multiple worker addresses enable round-robin and sticky-by-session routing;
//! a failed worker is retried on a different one.

use megarl_gateway::serve;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listen = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:0".to_string());
    let workers = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "127.0.0.1:8765".to_string())
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    let listener = TcpListener::bind(&listen).await?;
    println!("{}", listener.local_addr()?);
    serve(listener, workers).await
}
