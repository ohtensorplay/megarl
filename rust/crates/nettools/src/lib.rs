//! Hardened HTTP core for megaRL's network tools.
//!
//! Port of MegaLM's `rust/crates/server/src/remote_media.rs` (which itself was
//! ported from a Python media fetcher). Only public HTTP(S) endpoints are
//! reachable: DNS is resolved through a validating resolver, every redirect
//! hop is re-validated, and responses are size-capped while streaming. The
//! Python layer is a thin adapter — all security decisions live here.
//!
//! Two wire formats are supported:

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::{Duration, Instant};

use futures::StreamExt;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::header::{CONTENT_TYPE, LOCATION};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_BYTES: usize = 1 << 16;
const DEFAULT_MAX_REDIRECTS: usize = 3;
const DEFAULT_MAX_ATTEMPTS: usize = 2;
const DEFAULT_CONNECT_SECS: u64 = 5;
const DEFAULT_READ_SECS: u64 = 10;
const DEFAULT_TOTAL_SECS: u64 = 30;
const REDIRECT_STATUSES: [u16; 5] = [301, 302, 303, 307, 308];

fn default_max_bytes() -> usize {
    DEFAULT_MAX_BYTES
}
fn default_max_redirects() -> usize {
    DEFAULT_MAX_REDIRECTS
}
fn default_max_attempts() -> usize {
    DEFAULT_MAX_ATTEMPTS
}
fn default_connect_secs() -> u64 {
    DEFAULT_CONNECT_SECS
}
fn default_read_secs() -> u64 {
    DEFAULT_READ_SECS
}
fn default_total_secs() -> u64 {
    DEFAULT_TOTAL_SECS
}

/// A single bounded outbound HTTP request. All fields are JSON-stable.
#[derive(Debug, Clone, Deserialize)]
pub struct FetchRequest {
    pub method: String,
    pub url: String,
    #[serde(default)]
    pub body_json: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default = "default_max_bytes")]
    pub max_bytes: usize,
    #[serde(default = "default_max_redirects")]
    pub max_redirects: usize,
    #[serde(default = "default_max_attempts")]
    pub max_attempts: usize,
    #[serde(default = "default_connect_secs")]
    pub connect_timeout_secs: u64,
    #[serde(default = "default_read_secs")]
    pub read_timeout_secs: u64,
    #[serde(default = "default_total_secs")]
    pub total_timeout_secs: u64,
    #[serde(default)]
    pub allow_private: bool,
    #[serde(default)]
    pub allow_http: bool,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
}

/// Decoded response with provenance about caps and timing.
#[derive(Debug, Clone, Serialize)]
pub struct FetchResponse {
    pub status: u16,
    pub url: String,
    pub content_type: String,
    pub text: String,
    pub bytes_read: usize,
    pub truncated: bool,
    pub attempts: usize,
    pub elapsed_ms: u64,
}

/// Structured failure with a stable code, HTTP-like status and message.
#[derive(Debug, Clone, Serialize)]
pub struct FetchError {
    pub code: &'static str,
    pub status: u16,
    pub message: String,
}

impl FetchError {
    fn new(code: &'static str, status: u16, message: impl Into<String>) -> Self {
        Self {
            code,
            status,
            message: message.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public-address checks (mirrors MegaLM remote_media.rs)
// ---------------------------------------------------------------------------

fn v4_is_public(address: Ipv4Addr) -> bool {
    let [a, b, c, _d] = address.octets();
    if a == 0 || a == 10 || a == 127 || a >= 224 {
        return false;
    }
    if a == 100 && (64..=127).contains(&b) {
        return false;
    }
    if a == 169 && b == 254 {
        return false;
    }
    if a == 172 && (16..=31).contains(&b) {
        return false;
    }
    if a == 192 && b == 168 {
        return false;
    }
    if a == 192 && b == 0 && (c == 0 || c == 2) {
        return false;
    }
    if a == 192 && b == 88 && c == 99 {
        return false;
    }
    if a == 198 && (b == 18 || b == 19) {
        return false;
    }
    if a == 198 && b == 51 && c == 100 {
        return false;
    }
    if a == 203 && b == 0 && c == 113 {
        return false;
    }
    true
}

fn v6_is_public(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4_mapped() {
        return v4_is_public(mapped);
    }
    let o = address.octets();
    if o[..15].iter().all(|byte| *byte == 0) && (o[15] == 0 || o[15] == 1) {
        return false; // :: and ::1
    }
    if o[0] & 0xfe == 0xfc {
        return false; // fc00::/7 unique local
    }
    if o[0] == 0xfe && (o[1] & 0xc0) == 0x80 {
        return false; // fe80::/10 link local
    }
    if o[0] == 0xff {
        return false; // multicast
    }
    if o[0] == 0x20 && o[1] == 0x01 && o[2] == 0x0d && o[3] == 0xb8 {
        return false; // 2001:db8::/32 documentation
    }
    if o[0] == 0x20 && o[1] == 0x01 && (o[2] & 0xfe) == 0x00 {
        return false; // 2001::/23
    }
    if o[0] == 0x00 && o[1] == 0x64 && o[2] == 0xff && o[3] == 0x9b {
        return false; // 64:ff9b::/96 NAT64
    }
    if o[..8] == [0x01, 0x00, 0x00, 0x00, 0, 0, 0, 0] {
        return false; // 100::/64 discard-only
    }
    if o[0] == 0x20 && o[1] == 0x02 {
        return false; // 2002::/16 6to4
    }
    true
}

fn address_is_public(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(v4) => v4_is_public(v4),
        IpAddr::V6(v6) => v6_is_public(v6),
    }
}

/// Resolver that rejects non-public addresses *inside the connect path*, so a
/// second DNS lookup cannot rebind a validated hostname to an internal IP.
struct PublicResolver {
    allow_private: bool,
}

impl Resolve for PublicResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        let allow_private = self.allow_private;
        Box::pin(async move {
            let mut addresses = Vec::new();
            if let Ok(ip) = host.parse::<IpAddr>() {
                if !allow_private && !address_is_public(ip) {
                    return Err("host resolves to a non-public address".into());
                }
                addresses.push(SocketAddr::new(ip, 0));
            } else {
                let resolved = tokio::net::lookup_host((host.as_str(), 0))
                    .await
                    .map_err(|_| "host could not be resolved")?;
                for address in resolved {
                    if !allow_private && !address_is_public(address.ip()) {
                        return Err("host resolves to a non-public address".into());
                    }
                    addresses.push(SocketAddr::new(address.ip(), 0));
                }
            }
            if addresses.is_empty() {
                return Err("host has no usable address".into());
            }
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

// ---------------------------------------------------------------------------
// URL policy
// ---------------------------------------------------------------------------

fn parse_url(url: &str, request: &FetchRequest) -> Result<reqwest::Url, FetchError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|_| FetchError::new("invalid_url", 400, "URL is invalid"))?;
    let scheme = parsed.scheme().to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return Err(FetchError::new(
            "unsupported_scheme",
            400,
            format!("unsupported URL scheme: {scheme}"),
        ));
    }
    if scheme == "http" && !request.allow_http {
        return Err(FetchError::new(
            "http_disabled",
            400,
            "http URLs are disabled by default; set allow_http=true",
        ));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(FetchError::new(
            "userinfo_disallowed",
            400,
            "URL userinfo is not allowed",
        ));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| FetchError::new("missing_host", 400, "URL has no host"))?
        .to_ascii_lowercase();
    // Hyper does not consult a custom resolver for IP-literal hosts, so check
    // literals here; hostnames are checked inside PublicResolver.
    if !request.allow_private {
        if let Ok(ip) = host.parse::<IpAddr>() {
            if !address_is_public(ip) {
                return Err(FetchError::new(
                    "blocked_address",
                    400,
                    format!("host resolves to non-public address: {host}"),
                ));
            }
        }
    }
    if request.allowed_domains.is_empty() {
        return Ok(parsed);
    }
    let host = host.trim_end_matches('.');
    for rule in &request.allowed_domains {
        if let Some(suffix) = rule.strip_prefix('.') {
            if host == suffix || host.ends_with(rule) {
                return Ok(parsed);
            }
        } else if host == *rule {
            return Ok(parsed);
        }
    }
    Err(FetchError::new(
        "domain_not_allowed",
        400,
        format!("domain {host:?} is not in the allowlist"),
    ))
}

fn is_text(content_type: &str) -> bool {
    let mime = content_type.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    matches!(
        mime.as_str(),
        "text/plain" | "text/html" | "text/xml" | "application/json" | "application/xml"
    ) || mime.ends_with("+json")
        || mime.ends_with("+xml")
        || mime.ends_with("+text")
}

// ---------------------------------------------------------------------------
// Fetch
// ---------------------------------------------------------------------------

/// Perform one bounded, SSRF-hardened request. Redirects are followed manually
/// and re-validated; only 5xx/408/429 and transport failures are retried.
pub async fn http_fetch(request: FetchRequest) -> Result<FetchResponse, FetchError> {
    if request.max_bytes == 0 || request.max_attempts == 0 || request.total_timeout_secs == 0 {
        return Err(FetchError::new("invalid_limits", 400, "invalid HTTP limits"));
    }
    let method = request.method.to_ascii_uppercase();
    if method != "GET" && method != "POST" {
        return Err(FetchError::new(
            "unsupported_method",
            400,
            "method must be GET or POST",
        ));
    }
    let started = Instant::now();
    let deadline = started + Duration::from_secs(request.total_timeout_secs);
    let connect_timeout = Duration::from_secs(request.connect_timeout_secs.min(request.total_timeout_secs));
    let read_timeout = Duration::from_secs(request.read_timeout_secs.min(request.total_timeout_secs));

    let client = reqwest::Client::builder()
        .dns_resolver(std::sync::Arc::new(PublicResolver {
            allow_private: request.allow_private,
        }))
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(connect_timeout)
        .timeout(read_timeout)
        .user_agent("megarl-nettools/1.0")
        .build()
        .map_err(|e| FetchError::new("client_error", 500, format!("client build failed: {e}")))?;

    let mut current = request.url.clone();
    let mut redirects = 0usize;
    let mut attempts = 0usize;
    loop {
        if Instant::now() >= deadline {
            return Err(FetchError::new("timeout", 504, "fetch timed out"));
        }
        let parsed = parse_url(&current, &request)?;
        let mut builder = client.request(
            reqwest::Method::from_bytes(method.as_bytes())
                .map_err(|_| FetchError::new("invalid_method", 400, "invalid method"))?,
            parsed.clone(),
        );
        for (key, value) in &request.headers {
            builder = builder.header(key.as_str(), value.as_str());
        }
        if let Some(body) = &request.body_json {
            builder = builder.header(CONTENT_TYPE, "application/json").body(body.clone());
        }

        let mut response = None;
        for attempt in 0..request.max_attempts {
            attempts += 1;
            if Instant::now() >= deadline {
                return Err(FetchError::new("timeout", 504, "fetch timed out"));
            }
            match builder.try_clone().ok_or_else(|| {
                FetchError::new("invalid_body", 400, "request body cannot be retried")
            }) {
                Ok(request_builder) => match request_builder.send().await {
                    Ok(reply) => {
                        let status = reply.status().as_u16();
                        let retryable = (500..600).contains(&status) || status == 408 || status == 429;
                        if retryable && attempt + 1 < request.max_attempts {
                            continue;
                        }
                        response = Some(reply);
                        break;
                    }
                    Err(_) if attempt + 1 < request.max_attempts => continue,
                    Err(_) => {
                        return Err(FetchError::new(
                            "upstream_unreachable",
                            502,
                            "host could not be reached",
                        ))
                    }
                },
                Err(e) => return Err(e),
            }
        }
        let response = response.ok_or_else(|| FetchError::new("timeout", 504, "fetch timed out"))?;
        let status = response.status().as_u16();

        if REDIRECT_STATUSES.contains(&status) {
            if let Some(location) = response.headers().get(LOCATION).and_then(|v| v.to_str().ok()) {
                if redirects >= request.max_redirects {
                    return Err(FetchError::new(
                        "too_many_redirects",
                        400,
                        format!("too many redirects (limit {})", request.max_redirects),
                    ));
                }
                redirects += 1;
                current = parsed
                    .join(location)
                    .map_err(|_| FetchError::new("invalid_redirect", 400, "invalid redirect URL"))?
                    .to_string();
                continue;
            }
        }

        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let mut payload = Vec::new();
        let mut truncated = false;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| {
                FetchError::new("upstream_unreachable", 502, "response stream failed")
            })?;
            if payload.len() + chunk.len() > request.max_bytes {
                truncated = true;
                break;
            }
            payload.extend_from_slice(&chunk);
        }
        let text = if is_text(&content_type) {
            // Decoded leniently: malformed UTF-8 must never abort an episode.
            String::from_utf8_lossy(&payload).into_owned()
        } else {
            format!(
                "[binary content omitted: {}, {} bytes]",
                if content_type.is_empty() { "unknown" } else { &content_type },
                payload.len()
            )
        };
        return Ok(FetchResponse {
            status,
            url: current,
            content_type,
            text,
            bytes_read: payload.len(),
            truncated,
            attempts,
            elapsed_ms: started.elapsed().as_millis() as u64,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(url: &str) -> FetchRequest {
        FetchRequest {
            method: "GET".into(),
            url: url.into(),
            body_json: None,
            headers: BTreeMap::new(),
            max_bytes: 1024,
            max_redirects: 3,
            max_attempts: 1,
            connect_timeout_secs: 1,
            read_timeout_secs: 1,
            total_timeout_secs: 2,
            allow_private: false,
            allow_http: true,
            allowed_domains: vec![],
        }
    }

    #[test]
    fn rejects_private_and_reserved_addresses() {
        for address in [
            "127.0.0.1",
            "10.0.0.1",
            "192.168.1.1",
            "169.254.1.1",
            "0.0.0.0",
            "224.0.0.1",
        ] {
            assert!(!v4_is_public(address.parse().unwrap()), "{address}");
        }
        assert!(v4_is_public("8.8.8.8".parse().unwrap()));
        assert!(!v6_is_public("::1".parse().unwrap()));
        assert!(!v6_is_public("fc00::1".parse().unwrap()));
        assert!(v6_is_public("2606:4700:4700::1111".parse().unwrap()));
    }

    #[test]
    fn rejects_bad_schemes_and_userinfo() {
        let mut bad = request("file:///etc/passwd");
        assert_eq!(http_fetch_blocking(bad.clone()).unwrap_err().code, "unsupported_scheme");
        bad.url = "http://user:pass@example.com/".into();
        assert_eq!(http_fetch_blocking(bad).unwrap_err().code, "userinfo_disallowed");
    }

    #[test]
    fn blocks_ip_literals_before_connect() {
        // Hyper bypasses custom resolvers for literal IPs, so parse_url must
        // reject them itself (regression test for the SSRF guard).
        let req = request("http://127.0.0.1:9/secret");
        let err = parse_url(&req.url, &req).unwrap_err();
        assert_eq!(err.code, "blocked_address");
        assert!(err.message.contains("non-public address"));
        let mut allowed = request("http://127.0.0.1:9/ok");
        allowed.allow_private = true;
        assert!(parse_url(&allowed.url, &allowed).is_ok());
    }

    #[test]
    fn allowlist_matches_exact_and_suffix() {
        let mut req = request("https://example.com/x");
        req.allowed_domains = vec![".example.com".into()];
        assert!(parse_url(&req.url, &req).is_ok());
        req.url = "https://evil-example.com/".into();
        assert_eq!(parse_url(&req.url, &req).unwrap_err().code, "domain_not_allowed");
    }

    // Minimal blocking wrapper for tests only; production uses the async API.
    fn http_fetch_blocking(request: FetchRequest) -> Result<FetchResponse, FetchError> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(http_fetch(request))
    }
}
