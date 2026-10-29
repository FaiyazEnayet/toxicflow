//! Outbound networking: direct TLS or an HTTP CONNECT proxy.
//!
//! Connections are established under a validated policy, not a hardcoded
//! path. With no proxy configured the client connects directly; with a
//! proxy URL it tunnels through HTTP CONNECT, sending a
//! `Proxy-Authorization` header only when the URL carries credentials.
//! A malformed proxy URL is a configuration error: it fails fast with an
//! actionable message instead of being retried until a deadline.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use rustls::pki_types::ServerName;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// How outbound TCP connections are established.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyConfig {
    /// Connect straight to the destination.
    Direct,
    /// Tunnel through an HTTP CONNECT proxy. `auth` holds the raw
    /// `user:pass` pair and is present only when the URL carries it.
    Http {
        host: String,
        port: u16,
        auth: Option<String>,
    },
}

/// A proxy configuration problem. Retrying a bad config until a session
/// deadline only burns the recording window, so callers fail fast on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid proxy configuration: {}", self.0)
    }
}

impl Error for ConfigError {}

impl ProxyConfig {
    /// Parse a proxy URL of the form `http(s)://[user:pass@]host:port`.
    /// `None` or blank means direct. Credentials never appear in errors.
    pub fn parse(url: Option<&str>) -> Result<Self, ConfigError> {
        let url = match url {
            None => return Ok(ProxyConfig::Direct),
            Some(u) if u.trim().is_empty() => return Ok(ProxyConfig::Direct),
            Some(u) => u.trim(),
        };
        let without_scheme = url
            .strip_prefix("http://")
            .or_else(|| url.strip_prefix("https://"))
            .ok_or_else(|| {
                ConfigError(format!(
                    "proxy URL must start with http:// or https:// (got {url:?}); \
                     either unset https_proxy for a direct connection or fix the URL"
                ))
            })?;
        let (auth, hostport) = match without_scheme.split_once('@') {
            Some((a, h)) if !a.is_empty() => (Some(a.to_string()), h),
            _ => (None, without_scheme),
        };
        let (host, port) = hostport.split_once(':').ok_or_else(|| {
            ConfigError(format!(
                "proxy URL has no port; expected http://[user:pass@]host:port (got {without_scheme:?})"
            ))
        })?;
        if host.is_empty() {
            return Err(ConfigError(format!(
                "proxy URL has no host; expected http://[user:pass@]host:port (got {without_scheme:?})"
            )));
        }
        let port: u16 = port.parse().map_err(|_| {
            ConfigError(format!(
                "proxy port is not a number (got {port:?}); expected http://[user:pass@]host:port"
            ))
        })?;
        Ok(ProxyConfig::Http {
            host: host.to_string(),
            port,
            auth,
        })
    }

    /// Read the standard proxy env vars (`https_proxy` / `HTTPS_PROXY`).
    /// Absent or blank means direct.
    pub fn from_env() -> Result<Self, ConfigError> {
        let url = std::env::var("https_proxy")
            .or_else(|_| std::env::var("HTTPS_PROXY"))
            .ok();
        Self::parse(url.as_deref())
    }

    /// One-line description safe for logs: credentials are never shown.
    pub fn describe(&self) -> String {
        match self {
            ProxyConfig::Direct => "direct".to_string(),
            ProxyConfig::Http { host, port, auth } => {
                if auth.is_some() {
                    format!("proxy {host}:{port} (credentials redacted)")
                } else {
                    format!("proxy {host}:{port} (no credentials)")
                }
            }
        }
    }
}

/// TCP stream to `host:port`, directly or via an HTTP CONNECT proxy.
async fn tcp_stream(
    host: &str,
    port: u16,
    proxy: &ProxyConfig,
) -> Result<TcpStream, Box<dyn Error>> {
    match proxy {
        ProxyConfig::Direct => Ok(TcpStream::connect((host, port)).await?),
        ProxyConfig::Http {
            host: phost,
            port: pport,
            auth,
        } => {
            let mut stream = TcpStream::connect((phost.as_str(), *pport)).await?;
            let mut req = format!("CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n");
            if let Some(creds) = auth {
                let encoded = base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    creds.as_bytes(),
                );
                req.push_str(&format!("Proxy-Authorization: Basic {encoded}\r\n"));
            }
            req.push_str("Connection: keep-alive\r\n\r\n");
            stream.write_all(req.as_bytes()).await?;
            let mut buf = vec![0u8; 4096];
            let mut head = Vec::new();
            loop {
                let n = stream.read(&mut buf).await?;
                if n == 0 {
                    return Err("proxy closed the CONNECT tunnel".into());
                }
                head.extend_from_slice(&buf[..n]);
                if head.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
                if head.len() > 8192 {
                    return Err("proxy CONNECT response too large".into());
                }
            }
            let status = String::from_utf8_lossy(&head);
            if !status.starts_with("HTTP/1.1 200") && !status.starts_with("HTTP/1.0 200") {
                return Err(format!(
                    "proxy refused CONNECT to {host}:{port}: {}",
                    status.lines().next().unwrap_or("")
                )
                .into());
            }
            Ok(stream)
        }
    }
}

/// TLS stream to `host:443`, ready for a websocket handshake.
/// The connection policy is validated up front so a bad proxy URL fails
/// here instead of somewhere downstream.
pub async fn tls_stream(
    host: &str,
) -> Result<tokio_rustls::client::TlsStream<TcpStream>, Box<dyn Error>> {
    let proxy = ProxyConfig::from_env().map_err(|e| Box::new(e) as Box<dyn Error>)?;
    let tcp = tcp_stream(host, 443, &proxy).await?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = rustls::ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13, &rustls::version::TLS12])?
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let server_name = ServerName::try_from(host.to_string())?;
    Ok(connector.connect(server_name, tcp).await?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;

    #[test]
    fn parse_direct_when_absent_or_blank() {
        assert_eq!(ProxyConfig::parse(None), Ok(ProxyConfig::Direct));
        assert_eq!(ProxyConfig::parse(Some("")), Ok(ProxyConfig::Direct));
        assert_eq!(ProxyConfig::parse(Some("   ")), Ok(ProxyConfig::Direct));
    }

    #[test]
    fn parse_proxy_without_auth() {
        assert_eq!(
            ProxyConfig::parse(Some("http://proxy.internal:8080")),
            Ok(ProxyConfig::Http {
                host: "proxy.internal".to_string(),
                port: 8080,
                auth: None,
            })
        );
    }

    #[test]
    fn parse_proxy_with_auth() {
        assert_eq!(
            ProxyConfig::parse(Some("http://user:pass@proxy.internal:8080")),
            Ok(ProxyConfig::Http {
                host: "proxy.internal".to_string(),
                port: 8080,
                auth: Some("user:pass".to_string()),
            })
        );
    }

    #[test]
    fn parse_rejects_bad_urls_with_actionable_errors() {
        for bad in [
            "socks5://proxy:8080",
            "proxy:8080",
            "http://proxy",
            "http://:8080",
            "http://proxy:notaport",
            "http://",
        ] {
            let err = ProxyConfig::parse(Some(bad)).expect_err("should reject");
            let msg = err.to_string();
            assert!(
                msg.starts_with("invalid proxy configuration: "),
                "actionable prefix, got: {msg}"
            );
        }
    }

    #[test]
    fn describe_redacts_credentials() {
        // A URL can be syntactically fine yet must never surface its
        // credentials in describe() output.
        let cfg = ProxyConfig::parse(Some("http://alice:s3cret@proxy:3128")).unwrap();
        let desc = cfg.describe();
        assert!(!desc.contains("s3cret"), "password redacted: {desc}");
        assert!(!desc.contains("alice"), "username redacted: {desc}");
        assert!(desc.contains("credentials redacted"));
    }

    #[test]
    fn describe_direct_and_anonymous() {
        assert_eq!(ProxyConfig::Direct.describe(), "direct");
        let cfg = ProxyConfig::parse(Some("http://proxy:8080")).unwrap();
        assert!(cfg.describe().contains("no credentials"));
    }

    /// A fake HTTP CONNECT proxy: captures the request, then answers `status`.
    async fn fake_proxy(status: &'static str) -> (u16, Arc<Mutex<Option<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let seen_clone = seen.clone();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let mut head = Vec::new();
            loop {
                let n = sock.read(&mut buf).await.unwrap();
                if n == 0 {
                    break;
                }
                head.extend_from_slice(&buf[..n]);
                if head.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            *seen_clone.lock().unwrap() = Some(String::from_utf8_lossy(&head).to_string());
            sock.write_all(status.as_bytes()).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        });
        (port, seen)
    }

    #[tokio::test]
    async fn direct_tcp_reaches_local_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let accept = tokio::spawn(async move { listener.accept().await.unwrap() });
        let stream = tcp_stream("127.0.0.1", port, &ProxyConfig::Direct)
            .await
            .expect("direct connect");
        assert!(accept.await.is_ok());
        drop(stream);
    }

    #[tokio::test]
    async fn proxy_tunnel_sends_auth_header() {
        let (port, seen) = fake_proxy("HTTP/1.1 200 OK\r\n\r\n").await;
        let proxy = ProxyConfig::Http {
            host: "127.0.0.1".to_string(),
            port,
            auth: Some("user:pass".to_string()),
        };
        let _stream = tcp_stream("example.com", 443, &proxy)
            .await
            .expect("tunnel");
        let req = seen.lock().unwrap().clone().expect("proxy saw a request");
        assert!(
            req.starts_with("CONNECT example.com:443 "),
            "tunnels the target: {req}"
        );
        let want = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            "user:pass".as_bytes(),
        );
        assert!(
            req.contains(&format!("Proxy-Authorization: Basic {want}\r\n")),
            "auth header present: {req}"
        );
    }

    #[tokio::test]
    async fn proxy_tunnel_omits_auth_header_without_credentials() {
        let (port, seen) = fake_proxy("HTTP/1.1 200 OK\r\n\r\n").await;
        let proxy = ProxyConfig::Http {
            host: "127.0.0.1".to_string(),
            port,
            auth: None,
        };
        let _stream = tcp_stream("example.com", 443, &proxy)
            .await
            .expect("tunnel");
        let req = seen.lock().unwrap().clone().expect("proxy saw a request");
        assert!(req.starts_with("CONNECT example.com:443 "));
        assert!(
            !req.contains("Proxy-Authorization"),
            "no auth header without credentials: {req}"
        );
    }

    #[tokio::test]
    async fn proxy_refusal_is_an_error() {
        let (port, _) = fake_proxy("HTTP/1.1 407 Proxy Authentication Required\r\n\r\n").await;
        let proxy = ProxyConfig::Http {
            host: "127.0.0.1".to_string(),
            port,
            auth: None,
        };
        let err = tcp_stream("example.com", 443, &proxy)
            .await
            .expect_err("refused");
        assert!(
            err.to_string().contains("407"),
            "surfaces the refusal: {err}"
        );
    }
}
