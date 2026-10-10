//! Outbound networking through the sandbox egress proxy.
//!
//! This environment has no direct internet: every connection must be
//! tunnelled through the HTTP proxy in `https_proxy` via CONNECT.
//! reqwest honors the proxy env vars on its own; raw websocket clients do
//! not, so this module builds a proxied TLS stream that
//! `tokio_tungstenite::client_async` can handshake over.

use std::error::Error;
use std::sync::Arc;

use rustls::pki_types::ServerName;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Parse `http://user:pass@host:port` from the proxy env vars.
fn proxy_target() -> Result<(String, String, String), Box<dyn Error>> {
    let url = std::env::var("https_proxy")
        .or_else(|_| std::env::var("HTTPS_PROXY"))
        .map_err(|_| "no https_proxy set")?;
    let url = url.strip_prefix("http://").unwrap_or(&url);
    let (auth, hostport) = url.split_once('@').ok_or("proxy URL has no auth")?;
    let (host, port) = hostport.split_once(':').ok_or("proxy URL has no port")?;
    Ok((auth.to_string(), host.to_string(), port.to_string()))
}

/// TCP tunnel to `host:443` through the egress proxy (HTTP CONNECT).
async fn connect_tunnel(host: &str) -> Result<TcpStream, Box<dyn Error>> {
    let (auth, phost, pport) = proxy_target()?;
    let mut stream = TcpStream::connect(format!("{phost}:{pport}")).await?;
    let creds = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        auth.as_bytes(),
    );
    let req = format!(
        "CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\
         Proxy-Authorization: Basic {creds}\r\nConnection: keep-alive\r\n\r\n"
    );
    stream.write_all(req.as_bytes()).await?;
    let mut buf = vec![0u8; 4096];
    let mut head = Vec::new();
    loop {
        let n = stream.read(&mut buf).await?;
        if n == 0 {
            return Err("proxy closed CONNECT".into());
        }
        head.extend_from_slice(&buf[..n]);
        if head.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if head.len() > 8192 {
            return Err("proxy response too large".into());
        }
    }
    let status = String::from_utf8_lossy(&head);
    if !status.starts_with("HTTP/1.1 200") && !status.starts_with("HTTP/1.0 200") {
        return Err(format!("proxy CONNECT refused: {}", status.lines().next().unwrap_or("")).into());
    }
    Ok(stream)
}

/// TLS-over-CONNECT stream to `host:443`, ready for a websocket handshake.
pub async fn tls_stream(
    host: &str,
) -> Result<
    tokio_rustls::client::TlsStream<TcpStream>,
    Box<dyn Error>,
> {
    let tcp = connect_tunnel(host).await?;
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
