//! Stage 3 toxicflow H2: proxy error messages must never disclose the
//! supplied URL or its credentials, and `https://` proxy URLs must be
//! rejected because the CONNECT socket is plaintext TCP.
//!
//! Scheme table (documented on `ProxyConfig::parse`):
//! - `http://`  supported: plaintext TCP CONNECT tunnel; TLS is
//!   established to the destination after CONNECT.
//! - `https://` rejected: TLS to the proxy is not supported.
//! - anything else rejected: unsupported scheme.

use toxicflow::net::ProxyConfig;

/// Parse `url` and return the error rendered both ways. Fails the test
/// when the URL parses: every crafted case here must be rejected.
fn error_texts(url: &str) -> (String, String) {
    let err = ProxyConfig::parse(Some(url)).expect_err("malformed proxy URL must be rejected");
    (err.to_string(), format!("{err:?}"))
}

/// No rendering of the error may contain the secret token or the username
/// that carried it.
fn assert_no_secret(url: &str, secret: &str, username: &str) {
    let (display, debug) = error_texts(url);
    for (label, text) in [("Display", &display), ("Debug", &debug)] {
        assert!(
            !text.contains(secret),
            "secret token leaked into {label} error for {url:?}: {text}"
        );
        assert!(
            !text.contains(username),
            "username leaked into {label} error for {url:?}: {text}"
        );
    }
}

#[test]
fn stage3_toxicflow_2_bad_scheme_with_credentials_leaks_nothing() {
    assert_no_secret(
        "socks5://alice:SECRET_TOKEN_ALPHA@proxy:8080",
        "SECRET_TOKEN_ALPHA",
        "alice",
    );
}

#[test]
fn stage3_toxicflow_2_no_port_with_credentials_leaks_nothing() {
    assert_no_secret(
        "http://bob:SECRET_TOKEN_BETA@proxy",
        "SECRET_TOKEN_BETA",
        "bob",
    );
}

#[test]
fn stage3_toxicflow_2_no_host_with_credentials_leaks_nothing() {
    assert_no_secret(
        "http://carol:SECRET_TOKEN_GAMMA@:8080",
        "SECRET_TOKEN_GAMMA",
        "carol",
    );
}

#[test]
fn stage3_toxicflow_2_bad_port_leaks_nothing() {
    // The port fragment is user input and was echoed verbatim, so a
    // secret typed in that position reached the error message. This URL
    // carries no username, so only the secret is asserted here.
    let (display, debug) = error_texts("http://proxy:SECRET_TOKEN_DELTA");
    for (label, text) in [("Display", &display), ("Debug", &debug)] {
        assert!(
            !text.contains("SECRET_TOKEN_DELTA"),
            "secret token leaked into {label} error: {text}"
        );
    }
}

#[test]
fn stage3_toxicflow_2_https_with_credentials_rejected_without_leak() {
    let url = "https://erin:SECRET_TOKEN_EPSILON@proxy:8443";
    let err = ProxyConfig::parse(Some(url)).expect_err("https:// proxy URL must be rejected");
    let msg = err.to_string();
    assert!(
        !msg.contains("SECRET_TOKEN_EPSILON"),
        "secret token leaked into error for {url:?}: {msg}"
    );
    assert!(
        !msg.contains("erin"),
        "username leaked into error for {url:?}: {msg}"
    );
    assert!(
        msg.contains("TLS") && msg.contains("http://"),
        "rejection must be actionable (TLS unsupported, http:// alternative): {msg}"
    );
}

#[test]
fn stage3_toxicflow_2_missing_scheme_with_credentials_leaks_nothing() {
    assert_no_secret(
        "frank:SECRET_TOKEN_ZETA@proxy:8080",
        "SECRET_TOKEN_ZETA",
        "frank",
    );
}

#[test]
fn stage3_toxicflow_2_https_without_credentials_rejected_actionably() {
    let err = ProxyConfig::parse(Some("https://proxy:8443"))
        .expect_err("https:// proxy URL must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("TLS") && msg.contains("http://"),
        "rejection must be actionable (TLS unsupported, http:// alternative): {msg}"
    );
}

#[test]
fn stage3_toxicflow_2_http_scheme_table_positive_controls() {
    // Supported schemes keep working, with and without credentials.
    assert!(matches!(
        ProxyConfig::parse(Some("http://proxy:8080")),
        Ok(ProxyConfig::Http {
            host,
            port: 8080,
            auth: None
        }) if host == "proxy"
    ));
    assert!(matches!(
        ProxyConfig::parse(Some("http://user:pass@proxy:8080")),
        Ok(ProxyConfig::Http {
            host,
            port: 8080,
            auth: Some(_)
        }) if host == "proxy"
    ));
    assert_eq!(ProxyConfig::parse(None), Ok(ProxyConfig::Direct));
}
