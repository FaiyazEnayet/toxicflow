//! Stage 3 toxicflow H2: proxy error messages must never disclose the
//! supplied URL or its credentials, and `https://` proxy URLs must be
//! rejected because the CONNECT socket is plaintext TCP.
//!
//! Scheme table (documented on `ProxyConfig::parse`):
//! - `http://`  supported: plaintext TCP CONNECT tunnel; TLS is
//!   established to the destination after CONNECT.
//! - `https://` rejected: TLS to the proxy is not supported.
//! - anything else rejected: unsupported scheme.

use toxicflow::metrics::{Quote, Trade};
use toxicflow::net::ProxyConfig;
use toxicflow::predict::{
    build_dataset, build_p2_dataset, p2_grid_from_json, preregistered_p2_grid, walk_forward_p2,
    P2Control, P2_B3_COLS, P2_B5_COLS,
};

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

// ---- Stage 3 P2: normalized flow features -------------------------

/// 40 minutes of synthetic quotes/trades with predictable flow:
/// drift regimes last 60 s and the bid stack grows before up moves,
/// so the direction signal is real but the sample is small.
fn p2_synth_session() -> (Vec<Quote>, Vec<Trade>) {
    let mut quotes = Vec::new();
    let mut trades = Vec::new();
    let mut mid = 100.0;
    let mut s: u64 = 42;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64) / (u64::MAX as f64) - 0.5
    };
    for i in 0..2400 {
        let t = i as i64 * 1000;
        let drift = if i % 120 < 60 { 0.02 } else { -0.02 };
        let bq = 5.0 + if drift > 0.0 { 4.0 } else { rnd() };
        let aq = 5.0 + if drift < 0.0 { 4.0 } else { rnd() };
        quotes.push(Quote {
            time_ms: t,
            bid_px: mid - 0.05,
            bid_sz: bq.max(0.5),
            ask_px: mid + 0.05,
            ask_sz: aq.max(0.5),
        });
        mid += drift + rnd() * 0.01;
        trades.push(Trade {
            time_ms: t + 100,
            price: mid,
            qty: 0.1,
            is_buyer_maker: drift <= 0.0,
        });
    }
    (quotes, trades)
}

#[test]
fn stage3_toxicflow_2_p2_ablation_runs_on_synthetic_session() {
    // End to end: the v4 frame builds, the B3 and B5 evaluations
    // run on the same rows, and every headline number is finite.
    let (quotes, trades) = p2_synth_session();
    let v3 = build_dataset(&quotes, &trades, 10_000, 3);
    let p2 = build_p2_dataset(&quotes, &trades, 10_000, 3);
    assert!(!v3.is_empty() && !p2.is_empty());
    assert!(p2.len() < v3.len(), "v4 rows start at bar 29, v3 earlier");
    let b3 = walk_forward_p2(&p2, P2_B3_COLS, P2Control::Off, 4, 3).expect("b3 eval failed");
    let b5 = walk_forward_p2(&p2, P2_B5_COLS, P2Control::Off, 4, 3).expect("b5 eval failed");
    assert_eq!(b3.n_oos, b5.n_oos, "B3 and B5 must score the same rows");
    assert_eq!(b3.n_labeled, p2.len());
    for (label, ev) in [("b3", &b3), ("b5", &b5)] {
        assert!(ev.accuracy.is_finite(), "{label} accuracy not finite");
        assert!(ev.baseline.is_finite(), "{label} baseline not finite");
        assert!(ev.brier.is_finite(), "{label} brier not finite");
        assert!(ev.log_loss.is_finite(), "{label} log loss not finite");
        assert_eq!(ev.fold_models.len(), 4, "{label} must fit 4 folds");
        assert_eq!(ev.n_pos_oos + ev.n_neg_oos, ev.n_oos);
        assert_eq!(ev.n_oos + ev.n_unscored, ev.n_labeled);
    }
    // The permutation control scores the same rows with the P2
    // block shuffled, so its denominator matches the clean run.
    let perm =
        walk_forward_p2(&p2, P2_B5_COLS, P2Control::Permute, 4, 3).expect("perm eval failed");
    assert_eq!(perm.n_oos, b5.n_oos);
    // The shift control drops the 5 warm-up rows of each fold.
    let shift =
        walk_forward_p2(&p2, P2_B5_COLS, P2Control::Shift, 4, 3).expect("shift eval failed");
    assert_eq!(shift.n_oos, b5.n_oos - 20);
}

#[test]
fn stage3_toxicflow_2_p2_grid_file_matches_library() {
    // The committed grid file must match the preregistered library
    // grid exactly; the runner refuses to run otherwise.
    let text = std::fs::read_to_string("experiments/p2/grid.json").expect("grid.json missing");
    let file_grid = p2_grid_from_json(&text).expect("grid.json must parse");
    assert_eq!(file_grid, preregistered_p2_grid());
}
