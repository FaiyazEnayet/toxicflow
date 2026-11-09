#!/usr/bin/env python3
"""Interactive Plotly dashboard for a toxicflow recorded session.

Reads the CSV/JSON outputs of `toxicflow export` and builds a single
self-contained HTML file (plotly.js is embedded, no server needed).

Usage:
    ./target/debug/toxicflow export data/session_325h --out runs/dashboard_data
    python3 scripts/dashboard.py --data runs/dashboard_data --out docs/dashboard.html --speed-report docs/info_speed_report.json
"""

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load(data_dir):
    vpin = pd.read_csv(os.path.join(data_dir, "vpin.csv"))
    kyle = pd.read_csv(os.path.join(data_dir, "kyle.csv"))
    ofi = pd.read_csv(os.path.join(data_dir, "ofi.csv"))
    cal = pd.read_csv(os.path.join(data_dir, "calibration.csv"))
    with open(os.path.join(data_dir, "summary.json")) as f:
        summary = json.load(f)
    for df in (vpin, kyle, ofi):
        df["time"] = pd.to_datetime(df["time_ms"], unit="ms")
    return vpin, kyle, ofi, cal, summary


def information_speed(data_dir, report_path, *, required):
    """Match the fresh export to the published, full-precision response curve."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    session = next(row for row in report["sessions"] if row["session"] == "session_325h")
    with open(os.path.join(data_dir, "summary.json"), encoding="utf-8") as f:
        exported = json.load(f).get("speed")
    horizons = session["horizons_s"]
    response = session["response_bps"]
    lo, hi = map(list, zip(*session["response_ci_bps"]))
    same_session = (
        exported and exported["horizons_s"] == horizons
        and len(exported["response_bps"]) == len(response)
        and all(math.isclose(a, b, abs_tol=1e-10)
                for a, b in zip(exported["response_bps"], response))
    )
    if not same_session:
        if required:
            raise ValueError("Export does not match the published session_325h response")
        return None, None
    csv = pd.read_csv(os.path.join(data_dir, "speed.csv"))
    if (csv["horizon_s"].tolist() != horizons
            or csv["n_origins"].tolist() != session["n_origins"]
            or any(abs(a - b) > 0.00005000001
                   for name, exact in (("response_bps", response), ("ci_lo_bps", lo), ("ci_hi_bps", hi))
                   for a, b in zip(csv[name], exact, strict=True))):
        raise ValueError("Export does not match the published session_325h information-speed curve")

    summary = report["summary"]
    best = max(report["sessions"], key=lambda row: row["fit_r_squared"])
    threshold = 0.85  # Headline threshold documented in the report and SpeedConfig.
    failed = summary["n_sessions"] - summary["n_fit_ok"]
    verdict = (f"Exponential adjustment fit rejected in {failed} of {summary['n_sessions']} sessions. "
               f"Best raw R² {best['fit_r_squared']:.3f} against a {threshold:.2f} threshold.")
    if "tau is not bounded" in (best["fit_failure"] or ""):
        verdict += " The best fit cleared the threshold, but its bootstrap did not bound tau."
    if not summary["n_fit_ok"]:
        verdict += " No tau or half-life is reported."
    highlights = " · ".join(f"{h} s: {response[horizons.index(h)]:+.2f} bps" for h in (0, 60, 300))
    fig = go.Figure([
        go.Scatter(x=horizons, y=lo, mode="lines", line=dict(width=0),
                   showlegend=False, hoverinfo="skip"),
        go.Scatter(x=horizons, y=hi, mode="lines", line=dict(width=0),
                   fill="tonexty", fillcolor="rgba(216,148,112,.24)",
                   name="Pointwise bootstrap interval", hoverinfo="skip"),
        go.Scatter(x=horizons, y=response, mode="lines+markers",
                   name="Cumulative response", customdata=list(zip(lo, hi, session["n_origins"])),
                   hovertemplate=("%{x} s: %{y:+.4f} bps<br>Bootstrap: "
                                  "[%{customdata[0]:+.4f}, %{customdata[1]:+.4f}] bps"
                                  "<br>%{customdata[2]} origins<extra></extra>")),
    ])
    fig.update_layout(xaxis_title="Horizon (seconds)", yaxis_title="Cumulative response (bps)",
                      showlegend=True, meta=dict(verdict=verdict, highlights=highlights))
    return fig, verdict


def build(data_dir):
    vpin, kyle, ofi, cal, s = load(data_dir)
    gate = s["vpin"]["gate_threshold"]

    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Midprice, 10s bars",
            "Signed trade volume per bar",
            "VPIN over time (equal-volume buckets)",
            "Kyle's lambda, rolling 60-bar OLS",
            "OFI vs next 30s mid return",
            "Predictor calibration (deciles)",
            "Toxicity-gated market-making demo",
            "Session summary",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    # Row 1: price and signed volume
    fig.add_trace(
        go.Scatter(
            x=ofi["time"], y=ofi["mid"], mode="lines", name="midprice",
            line=dict(color="#1f77b4", width=1.2),
            hovertemplate="%{x}<br>mid %{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ofi["signed_vol"]]
    fig.add_trace(
        go.Bar(
            x=ofi["time"], y=ofi["signed_vol"], name="signed vol",
            marker_color=colors, opacity=0.6,
            hovertemplate="%{x}<br>signed vol %{y:.3f} BTC<extra></extra>",
        ),
        row=1, col=2,
    )

    # Row 2: VPIN with gate threshold, Kyle's lambda
    fig.add_trace(
        go.Scatter(
            x=vpin["time"], y=vpin["vpin"], mode="lines+markers", name="VPIN",
            line=dict(color="#9467bd", width=1.5), marker=dict(size=4),
            hovertemplate="%{x}<br>VPIN %{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(
        y=gate, line_dash="dash", line_color="#d62728", row=2, col=1,
        annotation_text=f"gate {gate}", annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=kyle["time"], y=kyle["lambda"], mode="lines", name="lambda",
            line=dict(color="#ff7f0e", width=1.2),
            hovertemplate="%{x}<br>lambda %{y:.3f}<extra></extra>",
        ),
        row=2, col=2,
    )

    # Row 3: OFI vs forward return scatter + binned means
    fwd = ofi.dropna(subset=["fwd_ret_bps"])
    fig.add_trace(
        go.Scatter(
            x=fwd["ofi"], y=fwd["fwd_ret_bps"], mode="markers", name="bars",
            marker=dict(size=3, opacity=0.35, color="#1f77b4"),
            hovertemplate="OFI %{x:.2f}<br>fwd ret %{y:.2f} bps<extra></extra>",
        ),
        row=3, col=1,
    )
    fwd = fwd.copy()
    fwd["ofi_bin"] = pd.qcut(fwd["ofi"], 10, duplicates="drop")
    binned = fwd.groupby("ofi_bin", observed=True).agg(
        ofi_mid=("ofi", "mean"), ret_mean=("fwd_ret_bps", "mean")
    ).sort_values("ofi_mid")
    fig.add_trace(
        go.Scatter(
            x=binned["ofi_mid"], y=binned["ret_mean"], mode="lines+markers",
            name="binned mean", line=dict(color="#d62728", width=2.5),
            marker=dict(size=7),
            hovertemplate="OFI %{x:.2f}<br>mean fwd ret %{y:.2f} bps<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cal["mean_pred"], y=cal["realized"], mode="markers", name="deciles",
            marker=dict(size=[max(3, min(14, n / 8)) for n in cal["n"]], color="#2ca02c", opacity=0.8),
            hovertemplate="pred %{x:.3f}<br>realized %{y:.3f}<br>n=%{customdata}<extra></extra>",
            customdata=cal["n"],
        ),
        row=3, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="perfect",
            line=dict(dash="dash", color="gray"),
        ),
        row=3, col=2,
    )

    # Row 4: MM demo bars + summary table
    mm = s["mm"]
    fig.add_trace(
        go.Bar(
            x=["quoting always", "gated on VPIN"],
            y=[mm["pnl_open"], mm["pnl_gated"]],
            name="PnL", marker_color=["#1f77b4", "#9467bd"],
            text=[f'{mm["pnl_open"]:+.2f}', f'{mm["pnl_gated"]:+.2f}'],
            textposition="outside",
            hovertemplate="%{x}<br>PnL %{y:+.4f}<extra></extra>",
        ),
        row=4, col=1,
    )
    gated_pct = 100.0 * mm["seconds_gated"] / mm["seconds"]
    rows = [
        ["Trades", f'{s["n_trades"]:,}'],
        ["Quotes", f'{s["n_quotes"]:,}'],
        ["Span", f'{s["span_s"] / 3600:.2f} h'],
        ["Volume", f'{s["volume_btc"]:.1f} BTC'],
        ["Buy share", f'{s["buy_share"]:.3f}'],
        ["Price range", f'{s["price_lo"]:,.0f} - {s["price_hi"]:,.0f}'],
        ["VPIN mean / max", f'{s["vpin"]["mean"]:.3f} / {s["vpin"]["max"]:.3f}'],
        ["Kyle lambda mean", f'{s["kyle_lambda"]["mean"]:.3f}'],
        ["OFI mean +- std", f'{s["ofi"]["mean"]:+.2f} +- {s["ofi"]["std"]:.2f}'],
        ["Predictor acc vs base", f'{s["predict"]["accuracy"]:.3f} vs {s["predict"]["baseline"]:.3f}'],
        ["MM gated time", f'{gated_pct:.0f}%'],
        ["MM turnover", f'{mm["turnover"]:,.0f}'],
    ]
    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], fill_color="#e8e8e8", align="left"),
            cells=dict(values=list(zip(*rows)), align="left"),
        ),
        row=4, col=2,
    )

    fig.update_layout(
        title="toxicflow: order-flow toxicity on a real BTCUSDT session",
        height=1500,
        showlegend=False,
        template="plotly_white",
    )
    fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=1, col=2)
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=2, col=2)
    fig.update_yaxes(title_text="VPIN", row=2, col=1)
    fig.update_yaxes(title_text="price / BTC", row=2, col=2)
    fig.update_yaxes(title_text="fwd ret (bps)", row=3, col=1)
    fig.update_xaxes(title_text="OFI per 10s bar", row=3, col=1)
    fig.update_yaxes(title_text="realized", row=3, col=2)
    fig.update_xaxes(title_text="predicted", row=3, col=2)
    fig.update_yaxes(title_text="PnL", row=4, col=1)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/dashboard_data")
    ap.add_argument("--speed-report", type=Path, help="Published report for the main session export")
    ap.add_argument("--out", default="docs/dashboard.html")
    args = ap.parse_args()
    report = args.speed_report or Path(__file__).resolve().parents[1] / "docs/info_speed_report.json"
    speed, verdict = information_speed(args.data, report, required=args.speed_report is not None)
    if speed is None:
        fig = build(args.data)
    else:
        # Preserve the previously published panels and their evaluation protocol.
        # Only the response curve and its summary row use the new export.
        panels_path = Path(__file__).resolve().parents[1] / "portfolio_ui/main_session_panels.json"
        panels = json.loads(panels_path.read_text(encoding="utf-8"))
        cells = next(trace["cells"]["values"] for trace in panels["data"] if trace["type"] == "table")
        cells[0].append("Information speed")
        cells[1].append(verdict)
        fig = go.Figure(panels)
    raw = fig.to_html(include_plotlyjs=True, full_html=True)
    if speed is not None:
        raw += speed.to_html(include_plotlyjs=False, full_html=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(_portfolio_html(raw))
    print(f"dashboard written -> {args.out}")



def _portfolio_html(raw):
    """Presentation-only transform of this run’s freshly calculated figures."""
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "portfolio_ui"))
    from renderer import render_html
    return render_html(root, 'toxicflow', raw)[0]

if __name__ == "__main__":
    main()
