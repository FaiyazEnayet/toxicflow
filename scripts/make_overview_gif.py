"""Animate the recorded session, campaign response and held-out evaluation."""
import io
import json
from datetime import datetime
from pathlib import Path

from make_confirmation_visuals import BG, COPPER, GREEN, GRID, INK, MUTED, plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scene(index, category, title, subtitle, footer):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), dpi=100, facecolor=BG)
    fig.subplots_adjust(left=.14, right=.95, top=.65, bottom=.16, hspace=.82)
    fig.text(.07, .925, "TOXICFLOW  /  " + category, color=COPPER, fontsize=11, weight="bold")
    fig.text(.93, .925, f"{index} / 3", ha="right", color=MUTED, fontsize=10)
    fig.text(.07, .815, title, color=INK, fontsize=26, weight="bold")
    fig.text(.07, .74, subtitle, color=MUTED, fontsize=11)
    fig.text(.07, .045, footer, color=MUTED, fontsize=10)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
        ax.spines[:].set_visible(False)
        ax.grid(axis="y", color=GRID, alpha=.65)
        ax.set_axisbelow(True)
    return fig, axes


def label(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=9)


def draw_frames(fig, update, count=15):
    frames = []
    for progress in range(1, count + 1):
        update(progress / count)
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="png", facecolor=BG)
            buffer.seek(0)
            frames.append(Image.open(buffer).convert("RGB").quantize(colors=192))
    plt.close(fig)
    return frames


def session_scene():
    traces = read_json(ROOT / "portfolio_ui/main_session_panels.json")["data"]
    by_name = {trace.get("name"): trace for trace in traces}
    price, vpin = by_name["midprice"], by_name["VPIN"]
    start = datetime.fromisoformat(price["x"][0])
    fig, axes = scene(1, "SESSION REPLAY", "Follow the order book",
                      "Recorded BTCUSDT · midprice and trade-flow imbalance on the same clock",
                      "Explore signed volume, Kyle’s lambda and toxicity-gated market-making in the dashboard.")
    end = (datetime.fromisoformat(price["x"][-1]) - start).total_seconds() / 3600
    series = []
    for ax, trace, name, color in zip(axes, (price, vpin), ("Midprice · USDT", "VPIN"), (GREEN, COPPER)):
        hours = [(datetime.fromisoformat(t) - start).total_seconds() / 3600 for t in trace["x"]]
        values = trace["y"]
        label(ax, name)
        ax.set_xlim(0, end)
        padding = (max(values) - min(values)) * .1
        ax.set_ylim(min(values) - padding, max(values) + padding)
        line, = ax.plot([], [], color=color, linewidth=1.5)
        series.append((line, hours, values))
    axes[-1].set_xlabel("Hours into the session", color=MUTED)

    def update(progress):
        for line, hours, values in series:
            stop = sum(t <= end * progress for t in hours)
            line.set_data(hours[:stop], values[:stop])

    return draw_frames(fig, update)


def response_scene():
    report = read_json(DOCS / "info_speed_report_20x4h.json")
    pooled = report["variants"]["pooled"]
    fig, axes = scene(2, "RESPONSE ANALYSIS", "Trace the price response",
                      f"{pooled['n_sessions']} BTCUSDT sessions · cumulative response to an order-flow imbalance shock",
                      "Basis points per one-standard-deviation change in order-flow imbalance.")
    horizons = pooled["horizons_s"]
    series = []
    for ax, field, title, color in zip(
        axes, ("equal_weight_response_bps", "precision_weighted_response_bps"),
        ("Equal weight per session · bps", "Precision-weighted response · bps"), (COPPER, GREEN)
    ):
        values = pooled[field]
        label(ax, title)
        ax.set_xlim(horizons[0], horizons[-1])
        ax.set_ylim(0, max(values) * 1.18)
        line, = ax.plot([], [], color=color, marker="o", markersize=3, linewidth=1.6)
        series.append((line, values))
    axes[-1].set_xlabel("Seconds after the shock bar begins", color=MUTED)

    def update(progress):
        stop = max(2, round(len(horizons) * progress))
        for line, values in series:
            line.set_data(horizons[:stop], values[:stop])

    return draw_frames(fig, update)


def confirmation_scene():
    report = read_json(DOCS / "direction_confirmation_report.json")
    pooled, sessions = report["pooled"], report["per_session"]
    positive = sum(row["delta"] > 0 for row in sessions)
    fig, axes = scene(3, "HELD-OUT EVALUATION", "Test on sealed sessions",
                      f"{len(sessions)} BTCUSDT recordings held out from training · {pooled['n_scored']:,} scored rows",
                      f"Accuracy improved in {positive} of {len(sessions)} sessions. Full intervals are in the dashboard.")
    accuracy = axes[0]
    label(accuracy, "Pooled accuracy · model and training-majority baseline")
    accuracy.set_xlim(0, 1)
    accuracy.set_ylim(-.7, 1.7)
    accuracy.set_yticks([])
    accuracy.set_xticks([])
    scores = [pooled["accuracy"], pooled["baseline_accuracy"]]
    bars = accuracy.barh([1, 0], [0, 0], color=[COPPER, GREEN], height=.48)
    labels = [accuracy.text(0, y, text, va="center", fontsize=10, color=INK, visible=False) for y, text in zip(
        [1, 0], [f"Model  {scores[0]:.4f}", f"Baseline  {scores[1]:.4f}"]
    )]
    balance = axes[1]
    label(balance, "Balanced accuracy by held-out session")
    slots = [r["slot"] for r in sessions]
    values = [r["balanced_accuracy"] for r in sessions]
    balance.set_xlim(slots[0] - .4, slots[-1] + .4)
    balance.set_xticks(slots)
    balance.set_ylim(.5, max(values) + .01)
    balance.axhline(pooled["balanced_accuracy"], color=COPPER, linestyle="--", linewidth=1)
    line, = balance.plot([], [], color=GREEN, marker="o", markersize=4, linewidth=1.5)

    def update(progress):
        for bar, text, score in zip(bars, labels, scores):
            bar.set_width(score * progress)
            text.set_x(score + .025)
            text.set_visible(progress == 1)
        stop = max(1, round(len(slots) * progress))
        line.set_data(slots[:stop], values[:stop])

    return draw_frames(fig, update)


def main():
    frames, durations = [], []
    for build in (session_scene, response_scene, confirmation_scene):
        clip = build()
        frames.extend(clip)
        durations.extend([105] * (len(clip) - 1) + [1700])
    frames[0].save(DOCS / "demo.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)


if __name__ == "__main__":
    main()
