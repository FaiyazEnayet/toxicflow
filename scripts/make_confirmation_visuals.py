"""Render the README confirmation figure and animated evidence walkthrough."""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DOCS = Path(__file__).resolve().parents[1] / "docs"
BG, INK, MUTED, GRID = "#17191b", "#ede8db", "#b8b6ad", "#41474b"
COPPER, RED, GREEN = "#e5a17e", "#e87670", "#b4c4be"


def make_figure(title, subtitle):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=100, facecolor=BG)
    fig.subplots_adjust(left=.105, right=.96, top=.66, bottom=.20, wspace=.32)
    fig.text(.064, .92, "TOXICFLOW  /  DIRECTION CONFIRMATION", color=COPPER,
             fontsize=10, fontweight="bold")
    fig.text(.064, .822, title, color=INK, fontsize=24, fontweight="bold")
    fig.text(.064, .75, subtitle, color=MUTED, fontsize=12)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_axisbelow(True)
    return fig, axes


def session_figure(report, shown):
    pooled = report["pooled"]
    rows = report["per_session"]
    slots = [row["slot"] for row in rows]
    fig, axes = make_figure(
        "The model beats the training-majority baseline",
        f"Held-out accuracy {pooled['accuracy']:.4f} vs {pooled['baseline_accuracy']:.4f}"
        f"   ·   {pooled['n_scored']:,} scored rows",
    )
    for ax, field, title, floor, accent, fmt in zip(
        axes, ("delta", "balanced_accuracy"), ("Paired difference (Δ)", "Balanced accuracy"),
        (0, .5), (COPPER, GREEN), ("+.4f", ".4f")
    ):
        values = [row[field] for row in rows]
        data = values[:shown]
        ax.set_title(title, loc="left", color=INK, fontsize=14, pad=12)
        ax.set_xticks(slots)
        ax.set_xlim(slots[0] - .6, slots[-1] + .6)
        ax.set_ylim(min(floor, *values) - .008, max(values) + .010)
        ax.grid(axis="y", color=GRID, alpha=.55)
        ax.axhline(floor, color=MUTED, linewidth=.9)
        ax.axhline(pooled[field], color=accent, linestyle="--", linewidth=1.25)
        if field == "delta":
            ax.bar(slots[:shown], data, width=.62, color=[COPPER if v >= 0 else RED for v in data])
        else:
            ax.plot(slots[:shown], data, color=GREEN, marker="o", linewidth=1.6)
        ax.set_xlabel("Campaign slot", color=MUTED, labelpad=9)
        ax.text(1, 1.055, "Pooled " + format(pooled[field], fmt), color=accent,
                transform=ax.transAxes, ha="right", fontsize=10)
    positive = sum(row["delta"] > 0 for row in rows)
    footer = f"Δ positive in {positive} of {len(rows)} sealed sessions   ·   Dashed lines mark the pooled results."
    fig.text(.064, .065, footer, color=MUTED, fontsize=10)
    return fig


def interval_figure(report):
    fig, axes = make_figure(
        "The study did not establish a joint win",
        "Paired grouped bootstrap intervals at the two registered block lengths",
    )
    for ax, field, key, title in zip(
        axes, ("delta", "gamma"), ("delta_ci", "gamma_ci"),
        ("Paired difference (Δ)", "Balanced accuracy above 0.50 (Γ)")
    ):
        ax.set_title(title, color=INK, loc="left", fontsize=14, pad=12)
        point = report["pooled"][field]
        ax.set_yticks([1, 0], ["300 s", "600 s"])
        ax.set_ylim(-.58, 1.65)
        ax.axvline(0, color=MUTED, linestyle=":", linewidth=1)
        ax.grid(axis="x", color=GRID, alpha=.55)
        for y, length in [(1, 300), (0, 600)]:
            low, high = report[f"bootstrap_{length}s"][key]
            ax.errorbar(point, y, xerr=[[point-low], [high-point]], fmt="o",
                        color=COPPER if field == "delta" else GREEN, capsize=5)
            ax.annotate(f"[{low:+.4f}, {high:+.4f}]", (point, y), xytext=(0, 12),
                        textcoords="offset points", ha="center", color=INK, fontsize=10)
    fig.text(.064, .065, "The balanced accuracy gate cleared. Both Δ intervals include zero.",
             color=MUTED, fontsize=10)
    return fig


def frame(fig):
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=80, facecolor=BG)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB").quantize(colors=192)
    plt.close(fig)
    return image


def main():
    report = json.loads((DOCS / "direction_confirmation_report.json").read_text(encoding="utf8"))
    count = len(report["per_session"])
    final = session_figure(report, count)
    final.savefig(DOCS / "results.png", facecolor=BG)
    frames = [frame(session_figure(report, index)) for index in range(1, count)]
    frames.extend([frame(final), frame(interval_figure(report))])
    frames[0].save(DOCS / "demo.gif", save_all=True, append_images=frames[1:],
                   duration=[200] * (count - 1) + [2600, 3200], loop=0, optimize=True)


if __name__ == "__main__":
    main()
