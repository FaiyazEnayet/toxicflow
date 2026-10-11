#!/usr/bin/env python3
"""Animated GIF for the README, generated from real `toxicflow export` output.

Shows the midprice and VPIN building up over the recorded session.
Never faked: every frame comes from the exported CSVs.

Usage:
    ./target/debug/toxicflow export data/session_325h --out runs/dashboard_data
    python3 scripts/make_demo_gif.py --data runs/dashboard_data --out docs/demo.gif
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/dashboard_data")
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--frames", type=int, default=72)
    args = ap.parse_args()

    ofi = pd.read_csv(os.path.join(args.data, "ofi.csv"))
    vpin = pd.read_csv(os.path.join(args.data, "vpin.csv"))
    ofi["t"] = (ofi["time_ms"] - ofi["time_ms"].iloc[0]) / 3600e3
    vpin["t"] = (vpin["time_ms"] - vpin["time_ms"].iloc[0]) / 3600e3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.suptitle("toxicflow: BTCUSDT order-flow toxicity, real 3.25h session")
    ax1.set_ylabel("midprice")
    ax2.set_ylabel("VPIN")
    ax2.set_xlabel("hours into session")
    ax2.axhline(0.7, color="red", ls="--", lw=1, label="gate 0.7")
    ax2.set_ylim(0, 0.85)
    ax2.legend(loc="upper left", fontsize=8)
    (ln1,) = ax1.plot([], [], lw=1.2, color="#1f77b4")
    (ln2,) = ax2.plot([], [], lw=1.5, color="#9467bd")
    pt1, = ax1.plot([], [], "o", ms=4, color="#1f77b4")
    pt2, = ax2.plot([], [], "o", ms=4, color="#9467bd")
    hours = ofi["t"].iloc[-1]
    ax1.set_xlim(0, hours)
    ax1.set_ylim(ofi["mid"].min() - 5, ofi["mid"].max() + 5)

    n = len(ofi)
    step = max(1, n // args.frames)

    def frame(i):
        k = min(n, (i + 1) * step)
        t, m = ofi["t"].iloc[:k], ofi["mid"].iloc[:k]
        ln1.set_data(t, m)
        pt1.set_data([t.iloc[-1]], [m.iloc[-1]])
        vv = vpin[vpin["t"] <= t.iloc[-1]]
        ln2.set_data(vv["t"], vv["vpin"])
        if len(vv):
            pt2.set_data([vv["t"].iloc[-1]], [vv["vpin"].iloc[-1]])
        return ln1, ln2, pt1, pt2

    anim = FuncAnimation(fig, frame, frames=args.frames, blit=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim.save(args.out, writer="pillow", fps=9)
    print(f"gif written -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
