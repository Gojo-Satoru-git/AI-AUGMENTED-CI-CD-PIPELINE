"""Trajectory chart of a live run: the agent's trust tier over time, with the
injected fault phases shaded behind it and every wrong decision marked."""
from __future__ import annotations

from pathlib import Path

LABELS = {"none": "steady", "slow_db": "slow db", "flaky_net": "flaky net",
          "db_corruption": "corruption", "host_down": "host down",
          "code_bug": "code bug", "novel_bug": "unseen bug",
          "silent_bug": "SILENT bug"}
TIER_NAMES = ["T0 observe", "T1 recommend", "T2 act+rollback", "T3 autonomous"]


def plot_trajectory(steps, phases, path, title=None, start_tier=0, figsize=(14, 4.8)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    small = figsize[0] < 9
    for ph in phases:
        steady = ph["fault"] == "none"
        ax.axvspan(ph["first_step"] - 0.5, ph["first_step"] + ph["n"] - 0.5,
                   color="#f4f6fa" if steady else "#fde7d9", zorder=0)
        if not steady:
            ax.text(ph["first_step"] + ph["n"] / 2, 3.42, LABELS.get(ph["fault"], ph["fault"]),
                    ha="center", va="bottom", fontsize=(6.5 if small else 9),
                    color="#b03a00" if ph["fault"] != "silent_bug" else "#9b0000",
                    fontweight="bold" if ph["fault"] == "silent_bug" else "normal")

    xs = [0] + [s["step"] for s in steps]
    ys = [start_tier] + [s["tier"] for s in steps]
    ax.step(xs, ys, where="post", color="#1e6b52", linewidth=2.2, zorder=3)

    for s in steps:
        if s["tier_action"] == "promoted":
            ax.plot(s["step"], s["tier"], marker="^", color="#1e8a6e", markersize=10,
                    zorder=5, markeredgecolor="white")
        elif s["tier_action"] == "demoted":
            ax.plot(s["step"], s["tier"], marker="v", color="#c0392b", markersize=10,
                    zorder=5, markeredgecolor="white")
    miss = [s for s in steps if not s["ok"]]
    ax.scatter([s["step"] for s in miss], [-0.28] * len(miss), marker="|", s=90,
               color="#c0392b", zorder=4, label=f"wrong decision ({len(miss)})")
    unsafe = [s for s in steps if s["unsafe"]]
    ax.scatter([s["step"] for s in unsafe], [-0.28] * len(unsafe), marker="x", s=60,
               color="#7b0000", zorder=6, label=f"unsafe autonomous action ({len(unsafe)})")

    ax.set_yticks(range(4))
    ax.set_yticklabels(TIER_NAMES, fontsize=(7 if small else 10))
    ax.set_ylim(-0.5, 3.8)
    ax.set_xlim(0, max(xs) + 1)
    ax.set_xlabel("request #", fontsize=(8 if small else 10))
    ax.set_title(title or "Agent trust tier during a live run (faults injected on a schedule)",
                 fontsize=(8.5 if small else 12), fontweight="bold")
    ax.plot([], [], marker="^", color="#1e8a6e", linestyle="", label="promoted")
    ax.plot([], [], marker="v", color="#c0392b", linestyle="", label="demoted")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=(7 if small else 8), ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
