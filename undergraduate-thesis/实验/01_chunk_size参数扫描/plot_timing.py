import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DIR = Path(__file__).parent
df = pd.read_csv(DIR / "timing_stats.csv")

# 从 model 名称提取 chunk_size
df["chunk_size"] = df["model"].str.extract(r"cs(\d+)").astype(int)

# 计算衍生指标
df["fps"] = df["frames"] / df["episode_total_s"]
df["obs_ms_per_frame"]       = df["obs_s"]       / df["frames"] * 1000
df["inference_ms_per_frame"] = df["inference_s"] / df["frames"] * 1000
df["action_ms_per_frame"]    = df["action_s"]    / df["frames"] * 1000
df["wait_ms_per_frame"]      = df["wait_s"]      / df["frames"] * 1000

# 按 chunk_size 聚合（取均值）
agg = df.groupby("chunk_size")[
    ["obs_pct", "inference_pct", "action_pct", "wait_pct", "fps",
     "obs_ms_per_frame", "inference_ms_per_frame", "action_ms_per_frame", "wait_ms_per_frame"]
].mean().reset_index()
agg = agg.sort_values("chunk_size")

PHASE_COLORS = {
    "obs":       "#4C72B0",
    "inference": "#DD8452",
    "action":    "#55A868",
    "wait":      "#C44E52",
}
XTICKS = agg["chunk_size"].tolist()
XLABELS = [str(x) for x in XTICKS]


# ── 图1：时间占比折线图 ────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 5))

pct_cols = {"obs_pct": "obs", "inference_pct": "inference",
            "action_pct": "action", "wait_pct": "wait"}
for col, label in pct_cols.items():
    color = PHASE_COLORS[label]
    ax1.plot(agg["chunk_size"], agg[col], marker="o", linewidth=2,
             color=color, label=label)
    for _, row in agg.iterrows():
        ax1.annotate(f"{row[col]:.1f}%", (row["chunk_size"], row[col]),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8, color=color)

ax1.set_xlabel("Chunk Size", fontsize=11)
ax1.set_ylabel("Time %", fontsize=11)
ax1.set_title("Time breakdown by phase vs Chunk Size (mean over 20 episodes)", fontsize=13)
ax1.set_xticks(XTICKS)
ax1.set_xticklabels(XLABELS)
ax1.legend(loc="center right", fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_ylim(0, 110)

fig1.tight_layout()
out1 = DIR / "chart1_time_pct.png"
fig1.savefig(out1, dpi=150)
print(f"saved {out1}")


# ── 图2：FPS ──────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(agg["chunk_size"], agg["fps"], marker="s", linewidth=2, color="#8172B2")
for _, row in agg.iterrows():
    ax2.annotate(f"{row['fps']:.1f}", (row["chunk_size"], row["fps"]),
                 textcoords="offset points", xytext=(0, 6),
                 ha="center", fontsize=9)

ax2.set_xlabel("Chunk Size", fontsize=11)
ax2.set_ylabel("FPS (frames / s)", fontsize=11)
ax2.set_title("FPS vs Chunk Size", fontsize=13)
ax2.set_xticks(XTICKS)
ax2.set_xticklabels(XLABELS)
ax2.grid(True, linestyle="--", alpha=0.5)

fig2.tight_layout()
out2 = DIR / "chart2_fps.png"
fig2.savefig(out2, dpi=150)
print(f"saved {out2}")


# ── 图3：每帧绝对时间（log scale）────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))

ms_cols = {
    "obs_ms_per_frame":       "obs",
    "inference_ms_per_frame": "inference",
    "action_ms_per_frame":    "action",
    "wait_ms_per_frame":      "wait",
}
for col, label in ms_cols.items():
    color = PHASE_COLORS[label]
    ax3.plot(agg["chunk_size"], agg[col], marker="o", linewidth=2,
             color=color, label=label)
    for _, row in agg.iterrows():
        val = row[col]
        ax3.annotate(f"{val:.1f}", (row["chunk_size"], val),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8, color=color)

ax3.set_yscale("log")
ax3.set_xlabel("Chunk Size", fontsize=11)
ax3.set_ylabel("ms / frame  (log scale)", fontsize=11)
ax3.set_title("Absolute time per frame by phase vs Chunk Size", fontsize=13)
ax3.set_xticks(XTICKS)
ax3.set_xticklabels(XLABELS)
ax3.legend(loc="upper right", fontsize=9)
ax3.grid(True, linestyle="--", alpha=0.4, which="both")

fig3.tight_layout()
out3 = DIR / "chart3_ms_per_frame.png"
fig3.savefig(out3, dpi=150)
print(f"saved {out3}")
