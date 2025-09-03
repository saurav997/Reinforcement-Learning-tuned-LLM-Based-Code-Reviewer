# metrics/plot_curves.py
import json, os
from pathlib import Path
import matplotlib.pyplot as plt

TS_JSON = Path("metrics/timeseries.json")
DOCS_CHARTS = Path("docs/charts")
DOCS_CHARTS.mkdir(parents=True, exist_ok=True)

def load_ts():
    if not TS_JSON.exists(): return []
    with open(TS_JSON) as f: return json.load(f)

def line(xs, ys, title, ylabel, outpng):
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("checkpoints")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpng, dpi=160)
    plt.close()

def main():
    ts = load_ts()
    if not ts:
        print("No timeseries yet."); return

    xs = [t.get("model_tag", "") for t in ts]
    def getv(k): return [t.get(k) for t in ts]

    line(xs, getv("win_rate"),       "Win-rate vs Baseline", "win_rate",       DOCS_CHARTS/"winrate.png")
    line(xs, getv("reward_mean"),    "Mean Total Reward",    "reward_mean",    DOCS_CHARTS/"reward.png")
    line(xs, getv("rm_auc"),         "Reward Model AUC",     "rm_auc",         DOCS_CHARTS/"rm_auc.png")
    line(xs, getv("stars_mean"),     "Avg Stars (Normalized)","stars_mean",     DOCS_CHARTS/"stars.png")

    # Optional: also plot ROUGE-L and Lint Δ
    line(xs, getv("rougeL_mean"),    "ROUGE-L (Eval)",       "rougeL_mean",    DOCS_CHARTS/"rougeL.png")
    line(xs, getv("lint_delta_mean"),"Lint Δ (Eval)",        "lint_delta_mean",DOCS_CHARTS/"lint_delta.png")

    print("Wrote charts to docs/charts/")

if __name__ == "__main__":
    main()
