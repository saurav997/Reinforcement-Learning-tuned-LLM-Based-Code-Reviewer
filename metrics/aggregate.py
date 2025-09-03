# metrics/aggregate.py
import os, glob, json, time, jsonlines
from pathlib import Path

OUT_DIR = Path("metrics")
DOCS_CHARTS = Path("docs/charts")
TS_JSON = OUT_DIR / "timeseries.json"

def now_iso(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def load_json(path):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)

def stars_mean_from_logs(limit=500):
    files = sorted(glob.glob("logs/events-*.jsonl")) or (["logs/events.jsonl"] if os.path.exists("logs/events.jsonl") else [])
    stars = []
    for fp in files[::-1]:
        try:
            for e in jsonlines.open(fp):
                if e.get("type") == "rating" and "stars" in e:
                    stars.append(e["stars"])
                    if len(stars) >= limit: break
        except Exception:
            continue
        if len(stars) >= limit: break
    if not stars: return None
    return sum((s - 1) / 4 for s in stars) / len(stars)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_CHARTS.mkdir(parents=True, exist_ok=True)

    ts = []
    if TS_JSON.exists():
        with open(TS_JSON) as f: ts = json.load(f)

    eval_report = load_json("eval/report.json") or {}
    rm_report = load_json("reward_model/rm_report.json") or {}

    entry = {
        "timestamp": eval_report.get("timestamp") or now_iso(),
        "commit": eval_report.get("commit") or os.getenv("GITHUB_SHA", "local"),
        "model_tag": eval_report.get("model_tag", "policy@base+rm@unknown"),
        "win_rate": eval_report.get("win_rate"),
        "reward_mean": eval_report.get("reward_mean"),
        "rougeL_mean": eval_report.get("rougeL_mean"),
        "lint_delta_mean": eval_report.get("lint_delta_mean"),
        "rm_auc": rm_report.get("auc"),
        "rm_loss_bce": rm_report.get("loss_bce"),
        "rm_loss_rank": rm_report.get("loss_rank"),
        "stars_mean": stars_mean_from_logs(),
    }

    # Avoid duplicate commits in timeseries
    existing_idx = next((i for i, x in enumerate(ts) if x.get("commit") == entry["commit"]), None)
    if existing_idx is not None:
        ts[existing_idx] = entry
    else:
        ts.append(entry)

    with open(TS_JSON, "w") as f: json.dump(ts, f, indent=2)
    print(f"Updated {TS_JSON} with {len(ts)} points.")

if __name__ == "__main__":
    main()
