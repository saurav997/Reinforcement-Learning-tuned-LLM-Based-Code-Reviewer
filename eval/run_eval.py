# eval/run_eval.py
import os, time, json, jsonlines
from rouge_score import rouge_scorer
from app.reviewer import generate_candidates, select_best
from app.scoring import score_candidates

def now_iso(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def main():
    data = list(jsonlines.open("eval/dataset.jsonl"))
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rougeL, lint, reward = [], [], []
    for ex in data:
        code, bullets = ex["code"], ex["ref_bullets"]
        ref = "; ".join(bullets)
        cands = generate_candidates(code, k=3)
        scored = score_candidates(code, cands)
        best = select_best(scored)
        rougeL.append(rs.score(ref, best["text"])["rougeL"].fmeasure)
        lint.append(best["scores"]["lint_delta"])
        reward.append(best["total"])

    # Simple baseline win-rate: did the ranked best beat the first candidate?
    # (Approximate: compare totals)
    wins = 0
    for ex in data:
        code, bullets = ex["code"], ex["ref_bullets"]
        cands = generate_candidates(code, k=3)
        scored = score_candidates(code, cands)
        best = select_best(scored)
        wins += 1 if best["total"] > scored[0]["total"] else 0

    report = {
        "timestamp": now_iso(),
        "commit": os.getenv("GITHUB_SHA", "local"),
        "model_tag": os.getenv("POLICY_TAG", "policy@base") + "+" + os.getenv("RM_TAG", "rm@unknown"),
        "n": len(data),
        "win_rate": wins / max(1, len(data)),
        "reward_mean": sum(reward) / max(1, len(reward)),
        "rougeL_mean": sum(rougeL) / max(1, len(rougeL)),
        "lint_delta_mean": sum(lint) / max(1, len(lint)),
    }
    os.makedirs("eval", exist_ok=True)
    with open("eval/report.json", "w") as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
