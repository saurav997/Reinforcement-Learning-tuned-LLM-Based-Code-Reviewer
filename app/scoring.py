import subprocess, tempfile, os
from rouge_score import rouge_scorer

def _run_pylint(py_text: str) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as f:
        f.write(py_text)
        path = f.name
    try:
        # count total issues via pylint (simple grep on output length)
        p = subprocess.run(["pylint", path, "-sn", "--disable=all", "--enable=E,W,C,R"],
                           capture_output=True, text=True, timeout=15)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        return len(lines)
    except Exception:
        return 0
    finally:
        try: os.remove(path)
        except: pass

def _apply_review_to_code(code: str, review: str) -> str:
    # Dumb placeholder: we DO NOT auto-edit code.
    # We just assume short, concrete reviews correlate with better lint behavior.
    # Later: implement simple refactors for certain patterns.
    return code

def _heuristics(review: str, code: str) -> float:
    # simple length + keyword presence
    L = len(review.split())
    brevity = 1.0 if 10 <= L <= 120 else 0.0
    concrete = any(k in review.lower() for k in ["rename", "docstring", "type hint", "exception", "complexity"])
    return 0.5*brevity + 0.5*(1.0 if concrete else 0.0)

def _rouge_key(review: str, code: str) -> float:
    # crude: reference built from cue words we want in reviews
    ref = "naming clarity; add docstring; type hints; handle exceptions; reduce complexity"
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return rs.score(ref, review)["rougeL"].fmeasure

def _reward_model_score(review: str, code: str) -> float:
    # stub returns neutral score; replace with real RM later
    return 0.5

def score_candidates(code: str, candidates: list[str]):
    base_issues = _run_pylint(code)
    scored = []
    for text in candidates:
        new_code = _apply_review_to_code(code, text)
        new_issues = _run_pylint(new_code)
        lint_delta = base_issues - new_issues  # positive is good

        s_h = _heuristics(text, code)
        s_r = _rouge_key(text, code)
        s_rm = _reward_model_score(text, code)

        # Weighted total — tune later
        total = 0.5*lint_delta + 0.3*s_rm + 0.1*s_h + 0.1*s_r
        scored.append({"text": text, "scores": {
            "lint_delta": float(lint_delta),
            "rm": float(s_rm),
            "heur": float(s_h),
            "rougeL": float(s_r),
        }, "total": float(total)})
    return scored
