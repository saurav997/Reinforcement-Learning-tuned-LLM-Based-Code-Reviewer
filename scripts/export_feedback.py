import jsonlines, sys
src = "logs/events.jsonl"; out = "reward_model/data/feedback.jsonl"
with jsonlines.open(src) as r, jsonlines.open(out,"w") as w:
    for e in r:
        if e.get("type")=="rating":
            w.write({"code": e["code"], "review": e["top_review"], "label": 1 if e["stars"]>=4 else 0,
                     "lint_delta": 0, "tests_passed": 0})
print("wrote", out)
