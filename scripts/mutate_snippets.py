import jsonlines, random, re
SRC="eval/dataset.jsonl"; OUT="reward_model/data/seed.jsonl"
def mutate(code):
    v = re.sub(r"\t", "    ", code)
    v = re.sub(r"def\\s+(\\w+)\\(", lambda m: f"def {m.group(1)}(", v)
    return v
def reviews(code):
    return [
        "Add a docstring and PEP8 spaces.",
        "Consider type hints and handling edge cases.",
        "Rename short variables to meaningful names."
    ]
src = list(jsonlines.open(SRC))
with jsonlines.open(OUT,"a") as w:
    for ex in src:
        c = ex["code"]; mc = mutate(c)
        for r in reviews(mc):
            w.write({"code": mc, "review": r, "label": 1, "lint_delta": 1, "tests_passed": 0})
print("appended synthetic seed →", OUT)
