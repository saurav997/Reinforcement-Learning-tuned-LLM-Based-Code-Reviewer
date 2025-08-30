# app.py
import gradio as gr
from reviewer import generate_one_stream, generate_candidates, select_best, log_event
from scoring import score_candidates

def review_stream(code: str):
    if not code.strip():
        yield "Paste some Python code.", [], ""
        return

    # 1) Stream a first draft (user sees tokens as they appear)
    live_text = ""
    for piece in generate_one_stream(code):
        live_text += piece
        # While streaming, we can’t fill alternatives yet.
        yield live_text, [], "Streaming first draft..."

    # 2) After streaming ends, do full multi-candidate generation + ranking
    cands = generate_candidates(code, k=3)
    scored = score_candidates(code, cands)
    best = select_best(scored)
    alts = [f"Score {s['total']:.2f} | {s['text']}" for s in scored]

    log_event({"code": code, "candidates": scored, "chosen": best})
    # 3) Replace streamed draft with the true best + show alternatives and scores
    yield best["text"], [[a] for a in alts], str(best["scores"])

with gr.Blocks() as demo:
    gr.Markdown("# RL Code Review Reranker (Streaming)")
    code = gr.Code(lines=16, label="Paste Python code")
    btn = gr.Button("Review")
    best = gr.Textbox(label="Top Review", lines=8)
    alts = gr.Dataframe(headers=["Alternatives"], row_count=(3, "fixed"))
    dbg = gr.Textbox(label="Score Breakdown (debug)", interactive=False)
    btn.click(review_stream, inputs=[code], outputs=[best, alts, dbg])

# Enable Gradio’s event queue for streaming
demo.queue(max_size=32).launch()
