import streamlit as st
import json
import pandas as pd
import os

st.set_page_config(page_title="LLM Eval Leaderboard", page_icon="🏆", layout="wide")

st.title("🏆 LLM Structured Extraction Leaderboard")
st.caption("Benchmarking Groq (Llama 3.3), Cohere (Command-R), and Mistral on job posting extraction")

SCORES_PATH = "results/scores.json"

if not os.path.exists(SCORES_PATH):
    st.warning("No results yet. Run `python -m evals.pipeline` then `python -m evals.scorer` first.")
    st.stop()

with open(SCORES_PATH) as f:
    data = json.load(f)

summary = data["summary"]
sample_scores = data["sample_scores"]

# ── Leaderboard Table ──────────────────────────────────────────────────────────
st.subheader("📊 Model Summary")

rows = []
for model, s in summary.items():
    rows.append({
        "Model": model.upper(),
        "F1 Score ↑": s["f1_avg"],
        "Hallucination Rate ↓": s["hallucination_rate"],
        "Avg Latency (s) ↓": s["avg_latency"],
        "Success Rate": f"{s['success_rate']:.0%}",
        "Samples Scored": s["samples_scored"],
        "Errors": s["errors"],
    })

df = pd.DataFrame(rows).sort_values("F1 Score ↑", ascending=False).reset_index(drop=True)
df.index += 1  # Rank starts at 1

st.dataframe(
    df.style.highlight_max(subset=["F1 Score ↑"], color="#1a472a")
           .highlight_min(subset=["Hallucination Rate ↓"], color="#1a472a")
           .highlight_min(subset=["Avg Latency (s) ↓"], color="#1a472a"),
    use_container_width=True,
)

# ── Metric Cards ───────────────────────────────────────────────────────────────
st.subheader("🥇 Best in Class")
col1, col2, col3 = st.columns(3)

best_f1_candidates = sorted(summary.items(), key=lambda x: (-x[1]["f1_avg"], x[1]["avg_latency"]))
best_f1 = best_f1_candidates[0]
best_hal = min(summary.items(), key=lambda x: (x[1]["hallucination_rate"], x[1]["avg_latency"]))
best_lat = min((x for x in summary.items() if x[1]["avg_latency"] > 0), key=lambda x: x[1]["avg_latency"])

col1.metric("Highest F1", best_f1[0].upper(), f"{best_f1[1]['f1_avg']:.3f}")
col2.metric("Lowest Hallucination", best_hal[0].upper(), f"{best_hal[1]['hallucination_rate']:.3f}")
col3.metric("Fastest", best_lat[0].upper(), f"{best_lat[1]['avg_latency']:.2f}s")

# ── Per-Field F1 Breakdown ─────────────────────────────────────────────────────
st.subheader("🔬 Per-Field F1 Breakdown")

field_data = {model: {} for model in summary}
for sample in sample_scores:
    for model, score in sample["scores"].items():
        if "field_scores" in score:
            for field, fs in score["field_scores"].items():
                if not fs.get("skipped") and fs["f1"] is not None:
                    field_data[model].setdefault(field, []).append(fs["f1"])

field_rows = []
all_fields = ["title", "company", "location", "employment_type", "required_skills"]
for field in all_fields:
    row = {"Field": field}
    for model in summary:
        vals = field_data.get(model, {}).get(field, [])
        row[model.upper()] = round(sum(vals) / len(vals), 3) if vals else "—"
    field_rows.append(row)

st.dataframe(pd.DataFrame(field_rows), use_container_width=True)

# ── Sample Inspector ───────────────────────────────────────────────────────────
st.subheader("🔍 Sample Inspector")
sample_ids = [s["sample_id"] for s in sample_scores]
selected = st.selectbox("Select a sample", sample_ids)

sample = next(s for s in sample_scores if s["sample_id"] == selected)
cols = st.columns(len(sample["scores"]))

for col, (model, score) in zip(cols, sample["scores"].items()):
    with col:
        st.markdown(f"**{model.upper()}**")
        if score.get("error"):
            st.error(score["error"][:200])
        else:
            output = score.get("output") or {}
            st.json(output)

st.markdown("---")
st.caption("Built with Groq · Cohere · Mistral · RAGAS · LangFuse · Streamlit")