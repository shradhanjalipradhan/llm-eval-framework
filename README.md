# LLM Eval Framework — Structured Extraction Benchmark

A production-grade evaluation pipeline that benchmarks LLMs on document-to-structured-output tasks, with hallucination detection, F1 scoring, and a live leaderboard dashboard.

> Built to mirror what teams at Tennr, Novaflow, and DocuSign run internally — without any paid tooling.

---

## Benchmark Results (20 samples, LinkedIn Job Postings dataset)

| Model | F1 Score ↑ | Hallucination Rate ↓ | Avg Latency ↓ | Success Rate |
|---|---|---|---|---|
| **Groq** (Llama 3.3 70B) | **0.200** | **0.000** | **0.45s** | 100% |
| Mistral (Small) | 0.200 | 0.008 | 0.83s | 100% |
| Cohere (Command-R) | 0.200 | 0.000 | 18.88s | 100% |

**Key findings:**
- Groq is the clear winner on latency — 40x faster than Cohere at identical F1
- Mistral had a small but non-zero hallucination rate (0.8%)
- All models achieved 100% success rate (valid structured JSON every time)
- F1 scores are bounded by sparse ground truth — only `title` had scoreable labels across all samples

---

## Architecture

```
HuggingFace Dataset
       ↓
Pydantic v2 Schema (job_schema.py)
       ↓
Extraction Prompt (prompts/extraction_prompt.py)
       ↓
┌──────────────────────────────────┐
│  Groq   │  Cohere  │  Mistral   │  ← 3 models in parallel
└──────────────────────────────────┘
       ↓
RAGAS-style Scorer (F1 + Hallucination)
       ↓
┌─────────────────────┐
│  Supabase DB        │  ← stores every run
│  Supabase Storage   │  ← stores raw JSON results
└─────────────────────┘
       ↓
Streamlit Leaderboard Dashboard
```

---

## Stack

| Tool | Role | Enterprise Equivalent |
|---|---|---|
| Groq (Llama 3.3 70B) | Fast inference | OpenAI GPT-4o |
| Cohere Command-R | Extraction model | Anthropic Claude |
| Mistral Small | Extraction model | Mistral Large |
| Pydantic v2 | Schema validation | Pydantic v2 |
| RAGAS-style scoring | F1 + hallucination | RAGAS / custom eval |
| LangFuse | Observability / tracing | LangSmith |
| Supabase | Storage + database | AWS S3 + RDS |
| Streamlit | Dashboard | Internal tooling |
| pytest | Unit tests | pytest |

---

## Project Structure

```
llm-eval-framework/
├── data/
│   ├── loader.py          # HuggingFace dataset loader
│   └── samples.json       # cached 100 job postings
├── schemas/
│   └── job_schema.py      # Pydantic v2 extraction schema
├── prompts/
│   └── extraction_prompt.py  # system + user prompts
├── evals/
│   ├── pipeline.py        # calls all 3 models
│   └── scorer.py          # F1, hallucination, leaderboard
├── storage/
│   └── supabase_store.py  # uploads to Supabase
├── dashboard/
│   └── app.py             # Streamlit leaderboard
├── tests/
│   └── test_scorer.py     # 17 unit tests
└── results/
    ├── raw_outputs.json   # model outputs
    └── scores.json        # scored results
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/llm-eval-framework
cd llm-eval-framework
python -m venv venv && source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys
```

**Run the full pipeline:**
```bash
python -m data.loader          # download dataset
python -m evals.pipeline       # run all 3 models
python -m evals.scorer         # score + print leaderboard
python -m storage.supabase_store  # upload to Supabase
streamlit run dashboard/app.py # launch dashboard
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## API Keys Required (all free, no credit card)

| Service | Get Key | Used For |
|---|---|---|
| Groq | console.groq.com | Llama 3.3 inference |
| Cohere | dashboard.cohere.com | Command-R inference |
| Mistral | console.mistral.ai | Mistral inference |
| LangFuse | cloud.langfuse.com | Tracing |
| Supabase | supabase.com | Storage + DB |

---

## What This Evaluates

Each model receives the raw job posting text and must extract:

```python
class JobExtraction(BaseModel):
    title: Optional[str]
    company: Optional[str]
    location: Optional[str]
    salary_min: Optional[int]
    salary_max: Optional[int]
    required_skills: List[str]
    experience_years: Optional[int]
    employment_type: Optional[str]
    remote: Optional[bool]
```

Scoring uses:
- **Field-level F1** — precision/recall per field against ground truth
- **Hallucination detection** — checks if extracted values appear in source text
- **Latency** — wall-clock time per extraction
- **Success rate** — % of valid Pydantic-parseable responses

---

## Extending This

- **Add a new model**: implement `call_newmodel()` in `evals/pipeline.py`
- **Add a new dataset**: implement a loader in `data/` returning the same format
- **Add a new schema**: create a new Pydantic model in `schemas/`
- **Add RAGAS metrics**: swap the scorer for `ragas.evaluate()` with faithfulness + answer relevancy