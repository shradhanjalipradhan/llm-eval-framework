import os
import json
import time
from dotenv import load_dotenv
from groq import Groq
import cohere
from mistralai.client import MistralClient
from langfuse import Langfuse
from prompts.extraction_prompt import SYSTEM_PROMPT, build_extraction_prompt
from schemas.job_schema import JobExtraction

load_dotenv()

# Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

def make_trace(name: str):
    return langfuse.trace(name=name)


def parse_json_response(text: str) -> dict:
    """Safely parse JSON from model response."""
    text = text.strip()
    # Strip markdown code blocks if model ignores instructions
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def call_groq(prompt: str, trace) -> tuple[dict, float, int]:
    start = time.time()
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=1000,
    )
    text = resp.choices[0].message.content
    return parse_json_response(text), time.time() - start, resp.usage.total_tokens


def call_cohere(prompt: str, trace) -> tuple[dict, float, int]:
    time.sleep(1)  # Free tier: 20 RPM
    start = time.time()
    resp = cohere_client.chat(
        model="command-r-08-2024",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.message.content[0].text
    tokens = resp.usage.tokens.input_tokens + resp.usage.tokens.output_tokens
    return parse_json_response(text), time.time() - start, tokens


def call_mistral(prompt: str, trace) -> tuple[dict, float, int]:
    start = time.time()
    resp = mistral_client.chat(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=1000,
    )
    text = resp.choices[0].message.content
    return parse_json_response(text), time.time() - start, resp.usage.total_tokens


def run_extraction(sample: dict, sample_id: int) -> dict:
    """Run all 3 models on one job posting and return raw outputs."""
    prompt = build_extraction_prompt(sample["job_description"])

    # Langfuse tracing via generation log (compatible with all versions)
    def log_to_langfuse(model_name, prompt, output, latency):
        try:
            langfuse.generation(
                name=f"{model_name}-extraction-{sample_id}",
                model=model_name,
                input=prompt[:500],
                output=str(output)[:500],
                metadata={"latency": latency, "sample_id": sample_id},
            )
        except Exception:
            pass  # Don't let logging failures break the pipeline

    result = {"sample_id": sample_id, "ground_truth": sample["ground_truth"], "models": {}}

    for model_name, fn in [("groq", call_groq), ("cohere", call_cohere), ("mistral", call_mistral)]:
        try:
            output, latency, tokens = fn(prompt, None)
            validated = JobExtraction(**output)
            log_to_langfuse(model_name, prompt, output, latency)
            result["models"][model_name] = {
                "output": validated.model_dump(),
                "latency": round(latency, 3),
                "tokens": tokens,
                "error": None,
            }
            print(f"  ✓ {model_name} ({latency:.2f}s)")
        except Exception as e:
            result["models"][model_name] = {
                "output": None,
                "latency": None,
                "tokens": None,
                "error": str(e),
            }
            print(f"  ✗ {model_name} failed: {e}")

    langfuse.flush()
    return result


if __name__ == "__main__":
    from data.loader import load_job_samples

    samples = load_job_samples(n=100)
    print(f"Running on first 20 samples...\n")

    results = []
    for i, sample in enumerate(samples[:20]):
        print(f"Sample {i+1}:")
        r = run_extraction(sample, i)
        results.append(r)

    # Save raw results
    os.makedirs("results", exist_ok=True)
    with open("results/raw_outputs.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to results/raw_outputs.json")