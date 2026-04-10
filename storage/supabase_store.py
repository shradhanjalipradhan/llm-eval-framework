import os
import json
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    return create_client(url, key)


def upload_results(scores_path: str = "results/scores.json"):
    """Upload scores.json to Supabase Storage as a versioned file."""
    client = get_client()

    with open(scores_path, "rb") as f:
        content = f.read()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_results_{timestamp}.json"

    # Upload to 'eval-results' bucket
    response = client.storage.from_("eval-results").upload(
        path=filename,
        file=content,
        file_options={"content-type": "application/json"},
    )

    print(f"Uploaded to Supabase: {filename}")
    return filename


def upload_summary_to_table(scores_path: str = "results/scores.json"):
    """Insert model summary rows into Supabase eval_runs table."""
    client = get_client()

    with open(scores_path) as f:
        data = json.load(f)

    summary = data["summary"]
    run_id = datetime.utcnow().isoformat()
    rows = []

    for model, stats in summary.items():
        rows.append({
            "run_id": run_id,
            "model": model,
            "f1_avg": stats["f1_avg"],
            "hallucination_rate": stats["hallucination_rate"],
            "avg_latency": stats["avg_latency"],
            "success_rate": stats["success_rate"],
            "samples_scored": stats["samples_scored"],
            "errors": stats["errors"],
        })

    response = client.table("eval_runs").insert(rows).execute()
    print(f"Inserted {len(rows)} rows into eval_runs table")
    return response


if __name__ == "__main__":
    print("Uploading results to Supabase...")
    upload_results()
    upload_summary_to_table()
    print("Done!")