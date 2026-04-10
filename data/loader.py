from datasets import load_dataset
import json
import os

def load_job_samples(n: int = 100, save_path: str = "data/samples.json") -> list[dict]:
    """
    Load n job postings from HuggingFace and save locally.
    Only downloads once — reuses saved file after that.
    """
    if os.path.exists(save_path):
        print(f"Loading cached samples from {save_path}")
        with open(save_path, "r") as f:
            return json.load(f)

    print(f"Downloading dataset from HuggingFace (first time only)...")
    ds = load_dataset("batuhanmtl/job-skill-set", split="train")
    sample = ds.select(range(n))

    records = []
    for row in sample:
        records.append({
            "job_description": row.get("job_description", ""),
            "ground_truth": {
                "title": row.get("job_title", ""),
                "company": row.get("company", ""),
                "location": row.get("location", None),
                "employment_type": row.get("employment_type", None),
                "required_skills": row.get("skills", []),
                "salary_min": None,
                "salary_max": None,
                "experience_years": None,
                "remote": None,
            }
        })

    os.makedirs("data", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} samples to {save_path}")
    return records


if __name__ == "__main__":
    samples = load_job_samples(n=100)
    print(f"\nLoaded {len(samples)} samples")
    print("\nExample record:")
    print(json.dumps(samples[0], indent=2))