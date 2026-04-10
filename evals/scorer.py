import json
import os
from typing import Optional


def normalize(value) -> str:
    """Normalize a value to lowercase string for comparison."""
    if value is None:
        return ""
    return str(value).lower().strip()


def score_field(predicted, ground_truth) -> dict:
    """Score a single field — returns precision, recall, F1."""
    pred = normalize(predicted)
    gt = normalize(ground_truth)

    if not gt:
        # No ground truth — skip this field
        return {"precision": None, "recall": None, "f1": None, "skipped": True}

    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "skipped": False}

    match = 1.0 if pred == gt else 0.0
    return {"precision": match, "recall": match, "f1": match, "skipped": False}


def score_list_field(predicted: list, ground_truth: list) -> dict:
    """Score a list field (e.g. required_skills) using set-based F1."""
    if not ground_truth:
        return {"precision": None, "recall": None, "f1": None, "skipped": True}

    pred_set = {normalize(x) for x in (predicted or [])}
    gt_set = {normalize(x) for x in ground_truth}

    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "skipped": False}

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "skipped": False}


def detect_hallucination(predicted: dict, job_description: str) -> float:
    """
    Simple hallucination detector:
    Check if extracted values actually appear in the source text.
    Returns hallucination rate (0.0 = no hallucination, 1.0 = all hallucinated).
    """
    if not predicted:
        return 1.0

    jd_lower = job_description.lower()
    checks = []

    for field in ["title", "company", "location"]:
        val = predicted.get(field)
        if val:
            # Check if key words from the value appear in the job description
            words = [w for w in normalize(val).split() if len(w) > 3]
            if words:
                found = any(w in jd_lower for w in words)
                checks.append(found)

    for skill in (predicted.get("required_skills") or [])[:5]:  # sample first 5
        words = [w for w in normalize(skill).split() if len(w) > 2]
        if words:
            found = any(w in jd_lower for w in words)
            checks.append(found)

    if not checks:
        return 0.0

    hallucinated = checks.count(False)
    return round(hallucinated / len(checks), 3)


def score_sample(sample: dict) -> dict:
    """Score all models on a single sample."""
    gt = sample["ground_truth"]
    jd = ""
    scores = {}

    for model_name, model_data in sample["models"].items():
        if model_data["output"] is None:
            scores[model_name] = {"error": model_data["error"], "f1_avg": 0.0}
            continue

        pred = model_data["output"]

        field_scores = {
            "title":            score_field(pred.get("title"), gt.get("title")),
            "company":          score_field(pred.get("company"), gt.get("company")),
            "location":         score_field(pred.get("location"), gt.get("location")),
            "employment_type":  score_field(pred.get("employment_type"), gt.get("employment_type")),
            "required_skills":  score_list_field(pred.get("required_skills"), gt.get("required_skills", [])),
        }

        # Average F1 across non-skipped fields
        f1_values = [v["f1"] for v in field_scores.values() if not v.get("skipped") and v["f1"] is not None]
        f1_avg = round(sum(f1_values) / len(f1_values), 3) if f1_values else 0.0

        hallucination_rate = detect_hallucination(pred, sample.get("job_description", ""))

        scores[model_name] = {
            "field_scores": field_scores,
            "f1_avg": f1_avg,
            "hallucination_rate": hallucination_rate,
            "latency": model_data["latency"],
            "tokens": model_data["tokens"],
        }

    return {"sample_id": sample["sample_id"], "scores": scores}


def score_all(results_path: str = "results/raw_outputs.json") -> dict:
    """Score all samples and compute per-model aggregates."""
    with open(results_path) as f:
        results = json.load(f)

    # Re-attach job descriptions for hallucination detection
    from data.loader import load_job_samples
    samples_map = {i: s["job_description"] for i, s in enumerate(load_job_samples())}

    all_scores = []
    for r in results:
        r["job_description"] = samples_map.get(r["sample_id"], "")
        all_scores.append(score_sample(r))

    # Dynamically get model names from results instead of hardcoding
    model_names = list({model for r in results for model in r["models"].keys()})
    summary = {}

    for model in model_names:
        f1s, hallucinations, latencies = [], [], []
        errors = 0

        for s in all_scores:
            m = s["scores"].get(model, {})
            if "error" in m and m["error"]:
                errors += 1
            else:
                f1s.append(m.get("f1_avg", 0))
                hallucinations.append(m.get("hallucination_rate", 0))
                if m.get("latency"):
                    latencies.append(m["latency"])

        summary[model] = {
            "f1_avg": round(sum(f1s) / len(f1s), 3) if f1s else 0.0,
            "hallucination_rate": round(sum(hallucinations) / len(hallucinations), 3) if hallucinations else 0.0,
            "avg_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "success_rate": round(len(f1s) / (len(f1s) + errors), 3) if (f1s or errors) else 0.0,
            "samples_scored": len(f1s),
            "errors": errors,
        }

    output = {"sample_scores": all_scores, "summary": summary}

    os.makedirs("results", exist_ok=True)
    with open("results/scores.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== MODEL LEADERBOARD ===")
    for model, s in sorted(summary.items(), key=lambda x: -x[1]["f1_avg"]):
        print(f"{model:10} | F1: {s['f1_avg']:.3f} | Hallucination: {s['hallucination_rate']:.3f} | Latency: {s['avg_latency']:.2f}s | Success: {s['success_rate']:.0%}")

    return output


if __name__ == "__main__":
    score_all()