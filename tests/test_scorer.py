import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.scorer import score_field, score_list_field, detect_hallucination, score_sample


# ── score_field ────────────────────────────────────────────────────────────────

def test_score_field_exact_match():
    result = score_field("Software Engineer", "Software Engineer")
    assert result["f1"] == 1.0

def test_score_field_case_insensitive():
    result = score_field("software engineer", "Software Engineer")
    assert result["f1"] == 1.0

def test_score_field_mismatch():
    result = score_field("Data Scientist", "Software Engineer")
    assert result["f1"] == 0.0

def test_score_field_empty_prediction():
    result = score_field(None, "Software Engineer")
    assert result["f1"] == 0.0
    assert result["recall"] == 0.0

def test_score_field_no_ground_truth():
    result = score_field("Software Engineer", None)
    assert result["skipped"] is True
    assert result["f1"] is None

def test_score_field_both_empty():
    result = score_field(None, None)
    assert result["skipped"] is True


# ── score_list_field ───────────────────────────────────────────────────────────

def test_score_list_perfect():
    result = score_list_field(["Python", "AWS", "Docker"], ["Python", "AWS", "Docker"])
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0

def test_score_list_partial_overlap():
    result = score_list_field(["Python", "AWS", "Java"], ["Python", "AWS", "Docker"])
    assert result["precision"] == pytest.approx(2/3, abs=0.01)
    assert result["recall"] == pytest.approx(2/3, abs=0.01)

def test_score_list_no_overlap():
    result = score_list_field(["Java", "C++"], ["Python", "AWS"])
    assert result["f1"] == 0.0

def test_score_list_empty_prediction():
    result = score_list_field([], ["Python", "AWS"])
    assert result["f1"] == 0.0
    assert result["precision"] == 0.0

def test_score_list_no_ground_truth():
    result = score_list_field(["Python"], [])
    assert result["skipped"] is True

def test_score_list_case_insensitive():
    result = score_list_field(["python", "aws"], ["Python", "AWS"])
    assert result["f1"] == 1.0


# ── detect_hallucination ───────────────────────────────────────────────────────

def test_no_hallucination():
    predicted = {"title": "engineer", "company": "acme", "location": "austin", "required_skills": ["python"]}
    jd = "We are acme looking for an engineer in austin with python skills"
    rate = detect_hallucination(predicted, jd)
    assert rate == 0.0

def test_full_hallucination():
    predicted = {"title": "astronaut", "company": "nasa", "location": "mars", "required_skills": ["moonwalking"]}
    jd = "We need a baker in Paris who knows bread making"
    rate = detect_hallucination(predicted, jd)
    assert rate > 0.5

def test_hallucination_empty_prediction():
    rate = detect_hallucination(None, "some job description")
    assert rate == 1.0


# ── score_sample ───────────────────────────────────────────────────────────────

def test_score_sample_full():
    sample = {
        "sample_id": 0,
        "job_description": "Acme Corp needs a Python engineer in New York",
        "ground_truth": {
            "title": "Python Engineer",
            "company": "Acme Corp",
            "location": "New York",
            "employment_type": None,
            "required_skills": ["Python"],
        },
        "models": {
            "groq": {
                "output": {
                    "title": "Python Engineer",
                    "company": "Acme Corp",
                    "location": "New York",
                    "required_skills": ["Python"],
                    "employment_type": None,
                    "salary_min": None,
                    "salary_max": None,
                    "experience_years": None,
                    "remote": None,
                },
                "latency": 0.5,
                "tokens": 100,
                "error": None,
            }
        }
    }
    result = score_sample(sample)
    assert result["scores"]["groq"]["f1_avg"] == 1.0
    assert result["scores"]["groq"]["hallucination_rate"] == 0.0

def test_score_sample_with_error():
    sample = {
        "sample_id": 1,
        "job_description": "some job",
        "ground_truth": {"title": "Engineer"},
        "models": {
            "groq": {"output": None, "latency": None, "tokens": None, "error": "API failed"}
        }
    }
    result = score_sample(sample)
    assert result["scores"]["groq"]["error"] == "API failed"