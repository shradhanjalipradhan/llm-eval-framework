SYSTEM_PROMPT = """You are a precise information extraction engine.
Your job is to extract structured data from job postings.
You must respond with ONLY valid JSON. No explanation, no markdown, no code blocks.
If a field cannot be found, use null for optional fields or [] for lists.
Never hallucinate or guess values that are not in the text."""

def build_extraction_prompt(job_description: str) -> str:
    return f"""Extract the following fields from this job posting and return as JSON:

{{
  "title": "string - job title",
  "company": "string - company name",
  "location": "string or null - city, state or Remote",
  "salary_min": "integer or null - minimum annual salary in USD",
  "salary_max": "integer or null - maximum annual salary in USD",
  "required_skills": ["list of strings - technical skills only"],
  "experience_years": "integer or null - minimum years of experience",
  "employment_type": "string or null - Full-time / Part-time / Contract / Internship",
  "remote": "boolean or null - true if remote work is mentioned"
}}

Rules:
- Extract ONLY what is explicitly stated. Do not infer or guess.
- For salary: convert hourly rates to annual (multiply by 2080). If range not given, use null.
- For skills: include only technical/hard skills (e.g. Python, AWS, SQL). Exclude soft skills.
- For experience_years: extract the minimum number mentioned (e.g. "5-7 years" → 5).
- Return ONLY the JSON object. No other text.

Job Posting:
---
{job_description[:3000]}
---

JSON:"""