from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class JobExtraction(BaseModel):
    title: Optional[str] = Field(default=None, description="Job title or position name")
    company: Optional[str] = Field(default=None, description="Company or organization name")
    location: Optional[str] = Field(default=None, description="Job location or 'Remote'")
    salary_min: Optional[int] = Field(default=None, description="Minimum salary in USD per year")
    salary_max: Optional[int] = Field(default=None, description="Maximum salary in USD per year")
    required_skills: List[str] = Field(default=[], description="List of required technical skills")
    experience_years: Optional[int] = Field(default=None, description="Minimum years of experience required")
    employment_type: Optional[str] = Field(default=None, description="Full-time, Part-time, Contract, etc.")
    remote: Optional[bool] = Field(default=None, description="True if remote work is allowed")

    @field_validator("required_skills", mode="before")
    @classmethod
    def normalize_skills(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v or []

    @field_validator("title", "company", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("employment_type", mode="before")
    @classmethod
    def normalize_employment(cls, v):
        if not v:
            return None
        mapping = {
            "full time": "Full-time",
            "fulltime": "Full-time",
            "part time": "Part-time",
            "parttime": "Part-time",
            "contract": "Contract",
            "freelance": "Contract",
            "internship": "Internship",
        }
        return mapping.get(v.lower().strip(), v.strip())

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Senior Software Engineer",
                "company": "Acme Corp",
                "location": "San Francisco, CA",
                "salary_min": 120000,
                "salary_max": 160000,
                "required_skills": ["Python", "AWS", "Docker"],
                "experience_years": 5,
                "employment_type": "Full-time",
                "remote": True,
            }
        }