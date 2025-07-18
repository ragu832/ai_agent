# AI Resume Parser & ATS Analyzer

A web-based application that leverages large language models (LLMs) to analyze resumes, evaluate them against job descriptions, and provide actionable insights to improve Applicant Tracking System (ATS) compatibility.

## Features

### Resume Parser
- Extracts structured information such as:
  - Full Name
  - Email
  - Phone Number
  - Education
  - Work Experience
  - Skills
  - Certifications

### ATS Score Analyzer
- Compares resume content with a provided job description.
- Returns:
  - ATS Score (0-100)
  - Matched and missing keywords
  - Skills gap analysis
  - Experience alignment
  - Overall assessment

### Resume Improvement Suggestions
- Offers targeted, actionable advice to improve:
  - Keyword usage
  - Content structure
  - Skill presentation
  - Formatting

### Skill Upgrade Suggestions
- Recommends relevant technical and soft skills based on job requirements.

### Career Roadmap Generator
- Provides a detailed career pathway including education, certifications, skill-building, and networking tips based on the target role.

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Groq API with LLaMA 3 (OpenAI-compatible interface)
- **PDF Parsing**: pdfplumber
- **Language**: Python 3