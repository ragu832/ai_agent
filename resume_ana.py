import streamlit as st
import pdfplumber
from openai import OpenAI
import os
import json

st.set_page_config(
    page_title="AI Resume Parser & ATS Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.success-metric {
    background-color: #d4edda;
    color: #155724;
}
.warning-metric {
    background-color: #fff3cd;
    color: #856404;
}
.danger-metric {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ GROQ_API_KEY not found in Streamlit secrets. Please check your .streamlit/secrets.toml file.")
    st.stop()

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def get_resume_details(resume_text):
    prompt = f"""
You are a smart AI resume parser. Extract the following from the resume:

- Full Name
- Email
- Phone Number
- Education
- Work Experience
- Skills
- Certifications

Return the result in JSON format.

Resume:
\"\"\"{resume_text}\"\"\"
"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def calculate_ats_score(resume_text, job_description):
    prompt = f"""
You are an expert ATS (Applicant Tracking System) analyzer. Compare the resume with the job description and provide:

1. ATS Score (0-100): Based on keyword matching, skills alignment, and experience relevance
2. Matched Keywords: List keywords from job description found in resume
3. Missing Keywords: Critical keywords from job description missing in resume
4. Skills Gap Analysis: What skills are required but missing
5. Experience Alignment: How well the experience matches job requirements
6. Overall Assessment: Brief summary of candidacy strength

Provide scoring criteria:
- 90-100: Excellent match, highly likely to pass ATS
- 80-89: Good match, likely to pass ATS with minor gaps
- 70-79: Fair match, moderate chance with some improvements needed
- 60-69: Below average, significant improvements required
- Below 60: Poor match, major changes needed

Return the result in JSON format with the following structure:
{{
    "ats_score": <number>,
    "score_category": "<Excellent/Good/Fair/Below Average/Poor>",
    "matched_keywords": [<list of keywords>],
    "missing_keywords": [<list of keywords>],
    "skills_gap": [<list of missing skills>],
    "experience_alignment": "<assessment>",
    "overall_assessment": "<summary>",
    "recommendations": [<list of improvement suggestions>]
}}

Job Description:
\"\"\"{job_description}\"\"\"

Resume:
\"\"\"{resume_text}\"\"\"
"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def get_resume_improvement_suggestions(resume_text, ats_analysis):
    prompt = f"""
Based on the ATS analysis provided, give specific, actionable recommendations to improve the resume:

1. Keyword Optimization: Specific keywords to add and where
2. Skills Enhancement: What skills to highlight or add
3. Experience Formatting: How to better present work experience
4. Content Structure: Improvements to resume layout and sections
5. Industry-Specific Tips: Tailored advice for the target role

Make recommendations practical and specific. Focus on:
- Adding missing keywords naturally
- Improving existing content rather than fabricating experience
- ATS-friendly formatting suggestions
- Quantifying achievements where possible

ATS Analysis:
\"\"\"{ats_analysis}\"\"\"

Resume:
\"\"\"{resume_text}\"\"\"

Return actionable recommendations in a clear, numbered format.
"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

def get_skill_upgrade_suggestions(job_description):
    prompt = f"""
You are a career coach and skill development expert. Based on the following job description, suggest a list of technical and soft skills that a candidate should consider upgrading or learning to be a strong fit for this role. For each skill, briefly explain why it is important for the job.

Job Description:
\"\"\"{job_description}\"\"\"

Return your suggestions as a numbered list, each with a skill and a short explanation.
"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def get_job_role_roadmap(job_description):
    prompt = f"""
You are a career advisor. Based on the following job description, provide a step-by-step career pathway (roadmap) for someone aspiring to excel in this role. Include recommended education, certifications, skill development, experience milestones, and networking or portfolio tips. Present the roadmap as a clear, numbered or bulleted list.

Job Description:
\"\"\"{job_description}\"\"\"
"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def get_score_color_and_icon(score):
    if score >= 90:
        return "🟢", "#28a745"
    elif score >= 80:
        return "🟡", "#ffc107"
    elif score >= 70:
        return "🟠", "#fd7e14"
    elif score >= 60:
        return "🔴", "#dc3545"
    else:
        return "🔴", "#dc3545"

def sidebar_info():
    st.header("How to Use")
    st.markdown("""
    Step 1: Upload your resume in PDF format
    
    Step 2: Paste the job description you're targeting
    
    Step 3: Analyze your resume:
    - Parse Resume: Extract structured data
    - ATS Score: Get compatibility score (0-100)
    - Improvement Tips: Get actionable advice
    """)
    st.header("ATS Score Guide")
    st.markdown("""
    - 90-100: Excellent match
    - 80-89: Good match  
    - 70-79: Fair match
    - 60-69: Below average
    - <60: Poor match
    """)
    st.header("Quick Tips")
    st.markdown("""
    Use keywords from job description  
    Quantify your achievements  
    Include relevant technical skills  
    Use standard section headings  
    Keep formatting simple and clean
    """)

def page_resume_parser():
    st.title("Resume Parser")
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    if uploaded_file:
        with st.spinner("Extracting text from resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
        st.subheader("Extracted Resume Text (Preview)")
        with st.expander("View Resume Text", expanded=False):
            st.text_area("Resume Text", resume_text[:2000] + "...", height=300)
        if st.button("Parse Resume"):
            with st.spinner("Parsing resume with Groq..."):
                parsed_output = get_resume_details(resume_text)
            st.subheader("Structured Resume Information")
            st.code(parsed_output, language='json')

def page_ats_score():
    st.title("ATS Score Analyzer")
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    job_description = st.text_area(
        "Paste the Job Description here:", 
        height=200,
        placeholder="Paste the job description you want to match your resume against..."
    )
    if uploaded_file and job_description.strip():
        with st.spinner("Extracting text from resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
        if st.button("Calculate ATS Score"):
            with st.spinner("Analyzing ATS compatibility..."):
                ats_analysis = calculate_ats_score(resume_text, job_description)
            try:
                ats_data = json.loads(ats_analysis)
                score = ats_data.get('ats_score', 0)
                icon, color = get_score_color_and_icon(score)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ATS Score", f"{score}/100")
                with col2:
                    score_category = ats_data.get('score_category', 'Unknown')
                    st.metric("Category", f"{score_category}")
                with col3:
                    match_level = "Strong" if score >= 80 else "Moderate" if score >= 70 else "Weak"
                    st.metric("Match Level", match_level)
                if score >= 90:
                    st.success("Excellent! Your resume is highly likely to pass ATS screening.")
                elif score >= 80:
                    st.success("Good match! Minor improvements could make it even better.")
                elif score >= 70:
                    st.warning("Fair match. Some improvements needed to increase chances.")
                elif score >= 60:
                    st.warning("Below average. Significant improvements recommended.")
                else:
                    st.error("Poor match. Major changes needed to improve ATS compatibility.")
                st.subheader("Detailed ATS Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Matched Keywords:")
                    matched_keywords = ats_data.get('matched_keywords', [])
                    if matched_keywords:
                        for keyword in matched_keywords:
                            st.write(f"• {keyword}")
                    else:
                        st.write("No keywords matched")
                with col2:
                    st.write("Missing Keywords:")
                    missing_keywords = ats_data.get('missing_keywords', [])
                    if missing_keywords:
                        for keyword in missing_keywords:
                            st.write(f"• {keyword}")
                    else:
                        st.write("No missing keywords identified")
                if ats_data.get('skills_gap'):
                    st.write("Skills Gap Analysis:")
                    for skill in ats_data.get('skills_gap', []):
                        st.write(f"• {skill}")
                if ats_data.get('experience_alignment'):
                    st.write("Experience Alignment:")
                    st.write(ats_data.get('experience_alignment', ''))
                if ats_data.get('overall_assessment'):
                    st.write("Overall Assessment:")
                    st.write(ats_data.get('overall_assessment', ''))
            except json.JSONDecodeError:
                st.error("Error parsing ATS analysis. Raw output:")
                st.code(ats_analysis)
    else:
        st.info("Please upload a resume and enter a job description to calculate ATS score")

def page_improvement_tips():
    st.title("Resume Improvement Tips")
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    job_description = st.text_area(
        "Paste the Job Description here:", 
        height=200,
        placeholder="Paste the job description you want to match your resume against..."
    )
    if uploaded_file and job_description.strip():
        with st.spinner("Extracting text from resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
        if st.button("Get Improvement Suggestions"):
            with st.spinner("Generating improvement suggestions..."):
                ats_analysis = calculate_ats_score(resume_text, job_description)
                improvement_suggestions = get_resume_improvement_suggestions(resume_text, ats_analysis)
            st.subheader("Resume Improvement Recommendations")
            st.markdown(improvement_suggestions)
    else:
        st.info("Please upload a resume and enter a job description to get improvement suggestions")

def page_skill_upgrade():
    st.title("Skill Upgrade Suggestions")
    job_description = st.text_area(
        "Paste the Job Description here:", 
        height=200,
        placeholder="Paste the job description for skill upgrade suggestions..."
    )
    if job_description.strip():
        if st.button("Get Skill Upgrade Suggestions"):
            with st.spinner("Analyzing job description for skill upgrade suggestions..."):
                skill_suggestions = get_skill_upgrade_suggestions(job_description)
            st.markdown(skill_suggestions)
    else:
        st.info("Please enter a job description to get skill upgrade suggestions")

def page_job_roadmap():
    st.title("Job Role Roadmap")
    job_description = st.text_area(
        "Paste the Job Description here:", 
        height=200,
        placeholder="Paste the job description for roadmap suggestions..."
    )
    if job_description.strip():
        if st.button("Get Job Role Roadmap"):
            with st.spinner("Generating career pathway for this job role..."):
                roadmap = get_job_role_roadmap(job_description)
            st.markdown(roadmap)
    else:
        st.info("Please enter a job description to get a roadmap for this role")

with st.sidebar:
    page = st.radio(
        "Navigate",
        [
            "Resume Parser",
            "ATS Score",
            "Improvement Tips",
            "Skill Upgrade Suggestions",
            "Job Role Roadmap"
        ]
    )

if page == "Resume Parser":
    page_resume_parser()
elif page == "ATS Score":
    page_ats_score()
elif page == "Improvement Tips":
    page_improvement_tips()
elif page == "Skill Upgrade Suggestions":
    page_skill_upgrade()
elif page == "Job Role Roadmap":
    page_job_roadmap()