import streamlit as st
import pdfplumber
from openai import OpenAI
import os
import json
import pyttsx3
import threading
import time
import tempfile
import base64

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

# Voice Assistant Class
class VoiceAssistant:
    def __init__(self):
        self.engine = None
        self.is_speaking = False
        self.init_engine()
    
    def init_engine(self):
        """Initialize the TTS engine"""
        try:
            self.engine = pyttsx3.init()
            # Set properties
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
            
            # Try to set a nice voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice or use the first available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    self.engine.setProperty('voice', voices[0].id)
        except Exception as e:
            st.error(f"Voice engine initialization failed: {str(e)}")
            self.engine = None
    
    def speak(self, text):
        """Speak the given text"""
        if not self.engine:
            st.error("Voice engine not available")
            return
        
        # Clean the text for better speech
        clean_text = self.clean_text_for_speech(text)
        
        def speak_thread():
            try:
                self.is_speaking = True
                self.engine.say(clean_text)
                self.engine.runAndWait()
                self.is_speaking = False
            except Exception as e:
                st.error(f"Speech error: {str(e)}")
                self.is_speaking = False
        
        # Run in separate thread to avoid blocking
        thread = threading.Thread(target=speak_thread)
        thread.daemon = True
        thread.start()
    
    def clean_text_for_speech(self, text):
        """Clean text for better speech synthesis"""
        # Remove markdown formatting
        text = text.replace('**', '').replace('*', '')
        text = text.replace('#', '').replace('```', '')
        text = text.replace('•', '').replace('✓', 'check')
        text = text.replace('✗', 'cross').replace('🟢', 'green')
        text = text.replace('🟡', 'yellow').replace('🟠', 'orange')
        text = text.replace('🔴', 'red').replace('📊', '').replace('📋', '')
        text = text.replace('👤', '').replace('🎓', '').replace('💼', '')
        text = text.replace('🛠️', '').replace('🏆', '').replace('🔍', '')
        text = text.replace('✅', 'yes').replace('❌', 'no')
        text = text.replace('🎯', '').replace('💡', '')
        text = text.replace('📈', '').replace('🎉', '')
        text = text.replace('⚠️', 'warning').replace('🔄', '')
        text = text.replace('📚', '').replace('📍', '').replace('📅', '')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.engine and self.is_speaking:
            try:
                self.engine.stop()
                self.is_speaking = False
            except Exception as e:
                st.error(f"Error stopping speech: {str(e)}")

# Initialize voice assistant
if 'voice_assistant' not in st.session_state:
    st.session_state.voice_assistant = VoiceAssistant()

def add_voice_controls(text_content, section_name="content"):
    """Add voice control buttons to any section"""
    st.markdown("---")
    st.markdown("### 🎤 Voice Assistant")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        speak_key = f"speak_{section_name}"
        if st.button("🔊 Read Aloud", key=speak_key, type="primary"):
            st.session_state.voice_assistant.speak(text_content)
            st.success("🎤 Reading aloud...")
    
    with col2:
        stop_key = f"stop_{section_name}"
        if st.button("🔇 Stop Speech", key=stop_key):
            st.session_state.voice_assistant.stop_speaking()
            st.info("🔇 Speech stopped")
    
    with col3:
        test_key = f"test_{section_name}"
        if st.button("🎵 Test Voice", key=test_key):
            st.session_state.voice_assistant.speak("Voice test successful! Your AI assistant is ready.")
            st.success("🎵 Testing voice...")
    
    st.markdown("---")

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

IMPORTANT: Return ONLY a valid JSON object with the following structure:
{{
    "full_name": "John Doe",
    "email": "john.doe@email.com",
    "phone": "+1-234-567-8900",
    "education": [
        {{
            "degree": "Bachelor of Science in Computer Science",
            "institution": "University Name",
            "year": "2020"
        }}
    ],
    "work_experience": [
        {{
            "job_title": "Software Engineer",
            "company": "Company Name",
            "duration": "2020-2023",
            "responsibilities": ["Task 1", "Task 2"]
        }}
    ],
    "skills": ["Python", "JavaScript", "React"],
    "certifications": ["AWS Certified", "Google Cloud"]
}}

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

IMPORTANT: Return ONLY a valid JSON object with the following structure (no additional text or formatting):
{{
    "ats_score": 75,
    "score_category": "Fair",
    "matched_keywords": ["keyword1", "keyword2"],
    "missing_keywords": ["keyword3", "keyword4"],
    "skills_gap": ["skill1", "skill2"],
    "experience_alignment": "Brief assessment text",
    "overall_assessment": "Summary text",
    "recommendations": ["suggestion1", "suggestion2"]
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
        return "🔴", "#0026ff"
    else:
        return "🔴", "#dc3545"

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
            
            # Try to parse JSON and display nicely
            try:
                # Clean up the response
                clean_output = parsed_output.strip()
                if "```json" in clean_output:
                    start = clean_output.find("```json") + 7
                    end = clean_output.find("```", start)
                    if end != -1:
                        clean_output = clean_output[start:end].strip()
                elif "```" in clean_output:
                    start = clean_output.find("```") + 3
                    end = clean_output.find("```", start)
                    if end != -1:
                        clean_output = clean_output[start:end].strip()
                
                resume_data = json.loads(clean_output)
                
                # Display parsed information in a nice format
                st.subheader("📋 Parsed Resume Information")
                
                # Personal Information
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 👤 Personal Information")
                    if resume_data.get('full_name'):
                        st.write(f"**Name:** {resume_data['full_name']}")
                    if resume_data.get('email'):
                        st.write(f"**Email:** {resume_data['email']}")
                    if resume_data.get('phone'):
                        st.write(f"**Phone:** {resume_data['phone']}")
                
                with col2:
                    st.markdown("### 🎓 Education")
                    education = resume_data.get('education', [])
                    if education:
                        for edu in education:
                            if isinstance(edu, dict):
                                st.write(f"**{edu.get('degree', 'N/A')}**")
                                st.write(f"📍 {edu.get('institution', 'N/A')}")
                                st.write(f"📅 {edu.get('year', 'N/A')}")
                            else:
                                st.write(f"• {edu}")
                    else:
                        st.write("No education information found")
                
                # Work Experience
                st.markdown("### 💼 Work Experience")
                work_exp = resume_data.get('work_experience', [])
                if work_exp:
                    for job in work_exp:
                        if isinstance(job, dict):
                            st.markdown(f"**{job.get('job_title', 'N/A')}** at {job.get('company', 'N/A')}")
                            st.write(f"📅 {job.get('duration', 'N/A')}")
                            responsibilities = job.get('responsibilities', [])
                            if responsibilities:
                                st.write("**Key Responsibilities:**")
                                for resp in responsibilities:
                                    st.write(f"• {resp}")
                            st.write("---")
                        else:
                            st.write(f"• {job}")
                else:
                    st.write("No work experience found")
                
                # Skills and Certifications
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🛠️ Skills")
                    skills = resume_data.get('skills', [])
                    if skills:
                        skills_text = ", ".join(skills)
                        st.write(skills_text)
                    else:
                        st.write("No skills found")
                
                with col2:
                    st.markdown("### 🏆 Certifications")
                    certifications = resume_data.get('certifications', [])
                    if certifications:
                        for cert in certifications:
                            st.write(f"• {cert}")
                    else:
                        st.write("No certifications found")
                
                # Show raw JSON in expandable section
                with st.expander("View Raw JSON Data", expanded=False):
                    st.code(json.dumps(resume_data, indent=2), language='json')
                
                # Add voice controls for the entire resume at the end
                resume_summary = f"""
                Resume Summary for {resume_data.get('full_name', 'Unknown')}:
                Email: {resume_data.get('email', 'Not provided')}
                Phone: {resume_data.get('phone', 'Not provided')}
                
                Education: {', '.join([edu.get('degree', 'N/A') if isinstance(edu, dict) else str(edu) for edu in resume_data.get('education', [])])}
                
                Skills: {', '.join(resume_data.get('skills', []))}
                
                Work Experience: {len(resume_data.get('work_experience', []))} positions listed
                """
                
                add_voice_controls(resume_summary, "resume_summary")
                
            except json.JSONDecodeError:
                st.error("⚠️ Unable to parse the resume data. Showing raw output:")
                st.code(parsed_output, language='text')

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
                # Try to extract JSON from the response if it's wrapped in text
                ats_response = ats_analysis.strip()
                
                # Look for JSON content between ```json and ``` or just try direct parsing
                if "```json" in ats_response:
                    start = ats_response.find("```json") + 7
                    end = ats_response.find("```", start)
                    if end != -1:
                        ats_response = ats_response[start:end].strip()
                elif "```" in ats_response:
                    start = ats_response.find("```") + 3
                    end = ats_response.find("```", start)
                    if end != -1:
                        ats_response = ats_response[start:end].strip()
                
                # Try parsing JSON
                ats_data = json.loads(ats_response)
                score = ats_data.get('ats_score', 0)
                
                # Display ATS Score with visual indicators
                st.subheader("📊 ATS Analysis Results")
                
                # Progress bar for score
                st.subheader("� Score Breakdown")
                st.progress(score / 100)
                st.write(f"Your resume scored **{score} out of 100** points")
                
                # Add voice controls for ATS results at the end
                ats_summary = f"""
                ATS Analysis Results:
                Your resume scored {score} out of 100 points.
                Category: {ats_data.get('score_category', 'Unknown')}
                
                Matched Keywords: {', '.join(ats_data.get('matched_keywords', []))}
                
                Missing Keywords: {', '.join(ats_data.get('missing_keywords', []))}
                
                Overall Assessment: {ats_data.get('overall_assessment', 'No assessment provided')}
                """
                
                add_voice_controls(ats_summary, "ats_results")
                
                # Score display with color coding
                col1, col2, col3 = st.columns(3)
                with col1:
                    if score >= 90:
                        st.success(f"🟢 **{score}/100**")
                    elif score >= 80:
                        st.info(f"🟡 **{score}/100**")
                    elif score >= 70:
                        st.warning(f"🟠 **{score}/100**")
                    else:
                        st.error(f"🔴 **{score}/100**")
                    st.write("**ATS Score**")
                
                with col2:
                    score_category = ats_data.get('score_category', 'Unknown')
                    st.metric("Category", score_category)
                
                with col3:
                    match_level = "Strong" if score >= 80 else "Moderate" if score >= 70 else "Weak"
                    st.metric("Match Level", match_level)
                
                # Overall feedback
                if score >= 90:
                    st.success("🎉 **Excellent!** Your resume is highly likely to pass ATS screening.")
                elif score >= 80:
                    st.success("✅ **Good match!** Minor improvements could make it even better.")
                elif score >= 70:
                    st.warning("⚠️ **Fair match.** Some improvements needed to increase chances.")
                elif score >= 60:
                    st.warning("🔄 **Below average.** Significant improvements recommended.")
                else:
                    st.error("❌ **Poor match.** Major changes needed to improve ATS compatibility.")
                
                # Detailed Analysis in organized sections
                st.subheader("🔍 Detailed Analysis")
                
                # Keywords Analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### ✅ **Matched Keywords**")
                    matched_keywords = ats_data.get('matched_keywords', [])
                    if matched_keywords:
                        for keyword in matched_keywords:
                            st.success(f"✓ {keyword}")
                    else:
                        st.info("No keywords matched")
                
                with col2:
                    st.markdown("### ❌ **Missing Keywords**")
                    missing_keywords = ats_data.get('missing_keywords', [])
                    if missing_keywords:
                        for keyword in missing_keywords:
                            st.error(f"✗ {keyword}")
                    else:
                        st.success("No missing keywords identified")
                
                # Skills Gap Analysis
                if ats_data.get('skills_gap'):
                    st.markdown("### 🎯 **Skills Gap Analysis**")
                    st.write("Consider developing these skills:")
                    for skill in ats_data.get('skills_gap', []):
                        st.write(f"📚 {skill}")
                
                # Experience and Assessment
                col1, col2 = st.columns(2)
                with col1:
                    if ats_data.get('experience_alignment'):
                        st.markdown("### 💼 **Experience Alignment**")
                        st.write(ats_data.get('experience_alignment', ''))
                
                with col2:
                    if ats_data.get('overall_assessment'):
                        st.markdown("### 📝 **Overall Assessment**")
                        st.write(ats_data.get('overall_assessment', ''))
                
                # Recommendations
                if ats_data.get('recommendations'):
                    st.markdown("### 💡 **Recommendations**")
                    recommendations = ats_data.get('recommendations', [])
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                
                # Progress bar for score
                st.subheader("📈 Score Breakdown")
                progress_color = "green" if score >= 80 else "orange" if score >= 70 else "red"
                st.progress(score / 100)
                st.write(f"Your resume scored **{score} out of 100** points")
            except json.JSONDecodeError as e:
                st.error("⚠️ Error parsing ATS analysis. The AI response was not in valid JSON format.")
                st.write("**Debug Information:**")
                st.write(f"JSON Error: {str(e)}")
                
                # Show the raw response for debugging
                with st.expander("View Raw AI Response", expanded=False):
                    st.code(ats_analysis, language='text')
                
                # Try to extract useful information from the raw text
                st.subheader("Extracted Information")
                
                # Look for score patterns
                import re
                score_patterns = [
                    r"score[:\s]*(\d+)",
                    r"(\d+)/100",
                    r"(\d+)%",
                    r"ATS[:\s]*(\d+)"
                ]
                
                score_found = None
                for pattern in score_patterns:
                    matches = re.findall(pattern, ats_analysis, re.IGNORECASE)
                    if matches:
                        score_found = int(matches[0])
                        break
                
                if score_found:
                    st.metric("Estimated ATS Score", f"{score_found}/100")
                    if score_found >= 80:
                        st.success("Good match detected!")
                    elif score_found >= 70:
                        st.warning("Fair match detected.")
                    else:
                        st.error("Low match detected.")
                
                # Display the raw analysis text
                st.subheader("Analysis Summary")
                st.write(ats_analysis)
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
            
            # Add voice controls for improvement suggestions at the end
            add_voice_controls(improvement_suggestions, "improvement_tips")
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
            
            st.subheader("🎯 Skill Upgrade Recommendations")
            st.markdown(skill_suggestions)
            
            # Add voice controls for skill suggestions at the end
            add_voice_controls(skill_suggestions, "skill_suggestions")
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
            
            st.subheader("🗺️ Career Pathway Roadmap")
            st.markdown(roadmap)
            
            # Add voice controls for roadmap at the end
            add_voice_controls(roadmap, "job_roadmap")
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