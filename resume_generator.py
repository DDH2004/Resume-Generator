import json
import re
import argparse
import os
from datetime import datetime
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Pre-compile regex patterns
SKILL_PATTERNS = [
    r'python|java|javascript|js|html|css|c\+\+|ruby|php|swift|kotlin|go|rust|scala|sql',
    r'react|angular|vue|node|express|django|flask|spring|laravel|rails',
    r'aws|azure|gcp|docker|kubernetes|terraform|jenkins|ci/cd',
    r'machine learning|ml|ai|data science|nlp|computer vision',
    r'agile|scrum|kanban|waterfall|leadership|communication'
]
EXP_PATTERN = re.compile(r'(\d+)[\+]?\s+years?\s+(?:of\s+)?experience')
EDUCATION_PATTERN = re.compile(r"bachelor'?s?|master'?s?|phd|doctorate|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?")

# Set NLTK data path to a writable location in the project
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Download NLTK resources with proper error handling
def download_nltk_resources():
    """Download required NLTK resources and verify they're available"""
    # Resources needed for our processing
    resources = ['punkt', 'stopwords', 'punkt_tab']
    missing = []
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"Resource '{resource}' is already available.")
        except LookupError:
            missing.append(resource)
    
    if missing:
        print(f"Downloading missing NLTK resources: {', '.join(missing)}")
        for resource in missing:
            try:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
                print(f"Successfully downloaded '{resource}'")
            except Exception as e:
                print(f"ERROR: Failed to download '{resource}': {e}")
                return False
    
    return True

# Ensure NLTK resources are available
if not download_nltk_resources():
    print("WARNING: Some NLTK resources could not be downloaded. The application may not work correctly.")

class ResumeGenerator:
    """
    ResumeGenerator is a class designed to create, analyze, and tailor resumes based on job descriptions. 
    It provides functionality to load and save resume data, analyze job descriptions for key skills and 
    requirements, and generate tailored resumes in JSON or Markdown formats.
    
    Attributes:
        resume_data (dict): The resume data loaded from a JSON file or created as a template.
        stop_words (set): A set of stop words used for text processing in job description analysis.
        
    Methods:
        __init__(resume_file='resume_data.json'):
            Initializes the ResumeGenerator with a JSON resume file.
        load_resume_data(filename):
            Loads resume data from a JSON file. If the file is not found, 
            creates an example template.
        create_example_template():
            Creates and saves an example resume template in JSON format.
        save_resume_data(filename='resume_data.json'):
            Saves the current resume data to a JSON file.
        analyze_job_description(job_description):
            Extracts key skills, requirements, and frequently mentioned 
            terms from a job description.
        get_experience_years(work_item):
            Calculates the years of experience for a given work item based on 
            start and end dates.
        calculate_relevance_score(item, required_skills):
            Calculates a relevance score for a resume item based on its match with job requirements.
        generate_tailored_resume(job_description):
            Generates a tailored resume by analyzing a job description and 
            sorting resume sections by relevance.
        generate_markdown_resume(tailored_resume):
            Generates a Markdown version of the tailored resume.
        generate_html_resume(tailored_resume):
            Generates an HTML version of the tailored resume with CSS styling.
        export_tailored_resume(job_description, output_format="markdown", output_file=None, output_dir=None):
            Exports a tailored resume in the specified format (JSON, Markdown, HTML, or PDF) to a file.
    """

    def __init__(self, resume_file='resume_data.json'):
        """Initialize the resume generator with a JSON resume file."""
        self.resume_data = self.load_resume_data(resume_file)
        self.stop_words = set(stopwords.words('english'))

    def load_resume_data(self, filename):
        """Load resume data from JSON file."""
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Resume file {filename} not found. Creating example template.")
            return self.create_example_template()
        except json.JSONDecodeError:
            print(f"Error parsing {filename}. Creating example template.")
            return self.create_example_template()

    def create_example_template(self):
        """Create an example resume template."""
        template = {
            "basics": {
                "name": "Your Name",
                "label": "Your Title",
                "email": "your.email@example.com",
                "phone": "(555) 555-5555",
                "website": "https://yourwebsite.com",
                "summary": "Experienced professional with skills in X, Y, and Z.",
                "location": {
                    "address": "123 Main St",
                    "city": "Anytown",
                    "region": "State",
                    "postalCode": "12345",
                    "countryCode": "US"
                },
                "profiles": [
                    {
                        "network": "LinkedIn",
                        "username": "yourname",
                        "url": "https://linkedin.com/in/yourname"
                    },
                    {
                        "network": "GitHub",
                        "username": "yourusername",
                        "url": "https://github.com/yourusername"
                    }
                ]
            },
            "work": [
                {
                    "company": "Company Name",
                    "position": "Job Title",
                    "website": "https://company.com",
                    "startDate": "2018-01-01",
                    "endDate": "2021-01-01",
                    "summary": "Brief description of your role",
                    "highlights": [
                        "Accomplished X resulting in Y improvement",
                        "Managed team of X people",
                        "Developed X using Y technology"
                    ],
                    "keywords": ["leadership", "python", "teamwork"]
                }
            ],
            "education": [
                {
                    "institution": "University Name",
                    "area": "Major",
                    "studyType": "Bachelor",
                    "startDate": "2014-01-01",
                    "endDate": "2018-01-01",
                    "gpa": "3.8",
                    "courses": [
                        "Course 1",
                        "Course 2"
                    ],
                    "keywords": ["research", "thesis", "academic"]
                }
            ],
            "skills": [
                {
                    "name": "Web Development",
                    "level": "Advanced",
                    "keywords": ["HTML", "CSS", "JavaScript"]
                },
                {
                    "name": "Programming",
                    "level": "Advanced",
                    "keywords": ["Python", "Java", "C++"]
                }
            ],
            "projects": [
                {
                    "name": "Project Name",
                    "description": "Project description",
                    "highlights": [
                        "Developed X feature",
                        "Implemented Y technology"
                    ],
                    "keywords": ["python", "machine learning", "data analysis"],
                    "url": "https://project.com"
                }
            ],
            "certifications": [
                {
                    "name": "Certification Name",
                    "date": "2020-01-01",
                    "issuer": "Certification Authority",
                    "url": "https://certification.com",
                    "keywords": ["technical", "skill"]
                }
            ]
        }

        # Save the template
        with open('resume_data.json', 'w') as file:
            json.dump(template, file, indent=2)

        return template

    def save_resume_data(self, filename='resume_data.json'):
        """Save current resume data to JSON file."""
        with open(filename, 'w') as file:
            json.dump(self.resume_data, file, indent=2)

    def analyze_job_description(self, job_description):
        """Extract key skills and requirements from a job description."""
        try:
            # Tokenize and remove stopwords
            tokens = word_tokenize(job_description.lower())
            filtered_tokens = [w for w in tokens if w.isalnum() and w not in self.stop_words]
        except LookupError as e:
            # Fallback to simple tokenization if NLTK fails
            print(f"Warning: NLTK tokenization failed: {e}")
            print("Using simple tokenization instead.")
            # Simple tokenization fallback
            tokens = job_description.lower().split()
            filtered_tokens = [w for w in tokens if w.isalnum() and w not in self.stop_words]
        
        # Count word frequency
        word_freq = Counter(filtered_tokens)

        # Extract potential skills using regex (doesn't rely on NLTK)
        skills = set()
        for pattern in SKILL_PATTERNS:
            matches = re.findall(pattern, job_description.lower())
            skills.update(matches)

        # Look for years of experience
        experience_reqs = EXP_PATTERN.findall(job_description)

        # Look for education requirements
        education_reqs = EDUCATION_PATTERN.findall(job_description.lower())

        return {
            "frequent_words": word_freq.most_common(15),
            "skills": list(skills),
            "experience": experience_reqs,
            "education": education_reqs
        }

    def get_experience_years(self, work_item):
        """Calculate years of experience for a work item."""
        start = datetime.strptime(work_item["startDate"], "%Y-%m-%d")
        if work_item.get("endDate", "Present") == "Present":
            end = datetime.now()
        else:
            end = datetime.strptime(work_item["endDate"], "%Y-%m-%d")

        return (end - start).days / 365.25

    def calculate_relevance_score(self, item, required_skills):
        """Calculate relevance score for a resume item based on job requirements."""
        score = 0
        item_keywords = []
        
        # Extract keywords from the item
        if "keywords" in item:
            item_keywords.extend(item["keywords"])
        
        # Process text fields with fallback mechanisms
        def safe_tokenize(text):
            try:
                return word_tokenize(text.lower())
            except LookupError:
                # Simple fallback
                return text.lower().split()
        
        if "highlights" in item:
            for highlight in item["highlights"]:
                words = safe_tokenize(highlight)
                item_keywords.extend([w for w in words if w.isalnum()])
        
        if "summary" in item:
            words = safe_tokenize(item["summary"])
            item_keywords.extend([w for w in words if w.isalnum()])
        
        # Calculate match score based on keyword overlap
        for skill in required_skills:
            if any(skill in keyword.lower() for keyword in item_keywords):
                score += 1
        
        return score

    def generate_tailored_resume(self, job_description):
        """Generate a tailored resume based on job description."""
        # Analyze job description
        job_analysis = self.analyze_job_description(job_description)
        required_skills = job_analysis["skills"]

        # Create a copy of the resume data
        tailored_resume = self.resume_data.copy()

        # Sort work experience by relevance
        if "work" in tailored_resume:
            for item in tailored_resume["work"]:
                item["relevance_score"] = self.calculate_relevance_score(item, required_skills)

            tailored_resume["work"].sort(key=lambda x: x["relevance_score"], reverse=True)

            # Remove the relevance score as it's just used for sorting
            for item in tailored_resume["work"]:
                item.pop("relevance_score", None)

        # Sort projects by relevance
        if "projects" in tailored_resume:
            for item in tailored_resume["projects"]:
                item["relevance_score"] = self.calculate_relevance_score(item, required_skills)

            tailored_resume["projects"].sort(key=lambda x: x["relevance_score"], reverse=True)

            # Remove the relevance score
            for item in tailored_resume["projects"]:
                item.pop("relevance_score", None)

        # Sort skills by relevance to job description
        if "skills" in tailored_resume:
            for skill in tailored_resume["skills"]:
                skill["relevance_score"] = 0
                for req_skill in required_skills:
                    if req_skill in skill["name"].lower() or any(req_skill in kw.lower() for kw in skill.get("keywords", [])):
                        skill["relevance_score"] += 1
            
            tailored_resume["skills"].sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Remove the relevance score
            for skill in tailored_resume["skills"]:
                skill.pop("relevance_score", None)
        
        # Add job analysis info
        tailored_resume["job_analysis"] = job_analysis
        
        return tailored_resume
    
    def generate_markdown_resume(self, tailored_resume):
        """Generate a Markdown version of the tailored resume."""
        markdown = []
        
        # Header
        basics = tailored_resume.get("basics", {})
        markdown.append(f"# {basics.get('name', 'Your Name')}")
        markdown.append(f"## {basics.get('label', 'Your Title')}")
        
        # Contact Info
        contact_info = []
        if "email" in basics:
            contact_info.append(f"Email: {basics['email']}")
        if "phone" in basics:
            contact_info.append(f"Phone: {basics['phone']}")
        if "website" in basics:
            contact_info.append(f"Website: {basics['website']}")
        
        if contact_info:
            markdown.append("\n" + " | ".join(contact_info) + "\n")
        
        # Summary
        if "summary" in basics:
            markdown.append("## Summary")
            markdown.append(basics["summary"] + "\n")
        
        # Skills
        if "skills" in tailored_resume:
            markdown.append("## Skills")
            for skill in tailored_resume["skills"]:
                markdown.append(f"- **{skill['name']}:** {', '.join(skill.get('keywords', []))}")
            markdown.append("")
        
        # Work Experience
        if "work" in tailored_resume:
            markdown.append("## Work Experience")
            for job in tailored_resume["work"]:
                position = job.get("position", "")
                company = job.get("company", "")
                markdown.append(f"### {position} at {company}")
                
                dates = []
                if "startDate" in job:
                    start_date = job["startDate"].split("-")[0]  # Just get the year
                    dates.append(start_date)
                if "endDate" in job:
                    end_date = job["endDate"].split("-")[0] if job["endDate"] != "Present" else "Present"
                    dates.append(end_date)
                
                if dates:
                    markdown.append(f"_{' - '.join(dates)}_")
                
                if "summary" in job:
                    markdown.append(f"\n{job['summary']}")
                
                if "highlights" in job:
                    markdown.append("\nKey Achievements:")
                    for highlight in job["highlights"]:
                        markdown.append(f"- {highlight}")
                markdown.append("")
        
        # Projects
        if "projects" in tailored_resume and tailored_resume["projects"]:
            markdown.append("## Projects")
            for project in tailored_resume["projects"]:
                markdown.append(f"### {project.get('name', 'Project')}")
                
                if "description" in project:
                    markdown.append(project["description"])
                
                if "highlights" in project:
                    markdown.append("\nHighlights:")
                    for highlight in project["highlights"]:
                        markdown.append(f"- {highlight}")
                
                if "url" in project:
                    markdown.append(f"\n[Project Link]({project['url']})")
                markdown.append("")
        
        # Education
        if "education" in tailored_resume:
            markdown.append("## Education")
            for edu in tailored_resume["education"]:
                degree = edu.get("studyType", "")
                area = edu.get("area", "")
                institution = edu.get("institution", "")
                markdown.append(f"### {degree} in {area}, {institution}")
                
                dates = []
                if "startDate" in edu:
                    start_date = edu["startDate"].split("-")[0]  # Just get the year
                    dates.append(start_date)
                if "endDate" in edu:
                    end_date = edu["endDate"].split("-")[0]
                    dates.append(end_date)
                
                if dates:
                    markdown.append(f"_{' - '.join(dates)}_")
                
                if "gpa" in edu:
                    markdown.append(f"\nGPA: {edu['gpa']}")
                
                if "courses" in edu and edu["courses"]:
                    markdown.append("\nRelevant Coursework:")
                    markdown.append(", ".join(edu["courses"]))
                markdown.append("")
        
        # Certifications
        if "certifications" in tailored_resume and tailored_resume["certifications"]:
            markdown.append("## Certifications")
            for cert in tailored_resume["certifications"]:
                name = cert.get("name", "")
                issuer = cert.get("issuer", "")
                date = cert.get("date", "").split("-")[0] if "date" in cert else ""
                
                markdown.append(f"- **{name}** - {issuer} ({date})")
            markdown.append("")
        
        # Job match analysis
        if "job_analysis" in tailored_resume:
            markdown.append("## Job Match Analysis")
            markdown.append("_This section is for your reference and should be removed before sending the resume._\n")
            
            analysis = tailored_resume["job_analysis"]
            
            markdown.append("### Key Skills Detected")
            for skill in analysis["skills"]:
                markdown.append(f"- {skill}")
            markdown.append("")
            
            if analysis["experience"]:
                markdown.append("### Experience Requirements")
                for exp in analysis["experience"]:
                    markdown.append(f"- {exp} years of experience")
                markdown.append("")
            
            if analysis["education"]:
                markdown.append("### Education Requirements")
                for edu in analysis["education"]:
                    markdown.append(f"- {edu.capitalize()} degree")
                markdown.append("")
            
            markdown.append("### Frequently Mentioned Terms")
            for word, count in analysis["frequent_words"]:
                markdown.append(f"- {word}: {count} mentions")
        
        return "\n".join(markdown)

    def generate_html_resume(self, tailored_resume):
        """Generate an HTML version of the tailored resume with CSS styling."""
        html = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '  <meta charset="UTF-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '  <title>Professional Resume</title>',
            '  <style>',
            '    body { font-family: \'Calibri\', \'Arial\', sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }',
            '    h1 { color: #2a5885; margin-bottom: 5px; }',
            '    h2 { color: #2a5885; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 20px; }',
            '    h3 { margin-bottom: 0; }',
            '    .contact-info { display: flex; justify-content: space-between; flex-wrap: wrap; margin-bottom: 20px; }',
            '    .contact-item { margin-right: 20px; }',
            '    .date { color: #777; font-style: italic; margin: 0; }',
            '    .job-title { margin-bottom: 0; }',
            '    .company { margin-top: 0; }',
            '    ul { padding-left: 20px; }',
            '    li { margin-bottom: 5px; }',
            '    .skills-container { display: flex; flex-wrap: wrap; }',
            '    .skill-category { width: 48%; margin-right: 2%; margin-bottom: 15px; }',
            '    .section { margin-bottom: 20px; }',
            '    .project { margin-bottom: 15px; }',
            '    .match-analysis { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }',
            '    @media print {',
            '      body { padding: 0; }',
            '      .match-analysis { display: none; }',
            '    }',
            '  </style>',
            '</head>',
            '<body>'
        ]
        
        # Header
        basics = tailored_resume.get("basics", {})
        html.append(f'  <div class="header-section">')
        html.append(f'    <h1>{basics.get("name", "Your Name")}</h1>')
        if basics.get("label"):
            html.append(f'    <p>{basics.get("label")}</p>')
        
        # Contact Info
        html.append('    <div class="contact-info">')
        if "email" in basics:
            html.append(f'      <div class="contact-item">📧 {basics["email"]}</div>')
        if "phone" in basics:
            html.append(f'      <div class="contact-item">📱 {basics["phone"]}</div>')
        if "website" in basics:
            html.append(f'      <div class="contact-item">🌐 <a href="{basics["website"]}">{basics["website"]}</a></div>')
        if "location" in basics:
            location = basics["location"]
            location_str = f'{location.get("city", "")}, {location.get("region", "")}'
            html.append(f'      <div class="contact-item">📍 {location_str}</div>')
        html.append('    </div>')
        
        # Profiles
        if "profiles" in basics and basics["profiles"]:
            html.append('    <div class="contact-info">')
            for profile in basics["profiles"]:
                html.append(f'      <div class="contact-item">{profile["network"]}: <a href="{profile["url"]}">{profile.get("username", "Profile")}</a></div>')
            html.append('    </div>')
        html.append('  </div>')
        
        # Summary
        if "summary" in basics:
            html.append('  <div class="section">')
            html.append('    <h2>Summary</h2>')
            html.append(f'    <p>{basics["summary"]}</p>')
            html.append('  </div>')
        
        # Skills
        if "skills" in tailored_resume and tailored_resume["skills"]:
            html.append('  <div class="section">')
            html.append('    <h2>Skills</h2>')
            html.append('    <div class="skills-container">')
            
            for skill in tailored_resume["skills"]:
                html.append('      <div class="skill-category">')
                html.append(f'        <h3>{skill["name"]}</h3>')
                if "keywords" in skill and skill["keywords"]:
                    html.append('        <ul>')
                    for keyword in skill["keywords"]:
                        html.append(f'          <li>{keyword}</li>')
                    html.append('        </ul>')
                html.append('      </div>')
            
            html.append('    </div>')
            html.append('  </div>')
        
        # Work Experience
        if "work" in tailored_resume and tailored_resume["work"]:
            html.append('  <div class="section">')
            html.append('    <h2>Work Experience</h2>')
            
            for job in tailored_resume["work"]:
                html.append('    <div class="job">')
                position = job.get("position", "")
                company = job.get("name", "")  # Using 'name' field
                
                html.append(f'      <h3>{position}</h3>')
                html.append(f'      <p class="company">{company}</p>')
                
                dates = []
                if "startDate" in job:
                    start_date = job["startDate"].split("-")[0] if "-" in job["startDate"] else job["startDate"]
                    dates.append(start_date)
                if "endDate" in job:
                    end_date = job["endDate"].split("-")[0] if "-" in job["endDate"] else job["endDate"]
                    dates.append(end_date)
                
                if dates:
                    html.append(f'      <p class="date">{" - ".join(dates)}</p>')
                
                if "summary" in job:
                    html.append(f'      <p>{job["summary"]}</p>')
                
                if "highlights" in job and job["highlights"]:
                    html.append('      <ul>')
                    for highlight in job["highlights"]:
                        html.append(f'        <li>{highlight}</li>')
                    html.append('      </ul>')
                
                html.append('    </div>')
            
            html.append('  </div>')
        
        # Add other sections (education, projects, certifications) similarly
        
        # Education
        if "education" in tailored_resume and tailored_resume["education"]:
            html.append('  <div class="section">')
            html.append('    <h2>Education</h2>')
            
            for edu in tailored_resume["education"]:
                html.append('    <div class="education">')
                degree = edu.get("studyType", "")
                area = edu.get("area", "")
                institution = edu.get("institution", "")
                
                html.append(f'      <h3>{degree} in {area}</h3>')
                html.append(f'      <p class="company">{institution}</p>')
                
                dates = []
                if "startDate" in edu:
                    start_date = edu["startDate"].split("-")[0] if "-" in edu["startDate"] else edu["startDate"]
                    dates.append(start_date)
                if "endDate" in edu:
                    end_date = edu["endDate"].split("-")[0] if "-" in edu["endDate"] else edu["endDate"]
                    dates.append(end_date)
                
                if dates:
                    html.append(f'      <p class="date">{" - ".join(dates)}</p>')
                
                if "gpa" in edu:
                    html.append(f'      <p>GPA: {edu["gpa"]}</p>')
                
                if "courses" in edu and edu["courses"]:
                    html.append('      <p><strong>Relevant Coursework:</strong></p>')
                    html.append('      <ul>')
                    for course in edu["courses"]:
                        html.append(f'        <li>{course}</li>')
                    html.append('      </ul>')
                
                html.append('    </div>')
            
            html.append('  </div>')
        
        # Projects
        if "projects" in tailored_resume and tailored_resume["projects"]:
            html.append('  <div class="section">')
            html.append('    <h2>Projects</h2>')
            
            for project in tailored_resume["projects"]:
                html.append('    <div class="project">')
                html.append(f'      <h3>{project.get("name", "")}</h3>')
                
                if "description" in project:
                    html.append(f'      <p>{project["description"]}</p>')
                
                if "highlights" in project and project["highlights"]:
                    html.append('      <ul>')
                    for highlight in project["highlights"]:
                        html.append(f'        <li>{highlight}</li>')
                    html.append('      </ul>')
                
                if "url" in project:
                    html.append(f'      <p><a href="{project["url"]}" target="_blank">Project Link</a></p>')
                
                html.append('    </div>')
            
            html.append('  </div>')
        
        # Job Match Analysis
        if "job_analysis" in tailored_resume:
            html.append('  <div class="section match-analysis">')
            html.append('    <h2>Job Match Analysis</h2>')
            html.append('    <p><em>This section is for your reference and will not appear when printed.</em></p>')
            
            analysis = tailored_resume["job_analysis"]
            
            if analysis["skills"]:
                html.append('    <h3>Key Skills Detected</h3>')
                html.append('    <ul>')
                for skill in analysis["skills"]:
                    html.append(f'      <li>{skill}</li>')
                html.append('    </ul>')
            
            if analysis["experience"]:
                html.append('    <h3>Experience Requirements</h3>')
                html.append('    <ul>')
                for exp in analysis["experience"]:
                    html.append(f'      <li>{exp} years of experience</li>')
                html.append('    </ul>')
            
            if analysis["education"]:
                html.append('    <h3>Education Requirements</h3>')
                html.append('    <ul>')
                for edu in analysis["education"]:
                    html.append(f'      <li>{edu.capitalize()} degree</li>')
                html.append('    </ul>')
            
            html.append('    <h3>Frequently Mentioned Terms</h3>')
            html.append('    <ul>')
            for word, count in analysis["frequent_words"]:
                html.append(f'      <li>{word}: {count} mentions</li>')
            html.append('    </ul>')
            
            html.append('  </div>')
        
        html.append('</body>')
        html.append('</html>')
        
        return '\n'.join(html)

    def export_tailored_resume(self, job_description, output_format="markdown", output_file=None, output_dir=None):
        """Export a tailored resume in the specified format."""
        tailored_resume = self.generate_tailored_resume(job_description)
        
        if output_format == "json":
            output = json.dumps(tailored_resume, indent=2)
            file_extension = "json"
        elif output_format == "markdown":
            output = self.generate_markdown_resume(tailored_resume)
            file_extension = "md"
        elif output_format == "html":
            output = self.generate_html_resume(tailored_resume)
            file_extension = "html"
        elif output_format == "pdf":
            # Generate HTML first
            html_content = self.generate_html_resume(tailored_resume)
            file_extension = "pdf"
            
            try:
                import weasyprint
                output = html_content  # We'll use this HTML directly when creating the PDF file
            except ImportError:
                print("WARNING: WeasyPrint not installed. Falling back to HTML output.")
                print("To enable PDF output, install WeasyPrint with: pip install weasyprint")
                output = html_content
                file_extension = "html"
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Determine file name
        if output_file:
            file_name = output_file if "." in output_file else f"{output_file}.{file_extension}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"tailored_resume_{timestamp}.{file_extension}"
        
        # Handle directory
        if output_dir:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, file_name)
        else:
            file_path = file_name
        
        # Write the file based on format
        if output_format == "pdf":
            try:
                import weasyprint
                html = weasyprint.HTML(string=output)
                html.write_pdf(file_path)
            except ImportError:
                # Already handled above
                with open(file_path, 'w') as file:
                    file.write(output)
        else:
            with open(file_path, 'w') as file:
                file.write(output)
        
        print(f"Resume exported to {file_path}")
        return file_path, output


def main():
    parser = argparse.ArgumentParser(description='Generate a tailored resume from JSON data.')
    parser.add_argument('--resume', default='resume_data.json', help='Path to the JSON resume data file')
    parser.add_argument('--job', required=True, help='Path to a text file containing the job description')
    parser.add_argument('--output', help='Output file name')
    parser.add_argument('--output-dir', help='Target directory for the generated resume')
    parser.add_argument('--format', default='markdown', choices=['markdown', 'json', 'html', 'pdf'], help='Output format')

    args = parser.parse_args()

    try:
        with open(args.job, 'r') as file:
            job_description = file.read()
    except FileNotFoundError:
        print(f"Job description file {args.job} not found.")
        return

    generator = ResumeGenerator(args.resume)
    generator.export_tailored_resume(job_description, args.format, args.output, args.output_dir)


if __name__ == "__main__":
    main()