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
        export_tailored_resume(job_description, output_format="markdown", output_file=None):
            Exports a tailored resume in the specified format (JSON or Markdown) to a file.
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

    def export_tailored_resume(self, job_description, output_format="markdown", output_file=None):
        """Export a tailored resume in the specified format."""
        tailored_resume = self.generate_tailored_resume(job_description)
        
        if output_format == "json":
            output = json.dumps(tailored_resume, indent=2)
            file_extension = "json"
        elif output_format == "markdown":
            output = self.generate_markdown_resume(tailored_resume)
            file_extension = "md"
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        if output_file:
            file_name = output_file if "." in output_file else f"{output_file}.{file_extension}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"tailored_resume_{timestamp}.{file_extension}"
        
        with open(file_name, 'w') as file:
            file.write(output)
        
        print(f"Resume exported to {file_name}")
        return file_name, output


def main():
    parser = argparse.ArgumentParser(description='Generate a tailored resume from JSON data.')
    parser.add_argument('--resume', default='resume_data.json', help='Path to the JSON resume data file')
    parser.add_argument('--job', required=True, help='Path to a text file containing the job description')
    parser.add_argument('--output', help='Output file name')
    parser.add_argument('--format', default='markdown', choices=['markdown', 'json'], help='Output format')

    args = parser.parse_args()

    try:
        with open(args.job, 'r') as file:
            job_description = file.read()
    except FileNotFoundError:
        print(f"Job description file {args.job} not found.")
        return

    generator = ResumeGenerator(args.resume)
    generator.export_tailored_resume(job_description, args.format, args.output)



if __name__ == "__main__":
    main()