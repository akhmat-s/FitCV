from __future__ import annotations
"""cv_generator.py

A complete object model for generating a personalized resume (CV) with an LLM
and exporting it to PDF for printing. 🟡‑🟣‑🔴‑🟠‑🟢‑🔵 correspond to the color-coding
convention from the instruction.

Instruction for the LLM
=======================
The LLM receives a JSON template mirroring this object model and fills in the fields
according to the docstrings of *each* class.

* `summary.text` — **3‑5 lines**. Personalize it as much as possible for the job
  (`job_target`).
* The number of pages must not exceed `CV.max_pages` (2 by default).
* Use only the action verbs from `ActionVerb`.
* Write numbers (`BulletPoint.impact`) in the format "+20%", "‑30 sec", etc.
* Black text, white background, 1‑2 fonts.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import TYPE_CHECKING, List, Optional
import re

if TYPE_CHECKING:
    # Type-only import: SkillProvenance lives in cv_generator, which imports this module — a
    # runtime import would be circular. `from __future__ import annotations` keeps the field
    # annotation a string, so this is never evaluated at runtime.
    from cv_generator import SkillProvenance

# ──────────────────────────────────────────────────────────────────────────────
#  BASIC STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

class ActionVerb(str, Enum):
    """🔴 List of allowed action verbs for BulletPoint.action_verb.
    
    Use strong action verbs that demonstrate your contribution and initiative.
    Avoid weak verbs like "Participated", "Assisted", "Helped".
    """

    # Core recommended verbs
    COLLABORATED = "Collaborated"
    MANAGED = "Managed"
    IMPROVED = "Improved"
    DEVELOPED = "Developed"
    CREATED = "Created"
    LED = "Led"
    DESIGNED = "Designed"
    DELIVERED = "Delivered"
    OPTIMIZED = "Optimized"
    BUILT = "Built"
    
    # Additional verbs to add variety to the experience
    IMPLEMENTED = "Implemented"
    LAUNCHED = "Launched"
    ARCHITECTED = "Architected"
    STREAMLINED = "Streamlined"
    AUTOMATED = "Automated"
    ENHANCED = "Enhanced"
    MENTORED = "Mentored"
    REDUCED = "Reduced"
    INCREASED = "Increased"
    TRANSFORMED = "Transformed"
    ACHIEVED = "Achieved"
    ESTABLISHED = "Established"
    SPEARHEADED = "Spearheaded"
    REVAMPED = "Revamped"
    RESEARCHED = "Researched"
    PIONEERED = "Pioneered"


@dataclass
class Link:
    """A hyperlink in the resume.

    Attributes
    ----------
    title : str
        The link text (for example, "LinkedIn").
    url : str
        The full URL (https://…).
    """

    url: str  #: Full URL (https://…); a link without a URL is meaningless — required.
    title: Optional[str] = None  #: Link text (optional; bare URL may have no label).

    URL_REGEX: re.Pattern[str] = re.compile(r"^https?://.+", re.I)

    def is_valid_format(self) -> bool:  # noqa: D401
        """Checks that the string looks like a URL."""
        return bool(self.URL_REGEX.fullmatch(self.url))


@dataclass
class PersonalInfo:
    """Contact block at the top of the CV."""

    name: str  #: Full name (uppercase string).
    email: str  #: Work email address.
    location: Optional[str] = None  #: City/country (optional; many CVs omit it).
    phone: Optional[str] = None  #: Phone (optional; ATS contact parsing).
    links: List[Link] = field(default_factory=list)  #: Validated URL links.


@dataclass
class Summary:
    """🟡 A personalized summary tailored to the job (3‑5 lines).
    
    The summary must be tailored to the job as much as possible (job_target).
    Include only the skills and experience that are directly related to the job
    requirements. Avoid generic phrases. Use concrete examples and achievements.
    """

    text: str
    relevant_skills: List[str] = field(default_factory=list)

    def line_count(self) -> int:
        """Estimate visual lines as max(newline-delimited lines, sentence count).

        Counting only "\\n" scored a well-formed single-paragraph summary as 1 line,
        so a truthful 3-5 sentence summary failed the range check and was capped every run.
        Taking the max keeps explicit line breaks honest while recognizing sentence prose.
        """
        stripped = self.text.strip()
        if not stripped:
            return 0
        newline_lines = len(stripped.splitlines())
        sentences = [s for s in re.split(r"[.!?]+(?:\s|$)", stripped) if s.strip()]
        return max(newline_lines, len(sentences))


@dataclass
class Category:
    """🟡🟠 A single named group of skills in the Skills section.

    The heading (`category`) is derived from the job structure (Languages / Tools /
    Frameworks / Concepts / …); `keywords` is a dense list of skills phrased in the JD wording.
    """

    category: str  #: Domain heading of the group (derived from the JD)
    keywords: List[str] = field(default_factory=list)  #: Group skills in the JD vocabulary


@dataclass
class Skills:
    """Skills section — JD-derived domain categories.

    Replaces the fixed Relevant/Hard/Soft buckets with a list of `Category`, so that
    the render matches the "gold standard" (bold domain headings, dense lines).
    """

    categories: List[Category] = field(default_factory=list)
    #: Validation-time evidence provenance per surfaced keyword (two-tier model). Carried so
    #: the deterministic Skills validator can trace a competency keyword to its CV anchor.
    #: NEVER rendered, copied, mapped to the API response, or scored — render reads only
    #: ``categories[].keywords`` (plain strings).
    provenance: list[SkillProvenance] = field(default_factory=list)


@dataclass
class BulletPoint:
    """A bullet in the `Experience.bullets` section. Rendered with a "•" marker.

    Example::

        BulletPoint(
            action_verb=ActionVerb.DEVELOPED,
            description="a UI Kit library with Storybook, decreasing UI dev time by 30%",
            skills=["React", "Storybook"],
            impact="‑30% time",  # 🟢 number
            benefit="Ensured design consistency"  # 🔵 benefit
        )
    """

    action_verb: ActionVerb  #: 🔴 Action verb from the ActionVerb list
    description: str  #: Free-form text (≤ 1 line when printed).
    skills: List[str] = field(default_factory=list)  #: 🟠 Skills used here
    impact: Optional[str] = None  #: 🟢 Quantified results ("+20%", "-30 sec", etc.)
    benefit: Optional[str] = None  #: 🔵 Benefit to the company, business outcome


@dataclass
class Experience:
    """Description of a single job."""

    role: str  #: Job title
    company: str  #: Company name
    start_date: Optional[str] = None  #: YYYY‑MM (optional; the source may not provide a date)
    end_date: Optional[str] = None  #: YYYY‑MM/"Present" (optional; a current role may have no date)
    company_description: Optional[str] = None  #: 🟣 "Selling" pitch for the company (optional)
    location: Optional[str] = None  #: City/country
    bullets: List[BulletPoint] = field(default_factory=list)  #: Key achievements


@dataclass
class Education:
    """Formal education."""

    institution: str
    degree: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    gpa: Optional[str] = None  #: "5.0/5", "3.8/4.0", etc.


@dataclass
class Project:
    """Optional section for notable pet/OSS projects."""

    name: str
    description: Optional[str] = None  #: Short description (optional; a project may have no blurb).
    skills: List[str] = field(default_factory=list)
    link: Optional[Link] = None


@dataclass
class Certificate:
    """Optional certificates (Coursera, AWS, PMP…)."""

    title: str
    issuer: Optional[str] = None  #: Issuer (optional; a certificate may not state the issuer).
    year: Optional[int] = None  #: Year (optional; a certificate may not state the year).
    link: Optional[Link] = None


@dataclass
class Language:
    """Foreign languages.
    
    List only languages relevant to the position. For international companies,
    always specify the English proficiency level.
    """

    language: str  #: Language name
    level: Optional[str] = None  #: "Native", "Fluent", "B2", etc. (optional; may be omitted).
    
    def is_valid(self) -> bool:
        """Checks that the language proficiency level is valid."""
        valid_levels = [
            "Native", "Fluent", "Professional", "Intermediate", "Basic",
            "A1", "A2", "B1", "B2", "C1", "C2"
        ]
        return self.level in valid_levels


@dataclass
class JobTarget:
    """Data about the job the CV is being personalized for."""

    title: str  #: "Senior Frontend Developer"
    company: str  #: "Acme Inc."
    keywords: List[str] = field(default_factory=list)  #: must‑have JD skills


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN CV CLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CVTemplate:
    """Complete CV model with validation and export."""

    personal_info: PersonalInfo
    summary: Summary
    skills: Skills
    experiences: List[Experience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    certificates: List[Certificate] = field(default_factory=list)
    languages: List[Language] = field(default_factory=list)

    #: Level-driven render order (TargetSection values); set by assemble_and_gate.
    section_order: List[str] = field(default_factory=list)

    # ─── PDF layout metadata ────────────────────────────────────────
    max_pages: int = 1  # At most 1 page
    file_format: str = "PDF"
    font_family: List[str] = field(default_factory=lambda: ["Helvetica", "Arial"])
    text_color: str = "black"
    background_color: str = "white"

    # ────────────────────────────────── utils

    def to_dict(self) -> dict:
        """Serializes the dataclass tree into a plain dict."""

        return asdict(self)

    # ─── Validations ────────────────────────────────────────────────────

    def validate(self, job_target: Optional[JobTarget] = None) -> None:
        """Full pass over the fields; raises ValueError on violations.
        
        Centralized validation of the entire CV. Includes:
        - Summary check (length 3-5 lines)
        - Action verb check in bullet points
        - URL formatting check
        - Language check
        - CV size check
        - Bullet point check (presence of impact/benefit)
        - Company description check
        - Job skill coverage check (if job_target is provided)
        
        Parameters
        ----------
        job_target : Optional[JobTarget]
            Information about the target job for checking skill coverage.
            
        Raises
        ------
        ValueError
            When problems with the CV are detected.
        """
        warnings = []
        errors = []

        # 1. Summary check (length)
        lines = self.summary.line_count()
        if not (3 <= lines <= 5):
            errors.append(f"Summary must be 3-5 lines (actual: {lines}).")

        # 2. URL check in links
        for link in self.personal_info.links:
            if not link.is_valid_format():
                errors.append(f"Invalid URL format: {link.url}")

        # 3. Language check
        valid_language_levels = [
            "Native", "Fluent", "Professional", "Intermediate", "Basic",
            "A1", "A2", "B1", "B2", "C1", "C2"
        ]
        for lang in self.languages:
            # A missing level is a truthful absence (the source omitted proficiency), not an error;
            # only a PRESENT-but-unrecognized level is flagged (mirrors live _validate_languages).
            if lang.level and lang.level not in valid_language_levels:
                errors.append(f"Invalid language level '{lang.level}' for '{lang.language}'")

        # 4. CV size check
        if len(self.experiences) > 3:
            warnings.append(f"Large number of work experiences ({len(self.experiences)}). The CV should fit on 1 page.")

        # 5. Action verb and bullet point check
        for exp in self.experiences:
            # 5.1 company_description check (Optional — skip when absent, no crash)
            if exp.company_description and len(exp.company_description) < 20:
                warnings.append(f"Company description is too short: {exp.company_description}")
                
            # 5.2 The description should contain specifics
            if exp.company_description and not any(c.isdigit() for c in exp.company_description) and not any(
                term in exp.company_description.lower() 
                for term in ["largest", "leading", "top", "best", "premier", "крупнейший", "ведущий", "топ"]
            ):
                warnings.append(f"Consider adding specifics to the description of {exp.company} (market position, size, etc.)")
                
            # 5.3 Check for the presence of bullets
            if not exp.bullets:
                warnings.append(f"No achievements (bullet points) in the experience at {exp.company}")
            elif len(exp.bullets) < 2:
                warnings.append(f"Consider listing at least 2 achievements at {exp.company}")
            
            # 5.4 Check each bullet point
            for i, bp in enumerate(exp.bullets):
                if not isinstance(bp.action_verb, ActionVerb):
                    warnings.append(f"action_verb outside ActionVerb at {exp.company}: {bp.action_verb}")
                
                # impact check (should contain numbers)
                if bp.impact and not any(c.isdigit() for c in bp.impact):
                    warnings.append(f"Impact should contain numbers: {bp.impact} at {exp.company}")
                
                # Warn if there is no impact or benefit
                if not bp.impact and not bp.benefit:
                    warnings.append(f"Consider specifying impact (🟢) or benefit (🔵) in bullet point #{i+1} at {exp.company}")
                
                # description length
                if len(bp.description) > 120:
                    warnings.append(f"Description is too long ({len(bp.description)} characters) at {exp.company}")

        # 6. Job skill coverage check (if job_target is provided)
        if job_target and job_target.keywords:
            rendered_skills = [
                keyword for category in self.skills.categories for keyword in category.keywords
            ]
            missing_skills = []
            for keyword in job_target.keywords:
                if keyword.lower() not in [skill.lower() for skill in rendered_skills]:
                    missing_skills.append(keyword)
                    
            if missing_skills:
                warnings.append(f"Missing key skills from the job posting: {', '.join(missing_skills)}")
                
            # title check in summary
            if job_target.title and job_target.title.lower() not in self.summary.text.lower():
                warnings.append(f"Job title from the posting ({job_target.title}) is missing from the summary")

        # Print the warnings
        if warnings:
            print("CV validation warnings:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        # Raise errors if there are any
        if errors:
            raise ValueError("Problems detected with the CV:\n" + "\n".join(errors))
            
            
    # ─── Export ───────────────────────────────────────────────────────

    def to_html(self) -> str:
        """Generates a minimal HTML template (used by WeasyPrint)."""

        import html

        def h(tag: str, text: str, classes: str = "") -> str:
            if classes:
                return f'<{tag} class="{classes}">{html.escape(text)}</{tag}>'
            return f"<{tag}>{html.escape(text)}</{tag}>"

        # Base CSS to improve the appearance
        css = '''
        <style>
            body {
                font-family: Arial, Helvetica, sans-serif;
                color: black;
                background-color: white;
                margin: 0;
                padding: 30px;
                line-height: 1.4;
                max-width: 800px;
                margin: 0 auto;
                font-size: 12px;
            }
            h1 {
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 5px;
                text-transform: uppercase;
            }
            .contact-info {
                text-align: center;
                margin-bottom: 20px;
                font-size: 14px;
            }
            .contact-info a {
                color: #0073b1;
                text-decoration: none;
            }
            h2 {
                font-size: 18px;
                margin-top: 20px;
                margin-bottom: 10px;
                border-bottom: 1px solid #000;
                padding-bottom: 5px;
            }
            h3 {
                margin-bottom: 0;
                font-size: 16px;
                margin-top: 0;
                font-weight: bold;
            }
            .company-desc {
                font-style: italic;
                margin-top: 2px;
                margin-bottom: 5px;
                font-size: 14px;
                color: #555;
            }
            .company-name {
                font-style: italic;
                font-weight: normal;
                color: #555;
            }
            .date-location {
                margin-top: 0;
                margin-bottom: 5px;
                font-size: 14px;
                text-align: right;
                float: right;
            }
            .location {
                text-align: right;
                float: right;
                margin-top: 0;
            }
            .clear {
                clear: both;
            }
            ul {
                margin-top: 5px;
                padding-left: 20px;
                list-style-type: disc;
            }
            li {
                margin-bottom: 5px;
                padding-left: 5px;
            }
            .skills-section {
                margin-bottom: 5px;
            }
            .skills-category {
                font-weight: bold;
                margin-right: 5px;
            }
            .summary {
                margin-bottom: 20px;
                line-height: 1.5;
            }
            .job-header {
                margin-bottom: 5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .job-title {
                font-weight: bold;
                font-size: 16px;
            }
            .experience-section {
                margin-bottom: 15px;
            }
            .education-line {
                display: flex;
                justify-content: space-between;
                margin-bottom: 3px;
            }
            .gpa {
                margin-top: 0;
            }
            .section-divider {
                border-bottom: 1px solid #000;
                margin: 20px 0;
            }
        </style>
        '''

        parts: List[str] = [css]

        # Header
        parts.append(h("h1", self.personal_info.name.upper()))
        parts.append('<div class="contact-info">')
        contact_parts = (
            [html.escape(self.personal_info.location)] if self.personal_info.location else []
        )

        if self.personal_info.email:
            contact_parts.append(html.escape(self.personal_info.email))
            
        if self.personal_info.links:
            for link in self.personal_info.links:
                contact_parts.append(f'<a href="{html.escape(link.url)}">{html.escape(link.title or link.url)}</a>')
                
        parts.append(" • ".join(contact_parts))
        parts.append('</div>')

        # Summary
        parts.append('<div class="summary">')
        parts.append(h("div", self.summary.text))
        parts.append('</div>')

        # Experience Section
        parts.append(h("h2", "Experience"))
        
        for exp in self.experiences:
            parts.append('<div class="experience-section">')
            
            # Job header with role and location
            parts.append('<div class="job-header">')
            parts.append(f'<div class="job-title">{html.escape(exp.role)}</div>')
            if exp.location:
                parts.append(f'<div class="location">{html.escape(exp.location)}</div>')
            parts.append('</div>')
            
            # Company and dates
            parts.append('<div class="job-header">')
            parts.append(f'<div class="company-name">{html.escape(exp.company)} - <span class="company-desc">{html.escape(exp.company_description or "")}</span></div>')
            date_span = " - ".join(html.escape(d) for d in (exp.start_date, exp.end_date) if d)
            if date_span:
                parts.append(f'<div class="date-location">{date_span}</div>')
            parts.append('</div>')
            
            # Bullets
            if exp.bullets:
                parts.append('<ul>')
                for bp in exp.bullets:
                    bullet_text = f"{bp.action_verb.value} {bp.description}"
                    
                    # Add impact if present
                    if bp.impact:
                        bullet_text += f" by {bp.impact}"
                        
                    # Add benefit if present
                    if bp.benefit:
                        bullet_text += f", {bp.benefit}"
                        
                    parts.append(f'<li>{html.escape(bullet_text)}</li>')
                parts.append('</ul>')
                
            parts.append('</div>')  # Close experience-section

        # Education
        if self.education:
            parts.append(h("h2", "Education"))
            
            for ed in self.education:
                parts.append('<div class="education-line">')
                parts.append(f'<div><strong>{html.escape(ed.institution)}</strong></div>')
                parts.append(f'<div>{html.escape(ed.location if hasattr(ed, "location") else "Moscow")}</div>')
                parts.append('</div>')
                
                parts.append('<div class="education-line">')
                parts.append(f'<div>{html.escape(ed.degree)}</div>')
                year_text = ""
                if ed.end_year:
                    year_text = str(ed.end_year)
                if ed.start_year and ed.end_year:
                    year_text = f"{ed.start_year} - {ed.end_year}"
                parts.append(f'<div>{year_text}</div>')
                parts.append('</div>')
                
                if ed.gpa:
                    parts.append(f'<p class="gpa">GPA: {html.escape(ed.gpa)}</p>')

        # Skills
        if self.skills:
            parts.append(h("h2", "Skills"))

            # JD-derived domain categories, in order.
            skill_categories = {
                category.category: category.keywords for category in self.skills.categories
            }

            # Add languages from self.languages to Language Skills category
            language_skills = []
            for lang in self.languages:
                level = f" {lang.level.lower()}" if lang.level else ""
                language_skills.append(f"{lang.language}{level}")

            if language_skills:
                skill_categories["Language Skills"] = language_skills

            # Output skills by category
            for category, skills in skill_categories.items():
                if skills:
                    parts.append('<div class="skills-section">')
                    parts.append(f'<span class="skills-category">{html.escape(category)}:</span>')
                    parts.append(html.escape(", ".join(skills)))
                    parts.append('</div>')

        return "\n".join(parts)

    def generate_pdf(self, output_path: str = None) -> str:
        """Exports HTML → PDF via WeasyPrint.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the PDF file. If not provided, it is generated automatically
            based on the user's name.
            
        Returns
        -------
        str
            Full path to the generated PDF file.

        Notes
        -----
        Install the library::

            pip install weasyprint

        WeasyPrint requires system deps (Cairo/Pango). On macOS::

            brew install cairo pango gdk-pixbuf libffi
        """

        try:
            from pathlib import Path
            from weasyprint import HTML
            import re
        except ImportError:
            raise ImportError(
                "WeasyPrint is not installed. Install it with:\n"
                "pip install weasyprint\n\n"
                "On macOS you will also need system dependencies:\n"
                "brew install cairo pango gdk-pixbuf libffi"
            )

        # If no path is provided, generate one based on the user's name
        if output_path is None:
            # Prepare the user's name for use in the file name
            if self.personal_info.name:
                # Replace spaces with underscores and strip other invalid characters
                name_part = re.sub(r'[^\w\s-]', '', self.personal_info.name).strip().lower().replace(' ', '_')
                output_path = f"data/cv_final/{name_part}_cv.pdf"
            else:
                output_path = "data/cv_final/cv_final.pdf"

        html_content = self.to_html()
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html_content).write_pdf(str(out))
        
        return str(out)


# ──────────────────────────────────────────────────────────────────────────────
#  DEMO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick PDF generation test (uses demo data)."""

    try:
        cv = CVTemplate(
            personal_info=PersonalInfo(
                name="Anna Shigirdanova",
                location="Tbilisi, Georgia",
                email="annaishig@gmail.com",
                links=[Link(title="LinkedIn", url="https://www.linkedin.com/in/anna-shigirdanova")],
            ),
            summary=Summary(
                text="""Frontend developer with 4 years of experience in large tech companies, including 1 year leading a team. 
                Experienced in optimizing deployment processes and building high‑performance large‑scale applications.
                Studied at the National Research University of Electronic Technology (MIREA)""",
                relevant_skills=["JavaScript", "React", "Vue.js", "Next.js"],
            ),
            skills=Skills(
                categories=[
                    Category(category="Languages", keywords=["JavaScript", "React", "Vue.js"]),
                    Category(category="Tools", keywords=["Webpack", "Storybook", "CI/CD"]),
                    Category(category="Soft skills", keywords=["Mentoring", "Team Leadership"]),
                ],
            ),
            experiences=[
                Experience(
                    role="Frontend Developer",
                    company="SAMOLET Group",
                    company_description="Largest Eastern Europe real estate developer",
                    start_date="2023-10",
                    end_date="2024-09",
                    location="Moscow",
                    bullets=[
                        BulletPoint(
                            action_verb=ActionVerb.DESIGNED,
                            description="and implemented a micro‑frontend architecture for SaaS",
                            skills=["Micro‑frontends"],
                            impact="‑10% deploy time",
                            benefit="Faster time‑to‑market",
                        ),
                        BulletPoint(
                            action_verb=ActionVerb.DEVELOPED,
                            description="a UI Kit library with Storybook",
                            skills=["Storybook"],
                            impact="‑30% UI dev time",
                        ),
                    ],
                ),
            ],
            education=[Education(
                institution="National Research University of Electronic Technology",
                degree="BSc in Computer Science",
                end_year=2022,
                gpa="5.0/5",
            )],
        )

        cv.validate()
        print("CV passed validation and HTML generated (len=%d)." % len(cv.to_html()))
        
        # Generate PDF
        pdf_path = cv.generate_pdf()
        print(f"PDF generated at {pdf_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
