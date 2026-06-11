class Prompts:
    # CV Generator System Prompts
    CV_SYSTEM_PROMPT = """You are an expert CV/resume writer and career advisor. Your task is to
    help create a professional and compelling CV that highlights the candidate's strengths and
    matches the job requirements.
    Follow these key principles:
    1. Be specific and use concrete examples
    2. Focus on achievements and impact rather than just responsibilities
    3. Use strong action verbs
    4. Quantify results where possible
    5. Tailor content to the job requirements
    6. Keep information concise and relevant
    7. Maintain professional tone
    8. Ensure all information is accurate and verifiable"""

    # Personal Information Prompts
    PERSONAL_INFO_SYSTEM = """You are helping to create the personal information section of a CV.
    This section should be professional and include all necessary contact details."""

    PERSONAL_INFO_USER = """Please provide the following personal information for the CV:
    1. Full name (in uppercase)
    2. Current location (city, country)
    3. Professional email address
    4. Professional links (LinkedIn, portfolio, personal site, etc.)
    Additional context: {context}"""

    # Lowercase aliases realign the PERSONAL_INFO_SYSTEM/USER constants with the
    # `personal_info_system_prompt` / `personal_info_user_prompt` names that
    # cv_generator.py reads.
    personal_info_system_prompt = PERSONAL_INFO_SYSTEM
    personal_info_user_prompt = PERSONAL_INFO_USER

    # Summary Section Prompts
    SUMMARY_SYSTEM = """You are creating a compelling professional summary for a CV. The summary
    should be 3-5 lines long and highlight the candidate's key strengths and value proposition."""

    SUMMARY_USER = """Create a professional summary for a {job_title} position at {company_name}.
    Key requirements: {job_requirements}
    Candidate's experience: {experience}
    Key skills to highlight: {key_skills}
    The summary should be 3-5 lines and focus on the most relevant experience and skills."""

    # Skills Section Prompts
    SKILLS_SYSTEM = """You are categorizing and organizing professional skills for a CV. Focus on
    relevant skills that match the job requirements."""

    SKILLS_USER = """Based on the job requirements and candidate's experience, categorize the
    following skills:
    Job requirements: {job_requirements}
    Candidate's skills: {candidate_skills}

    Please organize skills into:
    1. Relevant skills (matching job requirements)
    2. Technical skills
    3. Soft skills
    Focus on the most relevant and impressive skills."""

    # Experience Section Prompts
    EXPERIENCE_SYSTEM = """You are creating detailed experience entries for a CV. Each entry should
    highlight achievements and impact using strong action verbs and quantifiable results."""

    EXPERIENCE_USER = """Create a detailed experience entry for the following position:
    Role: {role}
    Company: {company}
    Duration: {duration}
    Key responsibilities: {responsibilities}
    Achievements: {achievements}

    Please include:
    1. Company description (highlighting its significance)
    2. 3-4 bullet points with achievements
    3. Quantifiable results where possible
    4. Relevant skills used
    Format each bullet point as: [Action Verb] + [What] + [Impact] + [Benefit]"""

    # Education Section Prompts
    EDUCATION_SYSTEM = """You are creating the education section of a CV. Focus on relevant
    academic achievements and qualifications."""

    EDUCATION_USER = """Please provide education details for the CV:
    Institution: {institution}
    Degree: {degree}
    Years: {years}
    GPA: {gpa}
    Relevant coursework: {coursework}
    Academic achievements: {achievements}

    Format the information to highlight the most impressive and relevant aspects of the
    education."""

    # Project Section Prompts
    PROJECT_SYSTEM = """You are creating project entries for a CV. Focus on significant projects
    that demonstrate relevant skills and achievements."""

    PROJECT_USER = """Create a project entry for the CV:
    Project name: {name}
    Description: {description}
    Tools, methods, or technologies used: {technologies}
    Your role: {role}
    Key achievements: {achievements}
    Project link: {link}

    Focus on projects that demonstrate skills relevant to the target position."""

    # Language Skills Prompts
    LANGUAGE_SYSTEM = """You are documenting language proficiency for a CV. Use standardized
    language levels and focus on languages relevant to the position."""

    LANGUAGE_USER = """Please provide language proficiency information:
    Languages: {languages}
    Proficiency levels: {levels}
    Certifications: {certifications}

    Use standardized levels (A1-C2, Native, Fluent, etc.) and focus on languages relevant to the
    position."""

    # Certificate Section Prompts
    CERTIFICATE_SYSTEM = """You are documenting professional certifications for a CV. Focus on
    relevant and recent certifications."""

    CERTIFICATE_USER = """Please provide certification details:
    Certificates: {certificates}
    Issuing organizations: {issuers}
    Years obtained: {years}
    Links to verify: {links}

    Include only relevant and recent certifications that add value to the candidate's profile."""

    # --- Extract-pass prompts -------------------------------------------------
    # These two pairs drive the function-calling extract pass. Both are
    # truth-preserving: the model extracts only what is present and never fabricates
    # facts or keywords. The rules demand COMPLETENESS (empty arrays when entries
    # exist = failure), forbid invented placeholder strings, and disambiguate header
    # links vs project entries.
    # If extraction still under-extracts, the configured model (see
    # schemas.DEFAULT_MODEL_NAME) is likely the bottleneck on function-calling
    # volume — switch MODEL_NAME (e.g. gpt-4o-mini / Haiku); these rules are
    # necessary regardless of model.

    EXTRACT_CV_FACTS_SYSTEM = (
        "You extract truthful, structured facts from a candidate's CV text into the provided "
        "function schema. You are a high-recall extractor: your job is to capture EVERYTHING that "
        "is present, accurately, while inventing NOTHING.\n\nCOMPLETENESS (critical):\n1. Extract "
        "EVERY work-experience entry present in the CV — a CV typically lists multiple roles "
        "across different companies and dates. Returning an empty `experiences` array (or "
        "omitting roles) when roles are present in the text is a FAILURE, not caution.\n2. "
        "Extract EVERY education entry, certificate, project, and language that appears in the "
        "text. Empty arrays are correct ONLY when that information is genuinely absent.\n3. "
        "Extract EVERY skill group from the candidate's Skills section into `skills`: copy the "
        "candidate's OWN category label (or null if the list has no header) and every item string "
        "VERBATIM — do not reword, merge groups, or drop items. Spoken/natural languages (e.g. a "
        "\"Languages\" block listing tongues the person speaks) belong in `languages`, NOT in "
        "`skills`.\n4. For each experience entry, copy ALL of its bullet points — do not "
        "summarize, drop, or keep only the first bullet.\n5. Work through the CV top to bottom "
        "and account for every section it contains before calling the function.\n\nTRUTHFULNESS "
        "(non-negotiable):\n6. Extract ONLY what is explicitly present. Never invent roles, "
        "companies, dates, skills, metrics, titles, or links.\n7. Copy experience bullet points "
        "verbatim from the source text.\n8. Preserve company names, role titles, and dates "
        "exactly as written.\n\nFIELD DISCIPLINE (avoid placeholders and mis-bucketing):\n9. For "
        "a missing optional field, OMIT it or set it to null — NEVER write a placeholder string "
        "such as \"not provided\", \"N/A\", or \"unknown\".\n10. `personal_info.location` is the "
        "candidate's OWN location from the CV header. If the header has no location, leave it "
        "null — do NOT copy a job's location line (e.g. \"Remote\" next to an employer) into "
        "it.\n11. Profile/header links (LinkedIn, portfolio, personal site, or any field-specific "
        "profile) go in `personal_info.links` — NEVER in `projects`. The `projects` array is ONLY "
        "for described project entries (a project with a name and description), not for profile "
        "URLs. A bare link is a link, not a project.\n\n12. Call the extract_cv_facts function "
        "with the complete structured result."
    )

    EXTRACT_CV_FACTS_USER = (
        "Extract the structured facts from the following CV text. Capture every role, every "
        "bullet, every education entry, and every other section present — omit nothing that is in "
        "the text, and invent nothing that is not.\n\nCV text:\n{cv_text}"
    )

    ANALYZE_JD_SYSTEM = (
        "You analyze a job description into requirements, keywords, and a keyword-to-section plan "
        "using the provided function schema. Capture everything the posting emphasizes, "
        "accurately, while inventing nothing.\n\n1. Extract the role title, company, must-have "
        "and nice-to-have requirements, and the keywords the posting emphasizes.\n2. Be thorough "
        "on keywords: capture the concrete skills, tools, technologies, and domain terms the "
        "posting names — whatever the field (e.g. named software, instruments, certifications, "
        "methods, standards). Prefer the exact term the posting uses. Do NOT emit a bare umbrella "
        "token (a single generic word with no qualifier — e.g. \"AI\", \"Cloud\", \"Data\", "
        "\"Care\", \"Compliance\", \"Management\") on its own — it is not a discrete, matchable "
        "skill; keep only the specific/qualified form the posting names (e.g. \"AI integration "
        "into SDLC\", \"Wound Care\", \"Regulatory Compliance\"), and drop the bare umbrella.\n3. "
        "Map each keyword to the CV section where it best belongs (contact, summary, skills, "
        "experience, education, projects).\n4. Tag each keyword's evidence tier in keyword_tiers: "
        "\"concrete\" for a specific NAMED thing — a named tool, software, system, "
        "certification/license, instrument, standard, or technology (e.g. Epic EHR, Westlaw, AWS, "
        "QuickBooks, ACLS, CNC — only credible if it appears literally in a CV); \"competency\" "
        "for a method, practice, or capability (e.g. triage, case strategy, systems thinking, "
        "curriculum design — can be evidenced by a related accomplishment). Judge each keyword on "
        "its own meaning, in this candidate's field; do not rely on a fixed list.\n5. Infer the "
        "candidate level (new_grad, entry, mid, senior_ic, manager, director) from the "
        "posting.\n6. Use only terms present in or directly implied by the job description. Never "
        "fabricate keywords to reach a count.\n7. Call the analyze_job_description function with "
        "the structured result."
    )

    ANALYZE_JD_USER = (
        "Analyze the following job description. Capture every must-have and nice-to-have "
        "requirement and every concrete keyword the posting emphasizes:\n\n{jd_text}"
    )

    # --- Section generation prompts -------------------------------------------
    # Section-wise tailoring prompt pairs. For each pair the SYSTEM prompt is the
    # invariant color methodology + truth-preserving rules; the USER prompt carries
    # the per-request facts / JD analysis / keyword→section plan. The model is forced
    # to call the matching generate_* tool (tool_schemas.py). Truth-preserving:
    # reframe existing facts only — never invent companies, dates, metrics, skills,
    # projects, certificates, or links. `personal_info` has no prompt: it is carried
    # from CVFacts, not generated.

    _COLOR_METHODOLOGY = (
        "Color methodology (map 1:1 onto the output fields; never invent a parallel model):\n- 🟡 "
        "summary keyword surface / 🟡🟠 skills: surface the JD keywords assigned to this section, "
        "using the posting's own wording so they match.\n- 🔴 action_verb: a strong action verb "
        "opening each experience bullet.\n- 🟠 skills: the concrete skills used in a bullet.\n- 🟢 "
        "impact: a quantified result (contains a digit) when truthfully available.\n- 🔵 benefit: "
        "the business benefit when truthfully available.\n- 🟣 company_description: a short "
        "truthful pitch of the company.\nTruth-preserving: reframe and re-emphasize existing "
        "facts only. Never invent companies, roles,\ndates, metrics, skills, projects, "
        "certificates, or links. Process EVERY entry present in the\ncandidate facts — do not "
        "drop or shorten the list."
    )

    _SECTION_USER = """Candidate facts (truthful source of record):
{facts}

Job-description analysis:
{jd}

Keyword→section plan:
{keyword_plan}

{instruction}"""

    GENERATE_SUMMARY_SYSTEM = (
        "You write the CV summary section from the candidate's real facts and the job-description "
        "keyword plan, then call generate_summary.\n\n" + _COLOR_METHODOLOGY
    )
    GENERATE_SUMMARY_USER = _SECTION_USER.format(
        facts="{facts}",
        jd="{jd}",
        keyword_plan="{keyword_plan}",
        instruction=(
            "Write a professional summary of 3 to 5 full sentences (this is a hard requirement: "
            "not 1, not 2 — at least 3 sentences and at most 5), grounded in the candidate's real "
            "experience. Weave the JD keywords mapped to the summary section (🟡) directly into "
            "the prose sentences of `summary.text` as natural language — they must appear in the "
            "sentences themselves (this is the only text ATS scores), using the posting's own "
            "wording so they match. Include ONLY keywords the candidate's facts genuinely support; "
            "never fabricate to inflate coverage. Call generate_summary."
        ),
    )

    GENERATE_SKILLS_SYSTEM = (
        "You write the CV skills section from the candidate's real facts and the job-description "
        "keyword plan, then call generate_skills.\n\n" + _COLOR_METHODOLOGY
    )
    GENERATE_SKILLS_USER = _SECTION_USER.format(
        facts="{facts}",
        jd="{jd}",
        keyword_plan="{keyword_plan}",
        instruction=(
            "TAILOR the candidate's REAL skills to this posting — do not synthesize a skills "
            "section from scratch. START from the candidate's own declared skills in "
            "`facts.skills` (their Skills section): include ALL of them — they are facts and "
            "need no further evidence. Then REFRAME their wording toward the posting's vocabulary "
            "ONLY "
            "when it is an honest equivalent (reframing changes wording, never the underlying "
            "claim), FOREGROUND the skills the JD names (list them first), and SEED your category "
            "headers from the candidate's own group labels, reordered/reworded toward the JD. You "
            "may ADD a JD keyword the candidate has NOT declared ONLY under the tier rules below; "
            "never drop a real skill merely because the JD does not name it (🟡🟠). Give EACH "
            "keyword a short, "
            "conventional `category` header you DERIVE from THIS candidate's own field and the "
            "posting's wording — there is no fixed list, and you must NOT force a software-shaped "
            "set of headers onto a non-software CV. Examples of the KIND of headers different "
            "fields use (illustrative only — derive your own from the material):\n"
            "- a software CV might use 'Languages', 'Frameworks', 'Tools';\n"
            "- a nursing CV 'Clinical Skills', 'Certifications', 'Patient Care';\n"
            "- a finance CV 'Analysis', 'Regulatory', 'Software';\n"
            "- a teaching CV 'Instruction', 'Assessment', 'Subjects'.\n"
            "Each header is a short noun label (at most a few words), one concept per header, "
            "plain text with no punctuation tricks or markdown. Use the SAME header for keywords "
            "that belong to the same group so they aggregate under one line.\n"
            "Emit each keyword as an object with its tier (from the JD's keyword_tiers). The tier "
            "rules gate ONLY the ADDITION of a JD keyword the candidate has not declared (the "
            "candidate's own declared skills are facts and are always kept): to ADD a 'concrete' "
            "keyword (a specific named tool, software, system, certification, instrument, "
            "standard, or technology) it must appear literally in the facts — never infer "
            "a named thing the candidate did not state; to ADD a 'competency' keyword (a method, "
            "practice, "
            "or capability) you must also supply an anchor_ref: a phrase or bullet COPIED VERBATIM "
            "from the candidate's facts that demonstrates it (do not paraphrase the anchor). Use "
            "the posting's exact wording for JD-named skills. Drop bare umbrella tokens — "
            "keep only the specific/qualified form. Never invent a skill the facts do not support. "
            "Call "
            "generate_skills."
        ),
    )

    GENERATE_EXPERIENCE_SYSTEM = (
        "You rewrite the CV experience section from the candidate's real facts and the "
        "job-description keyword plan, then call generate_experience.\n\n" + _COLOR_METHODOLOGY
    )
    GENERATE_EXPERIENCE_USER = _SECTION_USER.format(
        facts="{facts}",
        jd="{jd}",
        keyword_plan="{keyword_plan}",
        instruction=(
            "Rewrite EVERY role in the candidate facts — do not drop any. Company names and dates "
            "MUST match the source facts exactly. Each bullet uses 🔴 action_verb + description, "
            "🟠 skills, and 🟢 impact / 🔵 benefit when truthful. Where a bullet genuinely "
            "involves a JD keyword, phrase it using the posting's wording so ATS matching "
            "succeeds — never invent involvement that isn't in the facts. Call "
            "generate_experience."
        ),
    )

    GENERATE_EDUCATION_SYSTEM = (
        "You write the CV education section from the candidate's real facts, then call "
        "generate_education.\n\n" + _COLOR_METHODOLOGY
    )
    GENERATE_EDUCATION_USER = _SECTION_USER.format(
        facts="{facts}",
        jd="{jd}",
        keyword_plan="{keyword_plan}",
        instruction=(
            "Pass through EVERY truthful education entry (institution, degree, years, GPA) present "
            "in the facts — do not drop any. Never invent. Call generate_education."
        ),
    )

    GENERATE_PROJECT_SYSTEM = (
        "You write the CV projects section from the candidate's real facts, then call "
        "generate_project.\n\n" + _COLOR_METHODOLOGY
    )
    GENERATE_PROJECT_USER = _SECTION_USER.format(
        facts="{facts}",
        jd="{jd}",
        keyword_plan="{keyword_plan}",
        instruction=(
            "Include ONLY projects present in the source facts; never invent a project, and never "
            "treat a bare profile link as a project. Emphasize JD-relevant skills truthfully. Call "
            "generate_project."
        ),
    )

    GENERATE_CERTIFICATE_SYSTEM = (
        "You write the CV certificates section from the candidate's real facts, then call "
        "generate_certificate.\n\n" + _COLOR_METHODOLOGY
    )
    GENERATE_CERTIFICATE_USER = _SECTION_USER.format(
        facts="{facts}",
        jd="{jd}",
        keyword_plan="{keyword_plan}",
        instruction=(
            "Include ONLY certificates present in the source facts; never invent. Call "
            "generate_certificate."
        ),
    )

    GENERATE_LANGUAGE_SYSTEM = (
        "You write the CV languages section from the candidate's real facts, then call "
        "generate_language.\n\n" + _COLOR_METHODOLOGY
    )
    GENERATE_LANGUAGE_USER = _SECTION_USER.format(
        facts="{facts}",
        jd="{jd}",
        keyword_plan="{keyword_plan}",
        instruction=(
            "Include ONLY languages present in the source facts, with a standardized level; never "
            "invent. Call generate_language."
        ),
    )

    # --- Cover-letter generation prompts --------------------------------------
    # The SYSTEM prompt combines the point-by-point methodology, the truth-preserving
    # rule (claims only from CVFacts), and the anti-pattern ban-list. The USER prompt
    # carries the shared extract's facts + jd as JSON so the JD is consumed as-is and
    # never re-parsed.
    GENERATE_COVER_LETTER_SYSTEM = """\
You write a truthful, well-structured cover letter from the candidate's real CV facts and the
job-description analysis, then call generate_cover_letter.

Produce a COMPLETE letter with this structure, paragraphs separated by blank lines:
1. A salutation line. Address the company / role from the JD when present (jd.company,
   jd.role_title); if the JD names no company, use "Dear Hiring Manager," — never invent a
   company name.
2. An opening paragraph naming the role applied for and a one-line value proposition grounded in
   real CV facts.
3. Themed body paragraphs, point by point: each paragraph names one requirement area from the JD
   (jd.requirements_must) and gives concrete evidence from the candidate's real facts — the
   specific role, company, and result or metric where truthful — in flowing prose, no bullets, in
   the order the requirements are listed.
4. A closing paragraph that is forward-looking and tied to the role/company.
5. A sign-off line (e.g. "Sincerely,") followed by the candidate's name from the facts.

Truthfulness (non-negotiable): every claim must be drawn only from the candidate's CV facts.
Structure ORGANIZES real evidence — never invent a company, an employer, a metric, a title, a
project, a skill, or a paragraph's evidence to satisfy the skeleton. Reframe and re-emphasize what
already exists. If a posted requirement cannot be backed by real CV evidence, omit it — a
candidate with thin evidence simply has fewer body paragraphs; never fabricate to fill one. Where
the candidate's real experience genuinely satisfies a requirement, say so concretely with the
specific role, company, or result.

Anti-pattern ban-list — do NOT write any of these:
- fan letters or flattery of the company ("I have always admired", "dream company");
- drama or emotional appeals ("I would be devastated", "my lifelong passion");
- generic, unbacked adjectives ("hardworking", "detail-oriented", "team player", "passionate",
  "results-driven") — replace each with the concrete CV evidence that demonstrates it.
Keep it professional, specific, and evidence-led."""

    GENERATE_COVER_LETTER_USER = """\
Candidate facts (truthful source of record):
{facts}

Job-description analysis (reuse this as-is; do NOT re-derive or re-parse the posting):
{jd}

Write the complete cover letter — a salutation, an opening paragraph, themed body paragraphs each
matching one jd.requirements_must area to concrete CV evidence in prose, a closing paragraph, and a
sign-off with the candidate's name — then call generate_cover_letter. Omit any requirement you
cannot back with real evidence; never invent evidence to fill a gap."""
