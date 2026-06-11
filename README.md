# FitCV — ATS Resume Tailor

FitCV tailors an existing CV to a specific job description so it passes ATS
keyword/parse screening. You paste a job description and upload your CV; you get
back an optimized, ATS-parseable CV reframed for that posting, a measurable
keyword-coverage score (before → after), and a matching cover letter.

The product is **truth-preserving**: it reframes and re-emphasizes what your CV
already contains and never invents roles, skills, metrics, or projects.

---

## How it works

The system is a stateless two-process app — a **FastAPI backend** and a
**Streamlit frontend** — with no accounts, no database, and no persistence. One
CV + one JD per request.

```
Streamlit UI (app.py)
      │  multipart/form-data
      ▼
FastAPI  POST /generate  (main.py)
      │
      ├── build_extract        parse CV (PDF/DOCX/TXT) + analyze JD → ExtractResult{facts, jd, flags}
      ├── generate_tailored_cv section-wise tailoring → TailoredResult{cv, ats_score, flags}
      └── generate_cover_letter point-by-point cover letter → CoverLetterResult{cover_letter, flags}
      │
      ▼
   JSON response (tailored CV + ATS before→after + cover letter + honest-gap flags)
```

### Pipeline stages

1. **Parse & extract** (`extract.py`) — the uploaded CV is parsed into structured
   facts and the job description into requirements + keywords + a keyword→section
   plan. Both produce one shared `ExtractResult`.
2. **Tailor** (`cv_generator.py`) — each CV section is generated against the shared
   plan via model function calling, validated deterministically (writing + ATS +
   factual-integrity checks), and only failing sections are regenerated (capped).
   Sections are assembled under a one-page global gate (dedup + compression).
3. **Score** — keyword coverage is computed before vs. after as the ATS score.
4. **Cover letter** (`cover_letter.py`) — reuses the same extract to write a
   truthful requirement → evidence cover letter (≤ ~300 words).
5. **Cleanup** (`helprers/text_preprocessing.py`) — a deterministic char-level +
   AI-tell cleanup pass runs before keyword matching.

Honest gaps (missing keywords/requirements) are surfaced as **flags** in the
response, never hidden and never treated as fatal errors.

### Tech stack

- **Backend:** FastAPI + Uvicorn + Pydantic
- **Frontend:** Streamlit
- **AI:** OpenRouter via the OpenAI-compatible client (default model
  `google/gemini-3.5-flash`, with native function calling)
- **CV parsing:** PyMuPDF (PDF) + python-docx (DOCX) + plain text

The API key lives **server-side only** — the Streamlit client only talks HTTP to
the backend and never sees it.

---

## Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key

---

## Setup

```bash
# 1. Clone and enter the project
cd FitCV

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
# For running tests/lint, install dev deps instead:
# pip install -r requirements-dev.txt
```

### Configure environment

Copy the example env file and fill in your key:

```bash
cp .env.example .env
```

Then set the values in `.env`:

| Variable               | Required | Default                          | Description                                              |
| ---------------------- | -------- | -------------------------------- | -------------------------------------------------------- |
| `OPENROUTER_API_KEY`   | **yes**  | —                                | Your OpenRouter API key (server-side only).              |
| `MODEL_NAME`           | no       | `google/gemini-3.5-flash`          | Model id used for all generation steps.                  |
| `OPENROUTER_BASE_URL`  | no       | `https://openrouter.ai/api/v1`   | OpenAI-compatible base URL.                              |
| `API_BASE_URL`         | no       | `http://localhost:8000`          | Where the Streamlit UI points to reach the backend.      |

> The `.env` file holds your secret key — it is gitignored and must never be committed.

---

## Running the app

The backend and frontend are two separate processes. Start the backend first,
then the UI.

### 1. Start the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

- API: `http://localhost:8000`
- Liveness probe: `GET http://localhost:8000/health`
- Interactive docs: `http://localhost:8000/docs`

### 2. Start the Streamlit frontend

In a second terminal (with the same virtualenv activated):

```bash
streamlit run app.py
```

Streamlit opens at `http://localhost:8501`.

### 3. Use it

1. Upload your CV (PDF, DOCX, or TXT).
2. Paste the target job description.
3. Click **Generate**.
4. Read the tailored CV, the ATS coverage score (before → after), the cover
   letter, and any honest-gap flags. Output is show/copy only — there is no
   download or in-UI editing in the MVP (re-run to iterate).

---

## Calling the API directly

The backend can be used without the UI:

```bash
curl -X POST http://localhost:8000/generate \
  -F "cv=@/path/to/your_cv.pdf" \
  -F "job_description=Paste the full job description text here"
```

A successful request returns `200` with the structured tailored CV, ATS score,
cover letter, and flags. Failures return an `{error, stage}` envelope (stages:
`parse`, `extract`, `generate`, `validate`, `assemble`) and never leak the API
key or raw provider error.

Accepted upload formats: **PDF, DOCX, TXT** (max **10 MB** per upload).

---

## Development

Run the verification gate before considering a change done:

```bash
pip install -r requirements-dev.txt

pytest        # tests
ruff check .  # lint
```

---