# Campus Assist — Final Version

AI Student Support Assistant built with **Python + FastAPI + Groq + Hybrid RAG + Tools + Memory**.

## What it does
- Answers general questions through Groq without unnecessarily searching the campus knowledge base.
- Answers college-specific questions from the local knowledge base using hybrid retrieval (TF-IDF + lexical/metadata/course-code boosts + confidence filtering).
- Uses tools for attendance, exam schedules, profile memory, and human-support tickets.
- Stores student profile and conversation history locally as JSON.
- Shows source records only when campus documents were actually retrieved.
- No separate Requests page; support tickets can still be created through the assistant when needed.
- Press **Enter** to submit a question. Use **Shift+Enter** for a new line.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set:

```env
GROQ_API_KEY=your_real_groq_key
GROQ_MODEL=openai/gpt-oss-120b
PORT=8000
```

Then run:

```bash
python run.py
```

Open `http://localhost:8000`.

## Architecture

```text
Student UI
   ↓
FastAPI
   ↓
Groq router/agent
   ├── General conversation → Groq
   ├── College question → Hybrid RAG
   └── Student service → Local tools
                         ├── Attendance
                         ├── Exam schedule
                         ├── Profile memory
                         └── Human-support ticket
```

## Important

The included attendance and exam data are deterministic demo data. Replace those tool implementations with your college SIS/ERP APIs for real deployment. The API key is loaded only by the Python backend; do not put it in frontend JavaScript.

## Security

Do not commit your real `.env` file. It is ignored by Git; commit `.env.example` only.
