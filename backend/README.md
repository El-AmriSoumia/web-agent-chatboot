# GSAM Backend

FastAPI backend for GSAM agent orchestration.

## Setup

1. Create a `.env` file in `backend/` with:

```env
GEMINI_API_KEY=your_gemini_api_key
```

2. Install dependencies:

```bash
pip install -r backend/requirements.txt
playwright install chromium
```

## Run
From the ROOT of the project (not inside backend/):
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

## Test the backend is working
    curl http://localhost:8000/health
    → {"status": "ok"}

## Available endpoints
- GET  /health   — check if backend is alive
- POST /run      — start agent, body: {"task": "your task here"}
- POST /abort    — stop the running agent
- POST /confirm  — confirm a safety-required action when the agent requests operator approval

## Runtime behavior
- The `/run` endpoint streams progress events over Server-Sent Events (SSE).
- If the client disconnects, the backend closes the SSE stream and clears the current agent state.
- When the agent believes an action is risky, it sends a safety event and waits up to 30 seconds for `/confirm`.

## Common errors
- "GEMINI_API_KEY is not set" → create backend/.env with your key
- "playwright._impl._errors.Error: Executable doesn't exist"
  → run: playwright install chromium
- "409 Agent is already running"
  → call POST /abort first, then retry

## Notes

- The backend uses Gemini Flash (`gemini-1.5-flash`) and Playwright.
- The `back/code.html` page falls back to the local mock loop if the backend is unavailable.
