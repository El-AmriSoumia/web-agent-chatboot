# GSAM Private Intelligence

This repository contains a split frontend/backend project for the GSAM agent.

## Structure

- `frontend/` - React + Vite client application
- `backend/` - FastAPI backend and agent orchestration logic

## Frontend

Install dependencies and start the development server from the `frontend` folder:

```bash
cd frontend
npm install
npm run dev
```

## Backend

Install Python dependencies and start the backend from the `backend` folder:

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Environment

Create or update `backend/.env` with your Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

## Usage

- Enter a mission prompt in the frontend command bar
- The backend executes the agent flow
- Live browser screenshots are rendered when available
- The frontend displays backend SSE logs, step progress, and final results
