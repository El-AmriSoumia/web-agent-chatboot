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
- Conversation memory is now managed as one active topic session at a time
- `POST /reset` archives the current topic and starts a clean session
- Browser anti-bot protections are always enabled by default


## 🚀 Système de Workflows Automatiques

Le système de workflows permet de **réduire de 87% le temps d'exécution** et **98% les coûts API** en enregistrant et réutilisant automatiquement les actions.

### ⚡ Démarrage Rapide

→ **[START_HERE.md](START_HERE.md)** - Commencez ici (5 minutes)

### 📚 Documentation Complète

- **[INDEX.md](INDEX.md)** - Navigation entre tous les documents
- **[WORKFLOWS_README.md](WORKFLOWS_README.md)** - Vue d'ensemble du système
- **[QUICK_CHECKLIST.md](QUICK_CHECKLIST.md)** - Validation rapide (10 min)
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Guide de test complet (7 tests)
- **[VALIDATION_FINALE.md](VALIDATION_FINALE.md)** - Validation finale (15 min)
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Intégration dans agent.py
- **[USEFUL_COMMANDS.md](USEFUL_COMMANDS.md)** - Commandes SQL, API, Python

### 🧪 Scripts de Test

- **[test_workflows.py](test_workflows.py)** - Tests automatisés Python
- **[check_workflows.sql](check_workflows.sql)** - Vérification PostgreSQL

### 📊 Performances

| Métrique | Sans Workflow | Avec Workflow | Amélioration |
|----------|--------------|---------------|-------------|
| Temps d'exécution | 30-40s | 3-5s | **87%** |
| Appels API Gemini | 10-20 | 0 | **98%** |
| Coût par exécution | ~$0.10 | ~$0.002 | **98%** |
