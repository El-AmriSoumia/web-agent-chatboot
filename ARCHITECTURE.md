# 🏗️ Architecture Globale - Web Agent Chatbot

## 1. Architecture système complète

```mermaid
graph TB
  subgraph Client["🖥️ Couche Client"]
    FE["Frontend<br/>(React + Vite)"]
  end
  
  subgraph API["🔌 Couche API"]
    FastAPI["FastAPI<br/>(Python uvicorn)"]
  end
  
  subgraph AI["🤖 Couche IA & Intelligence"]
    Agent["Conversational Agent<br/>(Logic métier)"]
    Gemini["Google Gemini<br/>(LLM)"]
    RPA["Script Generator<br/>(RPA)"]
  end
  
  subgraph Automation["⚙️ Couche Automatisation"]
    Playwright["Playwright<br/>(Web Automation)"]
  end
  
  subgraph External["🌐 Ressources Externes"]
    Website["Websites<br/>(Targets)"]
  end
  
  subgraph Data["💾 Couche Données"]
    PostgreSQL["PostgreSQL<br/>(Persistence)"]
    Memory["Memory Cache<br/>(Session)"]
  end
  
  FE -->|REST API| FastAPI
  FastAPI -->|Routes| Agent
  Agent -->|Query| Gemini
  Agent -->|Generate Scripts| RPA
  Gemini -->|Response| Agent
  Agent -->|Execute| Playwright
  Playwright -->|Navigate & Interact| Website
  Website -->|Data| Playwright
  Agent -->|Read/Write| PostgreSQL
  Agent -->|Session| Memory
  
  style Client fill:#e6f3ff,stroke:#0066cc,stroke-width:2px
  style API fill:#fff4e6,stroke:#cc6600,stroke-width:2px
  style AI fill:#e6ffe6,stroke:#00aa00,stroke-width:2px
  style Automation fill:#ffe6f3,stroke:#cc0066,stroke-width:2px
  style External fill:#f0f0f0,stroke:#333,stroke-width:2px
  style Data fill:#ffeae6,stroke:#cc3300,stroke-width:2px
```

## 2. Flux de Données : User → Frontend → FastAPI → Agent → Gemini → Playwright → Website

```mermaid
graph LR
  User["👤 User"] -->|Requête texte| FE["Frontend<br/>(React UI)"]
  
  FE -->|HTTP POST| API["FastAPI<br/>(Endpoint)"]
  
  API -->|Dispatch| Agent["Agent<br/>(Conversational)"]
  
  Agent -->|Prompt| Gemini["Google Gemini<br/>(LLM API)"]
  
  Gemini -->|Response| Agent
  
  Agent -->|Script + Selector| PW["Playwright<br/>(Browser Automation)"]
  
  PW -->|Navigate & Interact| Website["Website<br/>(Target)"]
  
  Website -->|Page Data| PW
  
  PW -->|Extracted Data| Agent
  
  Agent -->|Store State| DB["PostgreSQL<br/>(Persistence)"]
  
  Agent -->|Response JSON| API
  
  API -->|JSON Response| FE
  
  FE -->|Render Result| User
  
  DB -.->|Query Logs| Agent
  
  style User fill:#fff4e6,stroke:#333,stroke-width:2px
  style FE fill:#e6f3ff,stroke:#0066cc,stroke-width:2px
  style API fill:#fff4e6,stroke:#cc6600,stroke-width:2px
  style Agent fill:#e6ffe6,stroke:#00aa00,stroke-width:2px
  style Gemini fill:#e6ffe6,stroke:#00aa00,stroke-width:2px
  style PW fill:#ffe6f3,stroke:#cc0066,stroke-width:2px
  style Website fill:#f0f0f0,stroke:#333,stroke-width:2px
  style DB fill:#ffeae6,stroke:#cc3300,stroke-width:2px
```

## 3. Composants clés

| Couche | Composant | Rôle | Tech |
|--------|-----------|------|------|
| **Client** | Frontend | Interface utilisateur, chat UI | React + Vite |
| **API** | FastAPI | Serveur API, routes REST | Python uvicorn |
| **IA** | Conversational Agent | Orchestration logique métier | Python |
| **IA** | Google Gemini | Génération texte/réponses LLM | API Google |
| **IA** | Script Generator | Génération scripts RPA | Python |
| **Automatisation** | Playwright | Navigation & interaction web | Node.js / Python |
| **Externe** | Websites | Sites cibles du scraping/automation | HTTP |
| **Données** | PostgreSQL | Stockage persistant | SQL |
| **Données** | Memory Cache | Session/état temporaire | In-Memory |

---

✅ **Architecture professionnelle et scalable**  
📊 **Respecte les bonnes pratiques d'API REST et de microservices**  
🔄 **Cycle complet : User → Requête → Traitement IA → Automatisation → Résultat**
