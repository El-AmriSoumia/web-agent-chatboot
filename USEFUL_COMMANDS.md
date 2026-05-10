# 🛠️ Commandes Utiles - Système de Workflows

## 📦 Installation et Démarrage

```bash
# Backend
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

## 🗄️ PostgreSQL - Commandes de Base

### Connexion
```bash
psql -U votre_user -d votre_database
```

### Créer les tables manuellement
```sql
-- Exécuter depuis Python
from backend.auth_db import create_tables
create_tables()
```

### Vérifier les tables
```sql
\dt
\d generated_workflows
\d workflow_actions
\d workflow_executions
```

## 📊 Requêtes SQL Utiles

### Lister tous les workflows
```sql
SELECT id, prompt_normalized, is_active, execution_count, created_at 
FROM generated_workflows 
ORDER BY created_at DESC;
```

### Voir les détails d'un workflow
```sql
SELECT * FROM generated_workflows WHERE id = 1;

SELECT action_type, selector, value, order_index 
FROM workflow_actions 
WHERE workflow_id = 1 
ORDER BY order_index;

SELECT success, execution_time_ms, error_message, created_at 
FROM workflow_executions 
WHERE workflow_id = 1 
ORDER BY created_at DESC;
```

### Statistiques de performance
```sql
SELECT 
    gw.id,
    gw.prompt_normalized,
    gw.execution_count,
    AVG(we.execution_time_ms) as avg_time_ms,
    MIN(we.execution_time_ms) as min_time_ms,
    MAX(we.execution_time_ms) as max_time_ms
FROM generated_workflows gw
LEFT JOIN workflow_executions we ON gw.id = we.workflow_id
GROUP BY gw.id, gw.prompt_normalized, gw.execution_count
ORDER BY gw.execution_count DESC;
```

### Workflows avec erreurs
```sql
SELECT 
    gw.id,
    gw.prompt_normalized,
    we.error_message,
    we.created_at
FROM workflow_executions we
JOIN generated_workflows gw ON we.workflow_id = gw.id
WHERE we.success = false
ORDER BY we.created_at DESC
LIMIT 10;
```

### Nettoyer les workflows inactifs
```sql
-- Désactiver tous les workflows
UPDATE generated_workflows SET is_active = false;

-- Supprimer un workflow spécifique
DELETE FROM workflow_executions WHERE workflow_id = 1;
DELETE FROM workflow_actions WHERE workflow_id = 1;
DELETE FROM generated_workflows WHERE id = 1;
```

### Statistiques des screenshots
```sql
SELECT 
    COUNT(*) as total_screenshots,
    COUNT(DISTINCT session_id) as unique_sessions
FROM messages 
WHERE message_type = 'screenshot';

-- Screenshots par session
SELECT 
    s.id,
    s.topic,
    COUNT(m.id) as screenshot_count
FROM sessions s
JOIN messages m ON s.id = m.session_id
WHERE m.message_type = 'screenshot'
GROUP BY s.id, s.topic
ORDER BY screenshot_count DESC;
```

## 🌐 API - Commandes cURL

### Workflows
```bash
# Lister tous les workflows
curl http://127.0.0.1:8000/workflows

# Récupérer un workflow spécifique
curl http://127.0.0.1:8000/workflows/1

# Désactiver un workflow
curl -X POST http://127.0.0.1:8000/workflows/1/deactivate

# Réactiver un workflow
curl -X POST http://127.0.0.1:8000/workflows/1/activate

# Supprimer un workflow
curl -X DELETE http://127.0.0.1:8000/workflows/1
```

### Screenshots
```bash
# Récupérer l'historique des screenshots d'une session
curl http://127.0.0.1:8000/sessions/SESSION_ID/screenshots
```

### Health Check
```bash
# Vérifier que le backend est accessible
curl http://127.0.0.1:8000/health
```

## 🐍 Python - Scripts Utiles

### Créer les tables
```python
from backend.auth_db import create_tables
create_tables()
```

### Tester la normalisation de prompt
```python
from backend.workflow_manager import normalize_prompt

prompt = "cherche canva"
normalized = normalize_prompt(prompt)
print(f"Normalisé: {normalized}")
```

### Trouver un workflow correspondant
```python
from backend.workflow_manager import find_matching_workflow

prompt = "cherche canva"
workflow = find_matching_workflow(prompt)
if workflow:
    print(f"Workflow trouvé: {workflow['id']}")
else:
    print("Aucun workflow trouvé")
```

### Extraire les paramètres d'un prompt
```python
from backend.workflow_manager import extract_parameters

prompt = "connecte-toi avec email test@example.com et password 123456"
params = extract_parameters(prompt)
print(f"Paramètres: {params}")
```

### Tester le générateur de script
```python
from backend.script_generator import PlaywrightScriptGenerator

gen = PlaywrightScriptGenerator()
gen.add_action("navigate", url="https://example.com")
gen.add_action("click", selector="#button")
gen.add_action("type", selector="#input", text="test")

script = gen.generate_script()
print(script)
```

### Tester l'agent conversationnel
```python
from backend.conversational_agent import is_conversational_question, answer_conversational_question

question = "c'est quoi mon dernier prompt"
if is_conversational_question(question):
    answer = answer_conversational_question(question, [])
    print(f"Réponse: {answer}")
```

## 🧪 Tests

### Exécuter les tests automatisés
```bash
python test_workflows.py
```

### Vérifier la base de données
```bash
psql -U votre_user -d votre_database -f check_workflows.sql
```

### Tester un script généré
```bash
python backend/generated_workflows/workflow_1.py
```

## 📝 Logs et Débogage

### Voir les logs backend en temps réel
```bash
tail -f backend/logs/app.log
```

### Voir les logs PostgreSQL
```bash
# Linux/Mac
tail -f /var/log/postgresql/postgresql-*.log

# Windows
# Vérifier dans C:\Program Files\PostgreSQL\XX\data\log\
```

### Activer le mode debug
```python
# Dans backend/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔧 Maintenance

### Sauvegarder la base de données
```bash
pg_dump -U votre_user votre_database > backup_$(date +%Y%m%d).sql
```

### Restaurer la base de données
```bash
psql -U votre_user votre_database < backup_20240101.sql
```

### Nettoyer les anciens workflows
```sql
-- Supprimer les workflows non utilisés depuis 30 jours
DELETE FROM workflow_executions 
WHERE workflow_id IN (
    SELECT id FROM generated_workflows 
    WHERE updated_at < NOW() - INTERVAL '30 days'
);

DELETE FROM workflow_actions 
WHERE workflow_id IN (
    SELECT id FROM generated_workflows 
    WHERE updated_at < NOW() - INTERVAL '30 days'
);

DELETE FROM generated_workflows 
WHERE updated_at < NOW() - INTERVAL '30 days';
```

### Réinitialiser les compteurs
```sql
UPDATE generated_workflows 
SET execution_count = 0, avg_execution_time_ms = 0;
```

## 📊 Monitoring

### Surveiller les performances
```sql
-- Workflows les plus lents
SELECT 
    id, 
    prompt_normalized, 
    avg_execution_time_ms 
FROM generated_workflows 
WHERE execution_count > 0 
ORDER BY avg_execution_time_ms DESC 
LIMIT 10;

-- Taux de succès
SELECT 
    gw.id,
    gw.prompt_normalized,
    COUNT(we.id) as total_runs,
    SUM(CASE WHEN we.success THEN 1 ELSE 0 END) as successful_runs,
    ROUND(100.0 * SUM(CASE WHEN we.success THEN 1 ELSE 0 END) / COUNT(we.id), 2) as success_rate
FROM generated_workflows gw
LEFT JOIN workflow_executions we ON gw.id = we.workflow_id
GROUP BY gw.id, gw.prompt_normalized
HAVING COUNT(we.id) > 0
ORDER BY success_rate ASC;
```

## 🚀 Optimisation

### Indexer les tables pour de meilleures performances
```sql
CREATE INDEX IF NOT EXISTS idx_workflows_prompt ON generated_workflows(prompt_normalized);
CREATE INDEX IF NOT EXISTS idx_workflows_active ON generated_workflows(is_active);
CREATE INDEX IF NOT EXISTS idx_actions_workflow ON workflow_actions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_executions_workflow ON workflow_executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_executions_created ON workflow_executions(created_at);
```

### Analyser les requêtes lentes
```sql
-- Activer le logging des requêtes lentes
ALTER DATABASE votre_database SET log_min_duration_statement = 1000;

-- Voir les statistiques des requêtes
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;
```

---

**💡 Astuce** : Ajoutez ces commandes à vos favoris pour un accès rapide !
