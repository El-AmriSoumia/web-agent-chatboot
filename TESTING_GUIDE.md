# Guide de Test - Système de Workflows Automatiques

## 📋 Vue d'ensemble

Ce guide vous permet de tester et valider le système de génération et réutilisation de workflows automatiques.

## ⚙️ Prérequis

1. **Base de données PostgreSQL** configurée et accessible
2. **Backend** démarré : `uvicorn main:app --reload --host 127.0.0.1 --port 8000`
3. **Frontend** démarré : `npm run dev`
4. **Variables d'environnement** configurées dans `backend/.env`

## 🧪 Tests à effectuer

### Test 1 : Vérification des tables PostgreSQL

**Objectif** : S'assurer que les tables de workflows sont créées

```bash
# Connectez-vous à PostgreSQL
psql -U votre_user -d votre_database

# Vérifiez les tables
\dt

# Vous devriez voir :
# - generated_workflows
# - workflow_actions
# - workflow_executions
```

**Requêtes de vérification** :
```sql
-- Structure de la table generated_workflows
SELECT * FROM generated_workflows LIMIT 1;

-- Structure de la table workflow_actions
SELECT * FROM workflow_actions LIMIT 1;

-- Structure de la table workflow_executions
SELECT * FROM workflow_executions LIMIT 1;
```

**✅ Résultat attendu** : Les 3 tables existent avec les colonnes appropriées

---

### Test 2 : Premier enregistrement d'un workflow (Recording)

**Objectif** : Créer un workflow automatique lors de la première exécution

**Étapes** :
1. Ouvrez le frontend
2. Entrez un prompt simple et répétable : `"cherche canva"`
3. Laissez l'agent compléter la tâche
4. Observez les logs backend

**✅ Résultat attendu** :
- L'agent exécute la tâche normalement
- Un workflow est créé dans la base de données
- Le script Python est généré et sauvegardé

**Vérification en base** :
```sql
-- Vérifier que le workflow a été créé
SELECT id, prompt_normalized, script_path, is_active, created_at 
FROM generated_workflows 
ORDER BY created_at DESC 
LIMIT 1;

-- Vérifier les actions enregistrées
SELECT action_type, selector, value, order_index 
FROM workflow_actions 
WHERE workflow_id = (SELECT id FROM generated_workflows ORDER BY created_at DESC LIMIT 1)
ORDER BY order_index;
```

---

### Test 3 : Réutilisation du workflow (Replay)

**Objectif** : Vérifier que le workflow généré est réutilisé automatiquement

**Étapes** :
1. Entrez le **même prompt** : `"cherche canva"`
2. Observez le temps d'exécution (devrait être 87% plus rapide)
3. Vérifiez les logs backend pour voir "Workflow found"

**✅ Résultat attendu** :
- Le workflow existant est détecté
- L'exécution est beaucoup plus rapide (3-5s au lieu de 30-40s)
- Aucun appel à Gemini AI n'est effectué
- Le résultat est identique à la première exécution

**Vérification en base** :
```sql
-- Vérifier l'exécution du workflow
SELECT workflow_id, execution_time_ms, success, error_message, created_at
FROM workflow_executions
ORDER BY created_at DESC
LIMIT 1;
```

---

### Test 4 : Extraction de paramètres

**Objectif** : Tester la détection automatique des paramètres (email, password, phone)

**Étapes** :
1. Créez un workflow avec un formulaire : `"remplis le formulaire sur example.com"`
2. Fournissez des valeurs : email, mot de passe, téléphone
3. Vérifiez que le script généré contient des paramètres

**✅ Résultat attendu** :
- Le script Python contient des paramètres `email`, `password`, `phone`
- Les valeurs sont remplacées par des placeholders dans le script
- Le workflow peut être réutilisé avec de nouvelles valeurs

**Vérification du script** :
```bash
# Lisez le script généré
cat backend/generated_workflows/workflow_<id>.py

# Vous devriez voir :
# def execute_workflow(email=None, password=None, phone=None):
```

---

### Test 5 : Gestion des workflows via API

**Objectif** : Tester les endpoints de gestion des workflows

**5.1 - Lister tous les workflows**
```bash
curl http://127.0.0.1:8000/workflows
```

**✅ Résultat attendu** : Liste JSON de tous les workflows

**5.2 - Récupérer un workflow spécifique**
```bash
curl http://127.0.0.1:8000/workflows/1
```

**✅ Résultat attendu** : Détails du workflow avec actions et exécutions

**5.3 - Désactiver un workflow**
```bash
curl -X POST http://127.0.0.1:8000/workflows/1/deactivate
```

**✅ Résultat attendu** : `{"status": "deactivated"}`

**5.4 - Réactiver un workflow**
```bash
curl -X POST http://127.0.0.1:8000/workflows/1/activate
```

**✅ Résultat attendu** : `{"status": "activated"}`

**5.5 - Supprimer un workflow**
```bash
curl -X DELETE http://127.0.0.1:8000/workflows/1
```

**✅ Résultat attendu** : `{"status": "deleted"}`

---

### Test 6 : Historique des screenshots

**Objectif** : Vérifier que tous les screenshots sont sauvegardés et affichés

**Étapes** :
1. Lancez une tâche qui génère plusieurs screenshots
2. Cliquez sur le bouton déroulant dans la sidebar
3. Vérifiez que tous les screenshots sont affichés avec timestamp et URL

**✅ Résultat attendu** :
- Le dernier screenshot est affiché par défaut
- Le bouton "Show Screenshot History" affiche tous les anciens screenshots
- Chaque screenshot a un timestamp et une URL

**Vérification en base** :
```sql
-- Vérifier les screenshots d'une session
SELECT id, content, url, created_at 
FROM messages 
WHERE session_id = 'votre_session_id' 
  AND message_type = 'screenshot'
ORDER BY created_at DESC;
```

---

### Test 7 : Questions conversationnelles

**Objectif** : Vérifier que l'agent répond aux questions sans lancer Playwright

**Étapes** :
1. Posez une question conversationnelle : `"c'est quoi mon dernier prompt"`
2. Vérifiez que la réponse est immédiate (sans navigation web)

**✅ Résultat attendu** :
- Réponse instantanée basée sur l'historique
- Aucune navigation web n'est lancée
- La réponse est pertinente et basée sur la mémoire

---

## 🐛 Débogage

### Problème : Les tables ne sont pas créées

**Solution** :
```python
# Exécutez manuellement dans un shell Python
from backend.auth_db import create_tables
create_tables()
```

### Problème : Le workflow n'est pas détecté

**Vérifications** :
1. Le prompt est-il normalisé correctement ?
   ```sql
   SELECT prompt_normalized FROM generated_workflows;
   ```
2. Le workflow est-il actif ?
   ```sql
   SELECT is_active FROM generated_workflows WHERE id = X;
   ```

### Problème : Le script Python n'est pas exécuté

**Vérifications** :
1. Le fichier existe-t-il ?
   ```bash
   ls backend/generated_workflows/
   ```
2. Le script est-il valide ?
   ```bash
   python backend/generated_workflows/workflow_X.py
   ```

### Problème : Les screenshots ne s'affichent pas

**Vérifications** :
1. L'endpoint retourne-t-il les données ?
   ```bash
   curl http://127.0.0.1:8000/sessions/SESSION_ID/screenshots
   ```
2. Le frontend charge-t-il les screenshots ?
   - Ouvrez la console du navigateur
   - Vérifiez les erreurs réseau

---

## 📊 Métriques de performance

### Temps d'exécution attendu

| Scénario | Première exécution | Avec workflow |
|----------|-------------------|---------------|
| Recherche simple | 30-40s | 3-5s |
| Formulaire | 45-60s | 5-8s |
| Navigation complexe | 60-90s | 8-12s |

### Économies de coûts

- **Sans workflow** : ~10-20 appels API Gemini par tâche
- **Avec workflow** : 0 appel API (98% d'économie)

---

## ✅ Checklist finale

- [ ] Les 3 tables PostgreSQL sont créées
- [ ] Un workflow peut être enregistré (Recording)
- [ ] Un workflow peut être réutilisé (Replay)
- [ ] Les paramètres sont détectés automatiquement
- [ ] Les endpoints API fonctionnent
- [ ] L'historique des screenshots est affiché
- [ ] Les questions conversationnelles fonctionnent
- [ ] Les performances sont améliorées (87% plus rapide)
- [ ] Les coûts API sont réduits (98% d'économie)

---

## 🚀 Prochaines étapes

Une fois tous les tests validés, vous pouvez :
1. Créer des workflows pour vos tâches répétitives
2. Partager les workflows avec votre équipe
3. Monitorer les performances via les métriques
4. Optimiser les workflows existants

---

## 📞 Support

Si vous rencontrez des problèmes :
1. Vérifiez les logs backend : `tail -f backend/logs/app.log`
2. Vérifiez les logs PostgreSQL
3. Consultez la documentation des modules :
   - `backend/script_generator.py`
   - `backend/workflow_manager.py`
   - `backend/conversational_agent.py`
