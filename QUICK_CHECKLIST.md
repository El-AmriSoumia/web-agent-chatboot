# ✅ Checklist de Validation Rapide

## 🚀 Démarrage

```bash
# 1. Démarrer le backend
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# 2. Démarrer le frontend (nouveau terminal)
cd frontend
npm run dev

# 3. Vérifier PostgreSQL
psql -U votre_user -d votre_database -c "SELECT version();"
```

## 📋 Tests Manuels (5 minutes)

### ✅ Test 1 : Création d'un workflow (2 min)
1. Ouvrez http://localhost:5173
2. Entrez : `"cherche canva"`
3. Attendez la fin de l'exécution
4. **Vérification** : Un workflow doit être créé

```sql
-- Vérifier en base
SELECT id, prompt_normalized, is_active FROM generated_workflows ORDER BY created_at DESC LIMIT 1;
```

### ✅ Test 2 : Réutilisation du workflow (1 min)
1. Entrez à nouveau : `"cherche canva"`
2. **Vérification** : L'exécution doit être beaucoup plus rapide (3-5s)
3. Observez les logs backend : vous devriez voir "Workflow found"

### ✅ Test 3 : Historique des screenshots (1 min)
1. Cliquez sur le bouton déroulant dans la sidebar
2. **Vérification** : Tous les screenshots doivent s'afficher avec timestamp et URL

### ✅ Test 4 : Question conversationnelle (30 sec)
1. Entrez : `"c'est quoi mon dernier prompt"`
2. **Vérification** : Réponse instantanée sans navigation web

### ✅ Test 5 : API Workflows (30 sec)
```bash
# Lister les workflows
curl http://127.0.0.1:8000/workflows

# Récupérer un workflow spécifique
curl http://127.0.0.1:8000/workflows/1
```

## 🧪 Tests Automatisés

```bash
# Exécuter le script de test
python test_workflows.py

# Vérifier la base de données
psql -U votre_user -d votre_database -f check_workflows.sql
```

## 📊 Métriques Attendues

| Métrique | Valeur attendue |
|----------|----------------|
| Temps 1ère exécution | 30-40s |
| Temps avec workflow | 3-5s |
| Amélioration | 87% |
| Économie API | 98% |
| Appels Gemini (1ère fois) | 10-20 |
| Appels Gemini (workflow) | 0 |

## 🐛 Problèmes Courants

### ❌ "Table does not exist"
```python
# Solution
from backend.auth_db import create_tables
create_tables()
```

### ❌ "Workflow not found"
```sql
-- Vérifier la normalisation
SELECT prompt_normalized FROM generated_workflows;
```

### ❌ "Script execution failed"
```bash
# Vérifier le script
python backend/generated_workflows/workflow_1.py
```

### ❌ "Screenshots not loading"
```bash
# Vérifier l'endpoint
curl http://127.0.0.1:8000/sessions/SESSION_ID/screenshots
```

## ✅ Validation Finale

- [ ] Backend démarre sans erreur
- [ ] Frontend accessible sur http://localhost:5173
- [ ] PostgreSQL connecté
- [ ] Workflow créé lors de la 1ère exécution
- [ ] Workflow réutilisé lors de la 2ème exécution
- [ ] Temps d'exécution réduit de 87%
- [ ] Screenshots affichés dans l'historique
- [ ] Questions conversationnelles fonctionnent
- [ ] API /workflows accessible
- [ ] Script de test passe tous les tests

## 🎯 Résultat Attendu

Si tous les tests passent :
- ✅ Le système de workflows est opérationnel
- ✅ Les performances sont optimisées
- ✅ Les coûts API sont réduits
- ✅ L'historique est fonctionnel

## 📞 Prochaines Étapes

1. **Production** : Déployer le système
2. **Monitoring** : Surveiller les métriques
3. **Optimisation** : Améliorer les workflows existants
4. **Documentation** : Créer des workflows pour les tâches courantes

---

**Temps total de validation : ~10 minutes**
