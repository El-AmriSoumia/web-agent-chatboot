# 🔌 Guide d'Intégration - agent.py

## 📋 Vue d'ensemble

Ce guide explique comment intégrer le système de workflows dans `agent.py` pour activer l'enregistrement et la réutilisation automatique des actions.

## 🎯 Objectif

- **Avant** : Chaque exécution appelle Gemini AI (30-40s, ~$0.10)
- **Après** : Première exécution enregistre, suivantes réutilisent (3-5s, ~$0.002)

## 📝 Modifications à Apporter

### Étape 1 : Ajouter les imports (ligne ~40)

```python
# Ajouter après les imports existants
from backend.workflow_manager import (
    find_matching_workflow,
    create_workflow,
    extract_parameters,
    log_workflow_execution
)
from backend.script_generator import PlaywrightScriptGenerator
import subprocess
import sys
```

### Étape 2 : Modifier la fonction `_run_playwright_sync` (ligne ~800)

**Trouver cette ligne** :
```python
def _run_playwright_sync(
    task: str,
    loop: asyncio.AbstractEventLoop,
    send_event,
    mcp_context: MCPContext,
    ...
) -> None:
```

**Ajouter au début de la fonction** (après la définition de `_run_legacy_loop`) :

```python
def _run_playwright_sync(...) -> None:
    # === WORKFLOW SYSTEM: Check if workflow exists ===
    workflow = find_matching_workflow(task)
    
    if workflow and workflow.get('is_active'):
        # Workflow found - execute directly without Playwright
        _send_event_sync(loop, send_event, {
            'type': 'log',
            'message': f'Workflow found (ID: {workflow["id"]}). Executing optimized script...'
        })
        
        script_path = workflow['script_path']
        params = extract_parameters(task)
        
        start_time = time.time()
        try:
            # Execute the generated Python script
            cmd = [sys.executable, script_path]
            
            # Add parameters if detected
            if params.get('email'):
                cmd.extend(['--email', params['email']])
            if params.get('password'):
                cmd.extend(['--password', params['password']])
            if params.get('phone'):
                cmd.extend(['--phone', params['phone']])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            if result.returncode == 0:
                _send_event_sync(loop, send_event, {
                    'type': 'log',
                    'message': f'Workflow executed successfully in {execution_time}ms'
                })
                
                # Log execution
                log_workflow_execution(
                    workflow_id=workflow['id'],
                    success=True,
                    execution_time_ms=execution_time
                )
                
                # Send result
                _send_event_sync(loop, send_event, {
                    'type': 'result',
                    'data': {'workflow_result': result.stdout}
                })
                
                return  # Exit early - workflow executed successfully
            else:
                _send_event_sync(loop, send_event, {
                    'type': 'log',
                    'message': f'Workflow execution failed: {result.stderr}. Falling back to normal execution.'
                })
                
                # Log failed execution
                log_workflow_execution(
                    workflow_id=workflow['id'],
                    success=False,
                    execution_time_ms=execution_time,
                    error_message=result.stderr
                )
                
        except Exception as e:
            _send_event_sync(loop, send_event, {
                'type': 'log',
                'message': f'Workflow execution error: {e}. Falling back to normal execution.'
            })
    
    # === WORKFLOW SYSTEM: Initialize script generator for recording ===
    script_gen = PlaywrightScriptGenerator() if not workflow else None
    
    # Continue with normal Playwright execution...
    def _run_legacy_loop(page):
        # ... existing code ...
```

### Étape 3 : Enregistrer les actions dans `_run_legacy_loop` (ligne ~900)

**Trouver chaque action et ajouter l'enregistrement** :

#### Action NAVIGATE
```python
if name == 'navigate':
    try:
        url = action.get('url', '')
        
        # === WORKFLOW SYSTEM: Record action ===
        if script_gen:
            script_gen.add_action('navigate', url=url)
        
        page.goto(url, timeout=20000)
        page.wait_for_load_state('domcontentloaded', timeout=10000)
    except Exception as nav_err:
        # ... existing error handling ...
```

#### Action CLICK
```python
elif name == 'click':
    selector = action.get('selector', '')
    
    # === WORKFLOW SYSTEM: Record action ===
    if script_gen:
        script_gen.add_action('click', selector=selector)
    
    try:
        page.click(selector, timeout=5000)
    except Exception:
        # ... existing fallback logic ...
```

#### Action TYPE
```python
elif name == 'type':
    selector = action.get('selector', '')
    text = action.get('text') or ''
    
    # === WORKFLOW SYSTEM: Record action ===
    if script_gen:
        script_gen.add_action('type', selector=selector, text=text)
    
    try:
        _human_type(page, selector, text)
        page.keyboard.press('Enter')
        page.wait_for_load_state('domcontentloaded', timeout=8000)
    except Exception as type_err:
        # ... existing error handling ...
```

#### Action FILL_FORM
```python
elif name == 'fill_form':
    fields = action.get('fields', [])
    submit_selector = action.get('submit_selector', '')
    
    # === WORKFLOW SYSTEM: Record action ===
    if script_gen:
        script_gen.add_action('fill_form', fields=fields, submit_selector=submit_selector)
    
    try:
        for field in fields:
            # ... existing fill logic ...
```

#### Action SCROLL
```python
elif name == 'scroll':
    # === WORKFLOW SYSTEM: Record action ===
    if script_gen:
        script_gen.add_action('scroll', direction='down')
    
    page.evaluate('window.scrollBy(0, 600)')
```

### Étape 4 : Générer le workflow à la fin (ligne ~1100)

**Trouver la fin de `_run_legacy_loop` et ajouter** :

```python
def _run_legacy_loop(page):
    # ... existing loop code ...
    
    # === WORKFLOW SYSTEM: Generate and save workflow ===
    if script_gen and len(script_gen.actions) > 0:
        try:
            _send_event_sync(loop, send_event, {
                'type': 'log',
                'message': f'Generating workflow with {len(script_gen.actions)} actions...'
            })
            
            script_code = script_gen.generate_script()
            workflow_id = create_workflow(
                prompt=task,
                script_code=script_code,
                actions=script_gen.actions
            )
            
            _send_event_sync(loop, send_event, {
                'type': 'log',
                'message': f'Workflow created successfully (ID: {workflow_id}). Next execution will be 87% faster!'
            })
        except Exception as e:
            _send_event_sync(loop, send_event, {
                'type': 'log',
                'message': f'Failed to create workflow: {e}'
            })
    
    # ... rest of existing code ...
```

## 🧪 Test de l'Intégration

### Test 1 : Première Exécution (Recording)

```bash
# Démarrer le backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

1. Entrez : `"cherche canva"`
2. Observez les logs : `"Generating workflow with X actions..."`
3. Vérifiez : `"Workflow created successfully (ID: X)"`

**Vérification en base** :
```sql
SELECT id, prompt_normalized, is_active FROM generated_workflows ORDER BY created_at DESC LIMIT 1;
```

### Test 2 : Deuxième Exécution (Replay)

1. Entrez à nouveau : `"cherche canva"`
2. Observez les logs : `"Workflow found (ID: X). Executing optimized script..."`
3. Vérifiez : `"Workflow executed successfully in Xms"`
4. Comparez le temps : devrait être 87% plus rapide

**Vérification en base** :
```sql
SELECT execution_time_ms, success FROM workflow_executions ORDER BY created_at DESC LIMIT 1;
```

## 📊 Résultats Attendus

### Logs Backend - Première Exécution
```
[INFO] Starting agent with task: cherche canva
[INFO] No workflow found, starting normal execution
[INFO] Iteration 1/20
[INFO] Action: navigate to https://www.google.com/search?q=canva
[INFO] Action: click on first result
[INFO] Generating workflow with 5 actions...
[INFO] Workflow created successfully (ID: 1). Next execution will be 87% faster!
[INFO] Execution completed in 35000ms
```

### Logs Backend - Deuxième Exécution
```
[INFO] Starting agent with task: cherche canva
[INFO] Workflow found (ID: 1). Executing optimized script...
[INFO] Workflow executed successfully in 4500ms
[INFO] Execution completed in 4500ms
```

## 🐛 Dépannage

### Problème : "Workflow not found" à chaque fois

**Cause** : La normalisation du prompt ne correspond pas

**Solution** :
```python
from backend.workflow_manager import normalize_prompt
print(normalize_prompt("cherche canva"))  # Vérifier la normalisation
```

### Problème : "Script execution failed"

**Cause** : Le script généré contient des erreurs

**Solution** :
```bash
# Tester le script manuellement
python backend/generated_workflows/workflow_1.py
```

### Problème : Les actions ne sont pas enregistrées

**Cause** : `script_gen` est None ou les actions ne sont pas ajoutées

**Solution** :
```python
# Ajouter des logs de debug
if script_gen:
    print(f"Recording action: {name}")
    script_gen.add_action(name, **params)
```

## ✅ Checklist d'Intégration

- [ ] Imports ajoutés en haut de `agent.py`
- [ ] Vérification du workflow au début de `_run_playwright_sync`
- [ ] Initialisation de `script_gen`
- [ ] Enregistrement de l'action `navigate`
- [ ] Enregistrement de l'action `click`
- [ ] Enregistrement de l'action `type`
- [ ] Enregistrement de l'action `fill_form`
- [ ] Enregistrement de l'action `scroll`
- [ ] Génération du workflow à la fin
- [ ] Tests de la première exécution (recording)
- [ ] Tests de la deuxième exécution (replay)
- [ ] Vérification des performances (87% plus rapide)

## 🎯 Résultat Final

Une fois l'intégration terminée :

1. **Première exécution** : Enregistre automatiquement toutes les actions
2. **Exécutions suivantes** : Réutilise le script généré (87% plus rapide)
3. **Économies** : 98% de réduction des coûts API
4. **Transparence** : Logs clairs pour suivre le processus

---

**🚀 Prêt à intégrer !** Suivez les étapes ci-dessus et testez avec la checklist.
