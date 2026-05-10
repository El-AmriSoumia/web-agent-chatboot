"""
Workflow Manager - Gestion des workflows enregistrés et générés
"""
import json
import os
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4
from backend.auth_db import get_conn, _now, _json_dump, _json_load, _json_date


def normalize_prompt(prompt: str) -> str:
    """Normalise un prompt pour la comparaison."""
    normalized = prompt.lower().strip()
    # Remplace les emails par un placeholder
    normalized = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<EMAIL>', normalized)
    # Remplace les numéros par un placeholder
    normalized = re.sub(r'\b\d+\b', '<NUMBER>', normalized)
    # Remplace les URLs par un placeholder
    normalized = re.sub(r'https?://[^\s]+', '<URL>', normalized)
    # Supprime les mots de liaison
    stop_words = ['avec', 'with', 'pour', 'for', 'sur', 'on', 'à', 'to', 'de', 'of']
    words = normalized.split()
    normalized = ' '.join([w for w in words if w not in stop_words])
    return normalized


def extract_parameters_from_prompt(prompt: str, expected_params: Dict[str, str]) -> Dict[str, str]:
    """Extrait les paramètres d'un prompt."""
    params = {}
    
    # Extrait l'email
    if 'email' in expected_params:
        email_match = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', prompt)
        if email_match:
            params['email'] = email_match.group(0)
    
    # Extrait le téléphone
    if 'phone' in expected_params:
        phone_match = re.search(r'\+?\d{10,}', prompt.replace(' ', ''))
        if phone_match:
            params['phone'] = phone_match.group(0)
    
    # Pour le password, on ne peut pas l'extraire du prompt
    # Il faudra le demander à l'utilisateur
    
    return params


def find_matching_workflow(user_id: str, prompt: str) -> Optional[Dict]:
    """Trouve un workflow enregistré qui correspond au prompt."""
    normalized = normalize_prompt(prompt)
    
    with get_conn() as conn:
        workflows = conn.execute(
            '''SELECT * FROM generated_workflows 
               WHERE "userId" = %s AND "isActive" = TRUE 
               ORDER BY "updatedAt" DESC''',
            (user_id,)
        ).fetchall()
        
        for workflow in workflows:
            pattern = workflow.get('promptPattern', '')
            # Comparaison par similarité
            if pattern and _calculate_similarity(pattern, normalized) > 0.7:
                return dict(workflow)
    
    return None


def _calculate_similarity(pattern: str, prompt: str) -> float:
    """Calcule la similarité entre deux chaînes."""
    pattern_words = set(pattern.split())
    prompt_words = set(prompt.split())
    
    if not pattern_words or not prompt_words:
        return 0.0
    
    intersection = pattern_words & prompt_words
    union = pattern_words | prompt_words
    
    return len(intersection) / len(union)


def create_workflow(
    user_id: str,
    prompt: str,
    script_code: str,
    parameters: Dict[str, str],
    name: str = None,
    description: str = None
) -> Dict:
    """Crée un nouveau workflow."""
    workflow_id = str(uuid4())
    normalized_pattern = normalize_prompt(prompt)
    workflow_name = name or prompt[:100]
    
    # Crée le dossier workflows s'il n'existe pas
    workflows_dir = os.path.join(os.path.dirname(__file__), 'workflows')
    os.makedirs(workflows_dir, exist_ok=True)
    
    # Génère un nom de fichier sûr
    safe_name = re.sub(r'[^\w\s-]', '', workflow_name.lower())
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    file_path = os.path.join(workflows_dir, f'{safe_name}.py')
    
    # Sauvegarde le script dans un fichier
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(script_code)
    
    with get_conn() as conn:
        workflow = conn.execute(
            '''INSERT INTO generated_workflows 
               (id, "userId", "workflowName", "promptPattern", "scriptCode", parameters, "filePath", "createdAt", "updatedAt", "isActive")
               VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, TRUE)
               RETURNING *''',
            (workflow_id, user_id, workflow_name, normalized_pattern, script_code, 
             _json_dump(parameters), file_path, _now(), _now())
        ).fetchone()
        conn.commit()
    
    return dict(workflow)


def record_workflow_action(
    workflow_id: str,
    step_number: int,
    action_type: str,
    action_data: Dict[str, Any],
    page_url: str = None,
    selector: str = None,
    input_value: str = None,
    success: bool = True,
    error_message: str = None,
    execution_time_ms: int = 0
) -> Dict:
    """Enregistre une action dans le workflow."""
    action_id = str(uuid4())
    
    with get_conn() as conn:
        action = conn.execute(
            '''INSERT INTO workflow_actions 
               (id, "workflowId", "stepNumber", "actionType", "actionData", "pageUrl", 
                selector, "inputValue", success, "errorMessage", "executionTimeMs", "createdAt")
               VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
               RETURNING *''',
            (action_id, workflow_id, step_number, action_type, _json_dump(action_data),
             page_url, selector, input_value, success, error_message, execution_time_ms, _now())
        ).fetchone()
        conn.commit()
    
    return dict(action)


def get_workflow_actions(workflow_id: str) -> List[Dict]:
    """Récupère toutes les actions d'un workflow dans l'ordre."""
    with get_conn() as conn:
        actions = conn.execute(
            '''SELECT * FROM workflow_actions 
               WHERE "workflowId" = %s 
               ORDER BY "stepNumber" ASC''',
            (workflow_id,)
        ).fetchall()
    
    return [
        {
            'id': action['id'],
            'stepNumber': action['stepNumber'],
            'actionType': action['actionType'],
            'actionData': _json_load(action['actionData'], {}),
            'pageUrl': action.get('pageUrl'),
            'selector': action.get('selector'),
            'inputValue': action.get('inputValue'),
            'success': action.get('success', True),
            'errorMessage': action.get('errorMessage'),
        }
        for action in actions
    ]


def log_workflow_execution(
    workflow_id: str,
    parameters_used: Dict[str, Any],
    success: bool,
    execution_time_ms: int,
    result: Dict[str, Any] = None,
    error_message: str = None
):
    """Enregistre une exécution de workflow."""
    execution_id = str(uuid4())
    
    with get_conn() as conn:
        conn.execute(
            '''INSERT INTO workflow_executions 
               (id, "workflowId", "parametersUsed", success, "executionTimeMs", result, "errorMessage", "executedAt")
               VALUES (%s, %s, %s::jsonb, %s, %s, %s::jsonb, %s, %s)''',
            (execution_id, workflow_id, _json_dump(parameters_used), success, 
             execution_time_ms, _json_dump(result) if result else None, error_message, _now())
        )
        
        # Met à jour le compteur d'exécutions
        conn.execute(
            '''UPDATE generated_workflows 
               SET "executionCount" = "executionCount" + 1, 
                   "lastExecutedAt" = %s,
                   "updatedAt" = %s
               WHERE id = %s''',
            (_now(), _now(), workflow_id)
        )
        
        conn.commit()


def get_user_workflows(user_id: str) -> List[Dict]:
    """Récupère tous les workflows d'un utilisateur."""
    with get_conn() as conn:
        workflows = conn.execute(
            '''SELECT w.*, COUNT(a.id) as action_count
               FROM generated_workflows w
               LEFT JOIN workflow_actions a ON w.id = a."workflowId"
               WHERE w."userId" = %s
               GROUP BY w.id
               ORDER BY w."updatedAt" DESC''',
            (user_id,)
        ).fetchall()
    
    return [
        {
            'id': w['id'],
            'name': w['workflowName'],
            'pattern': w['promptPattern'],
            'parameters': _json_load(w.get('parameters'), {}),
            'actionCount': w.get('action_count', 0),
            'isActive': w.get('isActive', True),
            'executionCount': w.get('executionCount', 0),
            'lastExecutedAt': _json_date(w.get('lastExecutedAt')),
            'createdAt': _json_date(w['createdAt']),
            'updatedAt': _json_date(w['updatedAt']),
        }
        for w in workflows
    ]


def get_workflow_by_id(workflow_id: str, user_id: str) -> Optional[Dict]:
    """Récupère un workflow par son ID."""
    with get_conn() as conn:
        workflow = conn.execute(
            'SELECT * FROM generated_workflows WHERE id = %s AND "userId" = %s',
            (workflow_id, user_id)
        ).fetchone()
    
    if not workflow:
        return None
    
    return {
        'id': workflow['id'],
        'name': workflow['workflowName'],
        'pattern': workflow['promptPattern'],
        'scriptCode': workflow['scriptCode'],
        'parameters': _json_load(workflow.get('parameters'), {}),
        'filePath': workflow.get('filePath'),
        'isActive': workflow.get('isActive', True),
        'executionCount': workflow.get('executionCount', 0),
        'lastExecutedAt': _json_date(workflow.get('lastExecutedAt')),
        'createdAt': _json_date(workflow['createdAt']),
    }


def deactivate_workflow(workflow_id: str, user_id: str):
    """Désactive un workflow."""
    with get_conn() as conn:
        conn.execute(
            'UPDATE generated_workflows SET "isActive" = FALSE WHERE id = %s AND "userId" = %s',
            (workflow_id, user_id)
        )
        conn.commit()


def activate_workflow(workflow_id: str, user_id: str):
    """Active un workflow."""
    with get_conn() as conn:
        conn.execute(
            'UPDATE generated_workflows SET "isActive" = TRUE WHERE id = %s AND "userId" = %s',
            (workflow_id, user_id)
        )
        conn.commit()


def delete_workflow(workflow_id: str, user_id: str):
    """Supprime un workflow."""
    # Récupère le chemin du fichier
    workflow = get_workflow_by_id(workflow_id, user_id)
    
    if workflow and workflow.get('filePath'):
        # Supprime le fichier
        try:
            if os.path.exists(workflow['filePath']):
                os.remove(workflow['filePath'])
        except Exception:
            pass
    
    # Supprime de la base de données
    with get_conn() as conn:
        conn.execute(
            'DELETE FROM generated_workflows WHERE id = %s AND "userId" = %s',
            (workflow_id, user_id)
        )
        conn.commit()
