import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class MCPContext:
    def __init__(self, task: str, intent: str, entity: str, subtasks: List[str]):
        self.task_state: Dict[str, Any] = {
            'task': task,
            'intent': intent,
            'entity': entity,
            'subtasks': subtasks,
            'current_url': None,
            'iteration': 0,
            'status': 'initialized',
        }
        self.action_history: List[Dict[str, Any]] = []
        self.intermediate_results: Dict[str, Any] = {}
        self.error_log: List[Dict[str, Any]] = []

    def update_state(
        self,
        task: Optional[str] = None,
        intent: Optional[str] = None,
        entity: Optional[str] = None,
        subtasks: Optional[List[str]] = None,
        current_url: Optional[str] = None,
        iteration: Optional[int] = None,
        status: Optional[str] = None,
    ) -> None:
        if task is not None:
            self.task_state['task'] = task
        if intent is not None:
            self.task_state['intent'] = intent
        if entity is not None:
            self.task_state['entity'] = entity
        if subtasks is not None:
            self.task_state['subtasks'] = subtasks
        if current_url is not None:
            self.task_state['current_url'] = current_url
        if iteration is not None:
            self.task_state['iteration'] = iteration
        if status is not None:
            self.task_state['status'] = status

    def add_action(self, action: str, details: Dict[str, Any]) -> None:
        self.action_history.append({
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'action': action,
            'details': details,
        })

    def add_result(self, name: str, value: Any) -> None:
        self.intermediate_results[name] = value

    def add_error(self, error_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        entry: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'error': error_message,
        }
        if details is not None:
            entry['details'] = details
        self.error_log.append(entry)

    def get_context_summary(self) -> Dict[str, Any]:
        return {
            'task_state': self.task_state,
            'action_history': self.action_history,
            'intermediate_results': self.intermediate_results,
            'error_log': self.error_log,
        }

    def to_json(self) -> str:
        return json.dumps(self.get_context_summary(), default=str)
