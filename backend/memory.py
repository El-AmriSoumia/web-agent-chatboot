import json
import os
from datetime import datetime
from typing import Any, Dict, List

MEMORY_FILE = os.path.join(os.path.dirname(__file__), 'gsam_memory.json')
MAX_SESSIONS = 50


def _load_raw() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_FILE):
        return {'sessions': [], 'conversation': []}
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'sessions': [], 'conversation': []}


def _save_raw(data: Dict[str, Any]) -> None:
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


def append_conversation(role: str, content: str, task: str = '') -> None:
    """Append a message to the persistent conversation history."""
    data = _load_raw()
    if 'conversation' not in data:
        data['conversation'] = []
    data['conversation'].append({
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'role': role,       # 'user' | 'agent' | 'result'
        'content': content,
        'task': task,
    })
    # Keep last 200 messages
    data['conversation'] = data['conversation'][-200:]
    _save_raw(data)


def get_conversation_history(n: int = 20) -> List[Dict[str, Any]]:
    """Get last n conversation messages. If n=0, get all."""
    data = _load_raw()
    conv = data.get('conversation', [])
    if n == 0:
        return conv
    return conv[-n:]


def archive_and_reset() -> None:
    """Archive active conversation into a sealed session entry, then clear it."""
    data = _load_raw()
    conv = data.get('conversation', [])
    if conv:
        data.setdefault('sessions', []).append({
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'task': '[archived conversation]',
            'intent': 'ARCHIVE',
            'result': f'{len(conv)} messages archived on reset',
            'status': 'archived',
            'extracted_data': {},
        })
        data['sessions'] = data['sessions'][-MAX_SESSIONS:]
    data['conversation'] = []
    _save_raw(data)


def clear_conversation() -> None:
    """Clear conversation history (new session)."""
    data = _load_raw()
    data['conversation'] = []
    _save_raw(data)


def save_session(task: str, intent: str, result_summary: str, extracted_data: Dict = None, status: str = 'done') -> None:
    """Save a completed session to persistent memory."""
    data = _load_raw()
    session = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'task': task,
        'intent': intent,
        'result': result_summary,
        'status': status,
        'extracted_data': extracted_data or {},
    }
    data['sessions'].append(session)
    data['sessions'] = data['sessions'][-MAX_SESSIONS:]
    _save_raw(data)
    # Also append to conversation
    append_conversation('agent', f'Task completed: {result_summary}', task=task)


def get_recent_sessions(n: int = 10) -> List[Dict[str, Any]]:
    data = _load_raw()
    return data['sessions'][-n:]


def get_errors_from_history(n_conv: int = 30) -> List[str]:
    """Extract failed actions and error messages from recent conversation history."""
    errors = []
    conv = get_conversation_history(n_conv)
    for msg in conv:
        content = msg.get('content', '')
        role = msg.get('role', '')
        if role in ('agent', 'error') and any(k in content.lower() for k in (
            'error', 'failed', 'timeout', 'could not', 'invalid', 'not found', 'exception'
        )):
            errors.append(content)
    return errors


def get_memory_context(current_task: str, n_sessions: int = 5, n_conv: int = 15) -> str:
    """Build full memory context: recent conversation + past sessions + past errors."""
    lines = ['=== CONTEXT INSTRUCTION (PRIORITY) ===']

    # Past errors — shown first so the agent avoids repeating them
    errors = get_errors_from_history(n_conv * 2)
    if errors:
        lines.append('## PAST ERRORS TO AVOID REPEATING:')
        for e in errors[-5:]:
            lines.append(f'  - {e}')
        lines.append('')

    # Past completed sessions
    sessions = get_recent_sessions(n_sessions)
    if sessions:
        lines.append('## PAST COMPLETED TASKS:')
        for s in reversed(sessions):
            ts = s.get('timestamp', '')[:10]
            task = s.get('task', '')
            result = s.get('result', '')
            status = s.get('status', '').upper()
            lines.append(f'  [{ts}] [{status}] {task}')
            if result:
                lines.append(f'    => {result}')
        lines.append('')

    # Recent conversation history
    conv = get_conversation_history(n_conv)
    if conv:
        lines.append('## RECENT CONVERSATION HISTORY:')
        for msg in conv:
            ts = msg.get('timestamp', '')[:16].replace('T', ' ')
            role = msg.get('role', '?').upper()
            content = msg.get('content', '')
            lines.append(f'  [{ts}] {role}: {content}')
        lines.append('')

    lines.append('RULES FROM MEMORY:')
    lines.append('- Do NOT repeat actions or searches already completed above.')
    lines.append('- If the current task seems related to a previous one, ask the user for confirmation before continuing it.')
    lines.append('- If a past error is listed, use a different approach this time.')
    lines.append('=== END CONTEXT INSTRUCTION ===')
    return '\n'.join(lines)
