import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

MEMORY_FILE = os.path.join(os.path.dirname(__file__), 'gsam_memory.json')
MAX_SESSIONS = 50
MAX_CONVERSATION_MESSAGES = 200
MAX_SESSION_MESSAGES = 120
SUMMARY_MESSAGE_WINDOW = 12


def _default_data() -> Dict[str, Any]:
    return {
        'sessions': [],
        'conversation': [],
        'active_session': None,
    }


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + 'Z'


def _normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '').strip())


def _topic_keywords(text: str) -> set:
    words = re.findall(r'[a-zA-Z0-9]{4,}', (text or '').lower())
    stop_words = {
        'with', 'this', 'that', 'from', 'into', 'about', 'please', 'could', 'would',
        'have', 'what', 'when', 'where', 'which', 'pour', 'avec', 'dans', 'comment',
        'projet', 'rapport', 'session', 'topic', 'sujet', 'agent', 'task',
    }
    return {word for word in words if word not in stop_words}


def _is_new_topic(task: str, active_session: Optional[Dict[str, Any]]) -> bool:
    if not active_session:
        return False

    lowered = (task or '').lower()
    explicit_new_topic = (
        'new topic', 'nouveau sujet', 'autre sujet', 'change topic',
        'change de sujet', 'commence un nouveau sujet', 'start a new topic',
    )
    if any(marker in lowered for marker in explicit_new_topic):
        return True

    previous_topic = active_session.get('topic', '')
    if not previous_topic:
        return False

    new_words = _topic_keywords(task)
    old_words = _topic_keywords(previous_topic)
    if not new_words or not old_words:
        return False

    overlap = len(new_words & old_words)
    return overlap == 0


def _session_message(role: str, content: str, task: str = '') -> Dict[str, Any]:
    return {
        'timestamp': _utc_now(),
        'role': role,
        'content': content,
        'task': task,
    }


def _build_summary(messages: List[Dict[str, Any]], topic: str, fallback_task: str = '') -> str:
    if not messages:
        return _normalize_text(topic or fallback_task or 'No conversation yet.')

    user_messages = [msg['content'] for msg in messages if msg.get('role') == 'user' and msg.get('content')]
    agent_messages = [msg['content'] for msg in messages if msg.get('role') == 'agent' and msg.get('content')]
    recent_messages = messages[-SUMMARY_MESSAGE_WINDOW:]

    lines = []
    if topic:
        lines.append(f'Topic: {topic}')
    if fallback_task and fallback_task != topic:
        lines.append(f'Current task: {fallback_task}')
    if user_messages:
        lines.append(f'Initial user goal: {user_messages[0]}')
    if len(user_messages) > 1:
        lines.append(f'Latest user instruction: {user_messages[-1]}')
    if agent_messages:
        lines.append(f'Latest agent outcome: {agent_messages[-1]}')

    important_bits = []
    for msg in recent_messages:
        content = _normalize_text(msg.get('content', ''))
        role = (msg.get('role') or '').upper()
        if content:
            important_bits.append(f'[{role}] {content}')
    if important_bits:
        lines.append('Recent context: ' + ' | '.join(important_bits[-6:]))

    return '\n'.join(lines)


def _make_session(topic: str = '', initial_task: str = '') -> Dict[str, Any]:
    normalized_topic = _normalize_text(topic or initial_task or 'Untitled topic')
    now = _utc_now()
    return {
        'id': str(uuid.uuid4()),
        'topic': normalized_topic,
        'created_at': now,
        'updated_at': now,
        'status': 'active',
        'messages': [],
        'summary': normalized_topic,
        'task_history': [],
    }


def _ensure_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    if 'sessions' not in data or not isinstance(data['sessions'], list):
        data['sessions'] = []
    if 'conversation' not in data or not isinstance(data['conversation'], list):
        data['conversation'] = []
    if 'active_session' not in data:
        data['active_session'] = None

    active = data.get('active_session')
    if active is None and data.get('conversation'):
        active = _make_session(topic='Recovered legacy session')
        active['messages'] = data['conversation'][-MAX_SESSION_MESSAGES:]
        active['summary'] = _build_summary(active['messages'], active['topic'])
        active['updated_at'] = _utc_now()
        data['active_session'] = active

    if active:
        active.setdefault('id', str(uuid.uuid4()))
        active.setdefault('topic', 'Untitled topic')
        active.setdefault('created_at', _utc_now())
        active.setdefault('updated_at', active.get('created_at', _utc_now()))
        active.setdefault('status', 'active')
        active.setdefault('messages', [])
        active.setdefault('summary', active.get('topic', 'Untitled topic'))
        active.setdefault('task_history', [])
        active['messages'] = active['messages'][-MAX_SESSION_MESSAGES:]

    return data


def _load_raw() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_FILE):
        return _default_data()
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return _ensure_schema(raw)
    except Exception:
        return _default_data()


def _save_raw(data: Dict[str, Any]) -> None:
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


def _archive_active_session(data: Dict[str, Any], reason: str = 'archived') -> None:
    active = data.get('active_session')
    if not active:
        data['conversation'] = []
        return

    active['status'] = reason
    active['updated_at'] = _utc_now()
    active['summary'] = _build_summary(
        active.get('messages', []),
        active.get('topic', ''),
        active.get('task_history', [{}])[-1].get('task', '') if active.get('task_history') else '',
    )
    data['sessions'].append(active)
    data['sessions'] = data['sessions'][-MAX_SESSIONS:]
    data['active_session'] = None
    data['conversation'] = []


def ensure_topic_session(task: str, force_new: bool = False) -> Dict[str, Any]:
    data = _load_raw()
    active = data.get('active_session')

    if force_new or _is_new_topic(task, active):
        _archive_active_session(data, reason='topic_switched')
        active = None

    if not active:
        active = _make_session(topic=task, initial_task=task)
        data['active_session'] = active

    active['updated_at'] = _utc_now()
    active['status'] = 'active'
    if task and not active.get('messages'):
        active['topic'] = _normalize_text(task)
        active['summary'] = active['topic']

    data['conversation'] = active.get('messages', [])[-MAX_CONVERSATION_MESSAGES:]
    _save_raw(data)
    return active


def get_active_session() -> Optional[Dict[str, Any]]:
    data = _load_raw()
    return data.get('active_session')


def append_conversation(role: str, content: str, task: str = '') -> None:
    data = _load_raw()
    active = data.get('active_session')
    if not active:
        active = _make_session(topic=task or content, initial_task=task or content)
        data['active_session'] = active

    message = _session_message(role, content, task=task)
    active['messages'].append(message)
    active['messages'] = active['messages'][-MAX_SESSION_MESSAGES:]
    active['updated_at'] = message['timestamp']
    if role == 'user' and task:
        active['topic'] = active.get('topic') or _normalize_text(task)
    active['summary'] = _build_summary(active['messages'], active.get('topic', ''), task)

    data['conversation'] = active['messages'][-MAX_CONVERSATION_MESSAGES:]
    _save_raw(data)


def get_conversation_history(n: int = 20, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    data = _load_raw()
    if session_id:
        sessions = data.get('sessions', [])
        target = next((session for session in sessions if session.get('id') == session_id), None)
        messages = (target or {}).get('messages', [])
    else:
        active = data.get('active_session') or {}
        messages = active.get('messages', [])

    if n == 0:
        return messages
    return messages[-n:]


def archive_and_reset() -> None:
    data = _load_raw()
    _archive_active_session(data, reason='reset')
    _save_raw(data)


def clear_conversation() -> None:
    data = _load_raw()
    active = data.get('active_session')
    if active:
        active['messages'] = []
        active['summary'] = _normalize_text(active.get('topic', 'No conversation yet.'))
        active['updated_at'] = _utc_now()
    data['conversation'] = []
    _save_raw(data)


def save_session(
    task: str,
    intent: str,
    result_summary: str,
    extracted_data: Dict = None,
    status: str = 'done',
) -> None:
    data = _load_raw()
    active = data.get('active_session')
    if not active:
        active = _make_session(topic=task, initial_task=task)
        data['active_session'] = active

    task_entry = {
        'timestamp': _utc_now(),
        'task': task,
        'intent': intent,
        'result': result_summary,
        'status': status,
        'extracted_data': extracted_data or {},
    }
    active.setdefault('task_history', []).append(task_entry)
    active['task_history'] = active['task_history'][-MAX_SESSIONS:]
    active['updated_at'] = task_entry['timestamp']
    active['summary'] = _build_summary(active.get('messages', []), active.get('topic', ''), task)
    data['conversation'] = active.get('messages', [])[-MAX_CONVERSATION_MESSAGES:]
    _save_raw(data)
    append_conversation('agent', f'Task completed: {result_summary}', task=task)


def _session_snapshot(session: Dict[str, Any]) -> Dict[str, Any]:
    task_history = session.get('task_history', [])
    latest_task = task_history[-1] if task_history else {}
    return {
        'timestamp': session.get('updated_at') or session.get('created_at', ''),
        'task': latest_task.get('task') or session.get('topic', ''),
        'intent': latest_task.get('intent', 'TOPIC_SESSION'),
        'result': latest_task.get('result') or session.get('summary', ''),
        'status': latest_task.get('status') or session.get('status', ''),
        'topic': session.get('topic', ''),
        'summary': session.get('summary', ''),
        'session_id': session.get('id', ''),
    }


def get_recent_sessions(n: int = 10, include_active: bool = False) -> List[Dict[str, Any]]:
    data = _load_raw()
    snapshots = [_session_snapshot(session) for session in data.get('sessions', [])]
    if include_active and data.get('active_session'):
        snapshots.append(_session_snapshot(data['active_session']))
    return snapshots[-n:]


def get_errors_from_history(n_conv: int = 30) -> List[str]:
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
    data = _load_raw()
    active = data.get('active_session')
    lines = ['=== CONTEXT INSTRUCTION (PRIORITY) ===']

    if active:
        lines.append('## ACTIVE SESSION:')
        lines.append(f'  Topic: {active.get("topic", "Untitled topic")}')
        lines.append(f'  Session ID: {active.get("id", "")}')
        summary = active.get('summary', '')
        if summary:
            for part in summary.splitlines():
                lines.append(f'  {part}')
        lines.append('')

    errors = get_errors_from_history(n_conv * 2)
    if errors:
        lines.append('## PAST ERRORS TO AVOID REPEATING:')
        for e in errors[-5:]:
            lines.append(f'  - {e}')
        lines.append('')

    sessions = get_recent_sessions(n_sessions, include_active=False)
    if sessions:
        lines.append('## PREVIOUS TOPIC SESSIONS:')
        for s in reversed(sessions):
            ts = s.get('timestamp', '')[:10]
            topic = s.get('topic') or s.get('task', '')
            result = s.get('result', '')
            status = (s.get('status') or '').upper()
            lines.append(f'  [{ts}] [{status}] {topic}')
            if result:
                lines.append(f'    => {result}')
        lines.append('')

    conv = get_conversation_history(n_conv)
    if conv:
        lines.append('## ACTIVE SESSION RECENT MESSAGES:')
        for msg in conv:
            ts = msg.get('timestamp', '')[:16].replace('T', ' ')
            role = msg.get('role', '?').upper()
            content = msg.get('content', '')
            lines.append(f'  [{ts}] {role}: {content}')
        lines.append('')

    lines.append('RULES FROM MEMORY:')
    lines.append('- Stay inside the active topic unless the user clearly starts a new one.')
    lines.append('- Use the active session summary first, then the recent messages.')
    lines.append('- Do NOT mix details from previous topic sessions into the current one without confirmation.')
    lines.append('- If a past error is listed, use a different approach this time.')
    lines.append(f'- Current user task: {current_task}')
    lines.append('=== END CONTEXT INSTRUCTION ===')
    return '\n'.join(lines)
