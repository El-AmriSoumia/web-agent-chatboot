from pathlib import Path

def write(path, text):
    path = Path(path)
    path.write_text(text, encoding='utf-8')

files = {
    'backend/main.py': """import asyncio
import json
import os
from typing import Optional

if os.name == 'nt':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent import run_agent

app = FastAPI(title='GSAM Agent Backend')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173', '*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class RunRequest(BaseModel):
    task: str

class AgentState:
    task: Optional[asyncio.Task] = None
    abort_event: Optional[asyncio.Event] = None
    confirm_event: Optional[asyncio.Event] = None
    lock: asyncio.Lock = asyncio.Lock()

state = AgentState()


def sse_format(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode('utf-8')


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.get('/status')
async def get_status():
    async with state.lock:
        running = bool(state.task and not state.task.done())
        return {'running': running, 'status': 'executing' if running else 'idle'}


@app.post('/run')
async def run(request: Request, body: RunRequest):
    async with state.lock:
        if state.task and not state.task.done():
            raise HTTPException(status_code=409, detail='Agent is already running')
        state.abort_event = asyncio.Event()
        state.confirm_event = asyncio.Event()

    queue: asyncio.Queue = asyncio.Queue()

    async def send_event(data: dict):
        await queue.put(data)

    async def agent_task():
        try:
            await run_agent(task=body.task, send_event=send_event, abort_event=state.abort_event, confirmation_event=state.confirm_event)
        except asyncio.CancelledError:
            await queue.put({'type': 'log', 'message': 'Agent canceled.'})
            await queue.put({'type': 'done'})
        except Exception as exc:
            await queue.put({'type': 'error', 'message': str(exc)})

    async with state.lock:
        state.task = asyncio.create_task(agent_task())

    async def event_stream():
        try:
            while True:
                try:
                    get_task = asyncio.ensure_future(queue.get())
                    disc_task = asyncio.ensure_future(request.is_disconnected())
                    done, pending = await asyncio.wait(
                        [get_task, disc_task],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=60
                    )
                    for t in pending:
                        t.cancel()

                    if disc_task in done and await disc_task:
                        break

                    if not done:
                        yield sse_format({'type': 'log', 'message': 'keepalive'})
                        continue

                    if get_task in done:
                        event = get_task.result()
                        yield sse_format(event)
                        if event.get('type') in {'done', 'error'}:
                            break
                except asyncio.CancelledError:
                    break
        finally:
            async with state.lock:
                if state.task and not state.task.done():
                    if state.abort_event:
                        state.abort_event.set()
                    state.task.cancel()
                state.task = None
                state.abort_event = None
                state.confirm_event = None

    return StreamingResponse(event_stream(), media_type='text/event-stream')


@app.post('/abort')
async def abort():
    async with state.lock:
        if not state.task or state.task.done():
            raise HTTPException(status_code=404, detail='No active agent to abort')
        if state.abort_event and not state.abort_event.is_set():
            state.abort_event.set()
        state.task.cancel()
    return {'status': 'aborting'}


@app.post('/confirm')
async def confirm():
    async with state.lock:
        if state.confirm_event and not state.confirm_event.is_set():
            state.confirm_event.set()
            return {'status': 'confirmed'}
    return {'status': 'no_pending_confirmation'}


@app.post('/browser/{action}')
async def browser_action(action: str):
    valid = {'back', 'forward', 'reload'}
    if action not in valid:
        raise HTTPException(status_code=400, detail='Invalid browser action')
    async with state.lock:
        if not state.task or state.task.done():
            return {'status': 'no_active_agent'}
    return {'status': 'accepted', 'action': action}
""",

    'backend/nlp.py': """import asyncio
import json
import re
from typing import Dict

import google.generativeai as genai


def _parse_json_safe(text: str) -> dict:
    text = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`').strip()
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


async def analyze_task(text: str) -> Dict:
    """Use Gemini to analyze task intent, extract the entity, and decompose into subtasks."""
    prompt = f"""You are a web task analyzer. Analyze this task and return ONLY a raw JSON object, no markdown.

Task: \"{text}\"

Return this exact structure:
{{
  "intent": "one of: SCAN_SEARCH | FORM_FILL | DATA_EXTRACT | VAULT_EXPORT | DEEP_SWEEP",
  "entity": "the main target (domain, company, product name, etc.)",
  "subtasks": ["specific step 1", "specific step 2", "specific step 3", "specific step 4"]
}}

Rules:
- intent must be exactly one of the 5 values above
- entity must be extracted from the task, not generic
- subtasks must be specific to THIS task, not generic steps
- subtasks should be 3-5 items max"""

    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        response = await asyncio.to_thread(model.generate_content, prompt)
        raw = response.text if hasattr(response, 'text') else str(response)
        result = _parse_json_safe(raw)
        if result and 'intent' in result and 'entity' in result and 'subtasks' in result:
            return result
    except Exception:
        pass

    lower = text.lower()
    if any(w in lower for w in ['search', 'find', 'cherche', 'trouve', 'look for']):
        intent = 'SCAN_SEARCH'
    elif any(w in lower for w in ['fill', 'form', 'submit', 'remplis']):
        intent = 'FORM_FILL'
    elif any(w in lower for w in ['extract', 'scrape', 'get data', 'extrais']):
        intent = 'DATA_EXTRACT'
    elif any(w in lower for w in ['vault', 'archive', 'export', 'save']):
        intent = 'VAULT_EXPORT'
    else:
        intent = 'DEEP_SWEEP'

    words = [w for w in text.split() if len(w) > 3]
    entity = words[0].upper() if words else 'TARGET'

    subtask_map = {
        'SCAN_SEARCH': ['Open browser', 'Navigate to search engine', 'Enter query', 'Extract results'],
        'FORM_FILL': ['Navigate to page', 'Identify form fields', 'Fill inputs', 'Submit form'],
        'DATA_EXTRACT': ['Navigate to target', 'Scan page content', 'Extract structured data', 'Format results'],
        'VAULT_EXPORT': ['Navigate to target', 'Collect data', 'Format export', 'Archive findings'],
        'DEEP_SWEEP': ['Navigate to target', 'Analyze page structure', 'Execute sweep', 'Archive findings'],
    }

    return {
        'intent': intent,
        'entity': entity,
        'subtasks': subtask_map.get(intent, ['Analyze task', 'Execute', 'Report results'])
    }


# Backward-compatible stubs

def detect_intent(text: str) -> str:
    return 'DEEP_SWEEP'


def detect_entity(text: str) -> str:
    return 'TARGET'


def decompose_task(text: str):
    return ['Analyzing task...']
""",

    'backend/agent.py': """import asyncio
import base64
import json
import os
import re
import time
from typing import Any, Dict, Optional
import io

if os.name == 'nt':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass

from dotenv import load_dotenv
from PIL import Image
from playwright.sync_api import sync_playwright
import google.generativeai as genai

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

PROVIDER = os.getenv('PROVIDER', 'gemini').strip().lower()

if os.getenv('GEMINI_API_KEY'):
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

api_key = os.getenv('GEMINI_API_KEY')
if PROVIDER == 'gemini' and (not api_key or api_key == 'your_gemini_api_key'):
    raise RuntimeError(
        'GEMINI_API_KEY is not set. '
        'Create a backend/.env file with GEMINI_API_KEY=your_real_key'
    )

DEFAULT_GEMINI_MODELS = [
    'models/gemini-2.0-flash',
    'models/gemini-2.0-flash-lite',
    'gemini-1.5-flash'
]


def _create_gemini_model(model_name: str):
    return genai.GenerativeModel(model_name)


def _to_base64_png(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _get_page_text(page, max_chars: int = 3000) -> str:
    try:
        text = page.evaluate("""() => {
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                null
            );
            const texts = [];
            let node;
            while (node = walker.nextNode()) {
                const t = node.textContent.trim();
                if (t.length > 1) texts.push(t);
            }
            return texts.join(' ').replace(/\s+/g, ' ');
        }""")
        return text[:max_chars]
    except Exception:
        return ''


def _parse_json(text: str) -> Dict[str, Any]:
    match = re.search(r'{.*}', text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _send_event_sync(loop: asyncio.AbstractEventLoop, send_event, data: dict) -> None:
    try:
        future = asyncio.run_coroutine_threadsafe(send_event(data), loop)
        future.result(timeout=10)
    except Exception:
        pass


def _ask_gemini_sync(screenshot_base64: str, prompt: str) -> Dict[str, Any]:
    image_bytes = base64.b64decode(screenshot_base64)
    image = Image.open(io.BytesIO(image_bytes))

    raw = None
    last_error = None
    quota_keywords = ['429', 'quota', 'rate-limit', 'rate limit', 'free_tier']
    for model_name in DEFAULT_GEMINI_MODELS:
        for attempt in range(2):
            try:
                current_model = _create_gemini_model(model_name)
                response = current_model.generate_content([
                    prompt,
                    image
                ])
                raw = response.text if hasattr(response, 'text') else str(response)
                break
            except Exception as e:
                err_str = str(e).lower()
                last_error = e
                if any(keyword in err_str for keyword in quota_keywords):
                    return {
                        'error': str(e),
                        'quota_exceeded': True,
                        'raw_error': err_str
                    }
                if 'not found' in err_str or 'unsupported' in err_str or '404' in err_str:
                    break
                return {'error': str(e)}
        if raw is not None:
            break

    if raw is None:
        if last_error:
            return {'error': f'Gemini model error: {last_error}'}
        return {'error': 'Gemini failed to produce a valid response'}

    raw = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()
    result = _parse_json(raw)
    if not result:
        return {'error': f'Invalid JSON from Gemini: {raw}'}
    return result


def _ask_vision_claude(screenshot_base64: str, prompt: str) -> Dict[str, Any]:
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return {'error': 'ANTHROPIC_API_KEY not configured'}
    try:
        import anthropic
        Client = getattr(anthropic, 'Anthropic', getattr(anthropic, 'Client', None))
        if Client is None:
            return {'error': 'Anthropic client library not available'}
        client = Client(api_key=api_key)
        response = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=512,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/png',
                                'data': screenshot_base64,
                            },
                        },
                        {'type': 'text', 'text': prompt}
                    ],
                }
            ]
        )
        raw = response.content[0].text if hasattr(response, 'content') else str(response)
        raw = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()
        parsed = _parse_json(raw)
        return parsed if parsed else {'error': f'Invalid JSON from Claude: {raw}'}
    except Exception as e:
        return {'error': str(e)}


def _extract_start_url(task: str) -> str:
    explicit_url = re.search(r'(https?://[^\s,;\"]+)', task, re.IGNORECASE)
    if explicit_url:
        url = explicit_url.group(1).strip().rstrip('.?,;')
        return url

    domain_match = re.search(
        r'\b([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com|net|org|io|ai|gov|edu|fr|co|us)(?:/[^\s]*)?\b',
        task,
        re.IGNORECASE
    )
    if domain_match:
        domain = domain_match.group(0).strip().rstrip('.?,;')
        return f'https://{domain}'

    return ''


async def _get_start_url(task: str) -> str:
    extracted_url = _extract_start_url(task)
    if extracted_url:
        return extracted_url

    prompt = f"""Given this web task: \"{task}\"
What is the best URL to start at? Reply with ONLY the URL, nothing else.
Examples: https://www.google.com/search?q=... or https://www.amazon.com
If unsure, reply: https://www.google.com"""
    try:
        text_model = _create_gemini_model('models/gemini-2.0-flash')
        response = await asyncio.to_thread(
            text_model.generate_content, prompt
        )
        url = response.text.strip().split('\n')[0].strip()
        if url.startswith('http'):
            return url
    except Exception:
        pass
    return 'https://www.google.com'


async def run_agent(
    task: str,
    send_event,
    abort_event: Optional[asyncio.Event] = None,
    confirmation_event: Optional[asyncio.Event] = None
) -> None:
    from backend.nlp import analyze_task

    await send_event({'type': 'thinking', 'message': 'Analyzing task with AI...'})
    nlp_result = await analyze_task(task)

    intent = nlp_result.get('intent', 'DEEP_SWEEP')
    entity = nlp_result.get('entity', 'TARGET')
    subtasks = nlp_result.get('subtasks', ['Analyze', 'Execute', 'Report'])

    await send_event({'type': 'nlp', 'intent': intent, 'entity': entity, 'subtasks': subtasks})

    if abort_event and abort_event.is_set():
        await send_event({'type': 'log', 'message': 'Abort requested before starting.'})
        await send_event({'type': 'done'})
        return

    start_url = await _get_start_url(task)
    await send_event({'type': 'log', 'message': f'Starting at: {start_url}'})

    loop = asyncio.get_running_loop()
    try:
        await asyncio.to_thread(
            _run_playwright_sync, task, loop, send_event, abort_event, confirmation_event, start_url
        )
    except Exception as exc:
        await send_event({'type': 'error', 'message': str(exc)})
    finally:
        await send_event({'type': 'done'})
""",

    'frontend/src/App.jsx': """import { useEffect, useRef, useState } from 'react'

const ASSET_IMAGE = '/gsam-bg.png'
const ASSET_FALLBACK = 'https://lh3.googleusercontent.com/aida-public/AB6AXuAUIM0D8nnGRwv_7GGid6UFv98WwpaiRop_Gtpp67YfqZkBz9sAi4-3BFd0MhBxBlbSXuHqY9mJaO599DJk3_kSMBpP_gMnMiTna-be9IeMdlUvo63hnKy__gXNTheC4550o0JMQvqCdJ9gz5_MnI1EOXhSlgfYdVD1tXzcDDpdXLT7mu4zXj_RLN0pwrYEhk87oONWHhRC_bTQqMCHVfFnEmqUl4zXteVu14wnoRzs2DI3Fcs5X8MyeoJk40Hn9_xz5U3tge0TPrC0'

const INITIAL_LOGS = [
  { time: '14:21:44', text: 'NAVIGATING TO ROOT...', active: false },
  { time: '14:22:01', text: 'RESOLVING SSL HANDSHAKE...', active: false },
  { time: '14:22:12', text: 'EXTRACTING NODE METADATA...', active: true },
  { time: '--:--:--', text: 'AWAITING NEXT ACTION', active: false, faded: true }
]

const NAV_ITEMS = [
  { label: 'Concierge', icon: 'concierge', id: 'concierge' },
  { label: 'Archives', icon: 'inventory_2', id: 'archives' },
  { label: 'Analytics', icon: 'monitoring', id: 'analytics' },
  { label: 'The Vault', icon: 'lock', id: 'vault' },
  { label: 'Live Agent', icon: 'support_agent', id: 'live' },
]

function App() {
  const BACKEND_URL = 'http://127.0.0.1:8000'
  const [command, setCommand] = useState('')
  const [logs, setLogs] = useState(INITIAL_LOGS)
  const [steps, setSteps] = useState([])
  const [agentStatus, setAgentStatus] = useState('idle')
  const [currentTurn, setCurrentTurn] = useState(0)
  const [screenshot, setScreenshot] = useState(null)
  const [currentUrl, setCurrentUrl] = useState('')
  const [showModal, setShowModal] = useState(false)
  const [safetyMessage, setSafetyMessage] = useState('')
  const [toastVisible, setToastVisible] = useState(false)
  const [started, setStarted] = useState(false)
  const [chatEvents, setChatEvents] = useState([])
  const [resultData, setResultData] = useState(null)
  const [backendAvailable, setBackendAvailable] = useState(true)
  const [userName, setUserName] = useState('Analyst')
  const [activeView, setActiveView] = useState('concierge')
  const backendControllerRef = useRef(null)
  const chatCanvasRef = useRef(null)
  const timersRef = useRef([])
  const abortedRef = useRef(false)
  const resumeRef = useRef(null)
  const commandInputRef = useRef(null)

  useEffect(() => {
    return () => {
      timersRef.current.forEach((id) => clearTimeout(id))
      timersRef.current = []
    }
  }, [])

  useEffect(() => {
    if (!toastVisible) return undefined
    const timeout = window.setTimeout(() => setToastVisible(false), 5000)
    return () => window.clearTimeout(timeout)
  }, [toastVisible])

  useEffect(() => {
    let canceled = false
    const controller = new AbortController()

    fetch(`${BACKEND_URL}/health`, { signal: controller.signal })
      .then((response) => {
        if (!canceled) setBackendAvailable(response.ok)
      })
      .catch(() => {
        if (!canceled) setBackendAvailable(false)
      })

    return () => {
      canceled = true
      controller.abort()
    }
  }, [])

  useEffect(() => {
    if (chatCanvasRef.current) {
      chatCanvasRef.current.scrollTop = chatCanvasRef.current.scrollHeight
    }
  }, [chatEvents])

  const addLog = (text, active = false) => {
    setLogs((prev) => {
      const next = [{ time: new Date().toTimeString().slice(0, 8), text, active }, ...prev]
      return next.slice(0, 6)
    })
  }

  const backendCleanup = () => {
    if (backendControllerRef.current) {
      backendControllerRef.current.abort()
      backendControllerRef.current = null
    }
  }

  const updateStep = ({ name, args = '', status }) => {
    setSteps((prev) => {
      const existing = prev.find((step) => step.name === name)
      if (existing) {
        return prev.map((step) => (step.name === name ? { ...step, args, status } : step))
      }
      return [...prev, { name, args, status }]
    })
  }

  const handleBackendEvent = (event) => {
    switch (event.type) {
      case 'log':
        addLog(event.message || 'Log event', false)
        break
      case 'thinking':
        addLog(`⟳ ${event.message || 'Processing...'}`, true)
        break
      case 'iteration':
        setCurrentTurn(event.current || 0)
        break
      case 'result':
        setResultData(event.data || {})
        setChatEvents((prev) => [...prev, { type: 'result', data: event.data || {} }])
        addLog('RESULT RECEIVED', false)
        break
      case 'url':
        if (event.value) setCurrentUrl(event.value)
        break
      case 'screenshot':
        if (event.data) setScreenshot(`data:image/png;base64,${event.data}`)
        break
      case 'step':
        updateStep({ name: event.name || 'step', args: event.args || '', status: event.status || 'pending' })
        break
      case 'safety':
        setSafetyMessage(event.explanation || 'The agent is requesting approval for a sensitive action.')
        setAgentStatus('waiting_confirmation')
        setShowModal(true)
        addLog(`SAFETY: ${event.explanation || 'Awaiting confirmation'}`, false)
        break
      case 'nlp': {
        const subtasks = Array.isArray(event.subtasks)
          ? event.subtasks
          : event.subtasks
            ? [event.subtasks]
            : []
        setChatEvents((prev) => [
          ...prev,
          {
            type: 'nlp',
            intent: event.intent || 'UNKNOWN',
            entity: event.entity || 'UNKNOWN',
            subtask: subtasks.length > 0 ? subtasks[subtasks.length - 1] : 'UNKNOWN',
            subtasks
          }
        ])
        break
      }
      case 'error':
        addLog(`ERROR: ${event.message || 'Unknown error'}`, false)
        setAgentStatus('error')
        break
      case 'done':
        setAgentStatus('complete')
        setToastVisible(true)
        break
      default:
        addLog(`EVENT: ${JSON.stringify(event)}`, false)
        break
    }
  }

  const parseSseChunk = (chunk, onEvent) => {
    const lines = chunk.split('\n')
    const dataLines = lines.filter((line) => line.startsWith('data:'))
    if (!dataLines.length) return
    const payload = dataLines.map((line) => line.slice(5).trim()).join('')
    if (!payload) return
    try {
      const event = JSON.parse(payload)
      onEvent(event)
    } catch (err) {
      addLog('SSE parse error', false)
    }
  }

  const startBackendAgent = async (task) => {
    resetAgent()
    setStarted(true)
    setAgentStatus('executing')
    setSteps([])
    setCurrentTurn(0)
    setCurrentUrl('')
    setScreenshot(null)
    setShowModal(false)
    addLog('Backend task started.', true)

    backendControllerRef.current = new AbortController()
    try {
      const response = await fetch(`${BACKEND_URL}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task }),
        signal: backendControllerRef.current.signal
      })

      if (!response.ok || !response.body) {
        throw new Error(`Backend request failed: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()

        for (const part of parts) {
          if (!part.trim()) continue
          parseSseChunk(part, handleBackendEvent)
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        addLog('Backend request aborted.', false)
      } else {
        addLog(`Erreur connexion backend: ${err.message}. Vérifiez que le serveur tourne sur ${BACKEND_URL}.`, false)
        setAgentStatus('error')
      }
    } finally {
      backendCleanup()
    }
  }

  const handleCommand = () => {
    if (!command.trim() || agentStatus === 'executing') return
    if (!backendAvailable) {
      addLog('Backend indisponible. Démarre le serveur backend et réessaie.', false)
      return
    }
    appendUserMessage(command.trim())
    setCommand('')
    if (commandInputRef.current) commandInputRef.current.blur()
    startBackendAgent(command.trim())
  }

  const handleAbort = async () => {
    abortedRef.current = true
    if (backendControllerRef.current) {
      backendControllerRef.current.abort()
      try {
        await fetch(`${BACKEND_URL}/abort`, { method: 'POST' })
      } catch (_) {
        // ignore abort errors
      }
      backendControllerRef.current = null
    }
    resetAgent()
    setAgentStatus('aborted')
    addLog('Agent loop terminated by operator.', false)
    setChatEvents((prev) => [...prev, { type: 'abort' }])
  }

  const appendUserMessage = (text) => {
    setChatEvents((prev) => [...prev, { type: 'user', text, time: new Date().toTimeString().slice(0, 8) }])
  }

  const resetAgent = () => {
    timersRef.current.forEach((id) => clearTimeout(id))
    timersRef.current = []
    abortedRef.current = false
    resumeRef.current = null
  }


  const handleConfirm = async () => {
    try {
      await fetch(`${BACKEND_URL}/confirm`, { method: 'POST' })
    } catch (_) {
      // Ignore confirmation delivery failures; continue locally if needed.
    }
    setShowModal(false)
    setSafetyMessage('')
    setAgentStatus('executing')
    if (resumeRef.current) resumeRef.current()
  }

  const exportCsv = () => {
    const data = resultData || {}
    const rows = Object.entries(data)
      .map(([k, v]) => `${k},${v}`)
      .join('\n')
    const csv = `Metric,Value\n${rows}`
    const blob = new Blob([csv], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'gsam-report.csv'
    a.click()
  }

  const exportPdf = () => {
    window.print()
  }

  const progressWidth = `${Math.min(currentTurn, 10) / 10 * 100}%`

  const handleNewInquiry = () => {
    setActiveView('concierge')
    setChatEvents([])
    setSteps([])
    setCurrentTurn(0)
    setScreenshot(null)
    setCurrentUrl('')
    setResultData(null)
    setAgentStatus('idle')
    setStarted(false)
    setCommand('')
    if (commandInputRef.current) commandInputRef.current.focus()
  }

  return (
    <div className="app-root relative text-on-surface overflow-hidden min-h-screen">
      <div className="fixed inset-0 z-[-2] bg-cover bg-center" style={{ backgroundImage: `url(${ASSET_IMAGE}), url(${ASSET_FALLBACK})` }} />
      <div className="fixed inset-0 z-[-1] bg-black/40 pointer-events-none" />
      <div className="flex h-screen w-full overflow-hidden">
        <aside className="hidden lg:flex flex-col sticky top-0 h-screen w-72 bg-black/30 border-r border-[#4d4635]/15 shadow-2xl shadow-black/40 backdrop-blur-xl z-50">
          <div className="p-6 flex items-center gap-3">
            <img
              alt="GSAM logo"
              className="w-12 h-12 rounded-lg object-cover"
              src={ASSET_IMAGE}
              onError={(e) => { e.currentTarget.src = ASSET_FALLBACK }}
            />
            <div>
              <h1 className="text-xl font-bold tracking-tighter text-[#f2ca50]">GSAM</h1>
              <p className="font-['Manrope'] tracking-widest uppercase text-[9px] opacity-60">PRIVATE INTELLIGENCE</p>
            </div>
          </div>
          <div className="px-6 mb-8">
            <button onClick={handleNewInquiry} className="w-full bg-gradient-to-r from-[#f2ca50] via-[#eac249] to-[#d4af37] text-[#3d2f00] font-bold py-3 px-4 rounded-md flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-95 shadow-2xl shadow-[#f2ca50]/35 border border-[#f2ca50]/20">
              <span className="material-symbols-outlined text-[18px]">add</span>
              <span className="font-['Manrope'] tracking-widest uppercase text-xs">NEW INQUIRY</span>
            </button>
          </div>
          <nav className="flex-1 px-4 space-y-1">
            {NAV_ITEMS.map(({ label, icon, id }) => (
              <button
                key={id}
                type="button"
                onClick={() => setActiveView(id)}
                className={`flex items-center gap-4 px-4 py-3 w-full text-left ${
                  activeView === id
                    ? 'text-[#f2ca50] font-bold border-r-2 border-[#f2ca50] bg-[#1c1b1b]/50'
                    : 'text-[#e5e2e1]/60 hover:text-[#e5e2e1] hover:bg-[#1c1b1b]'
                } group transition-all duration-300`}
              >
                <span className="material-symbols-outlined">{icon}</span>
                <span className="font-['Manrope'] tracking-widest uppercase text-xs">{label}</span>
              </button>
            ))}
          </nav>
          <div className="mt-auto border-t border-[#4d4635]/15 p-4 space-y-1">
            <a className="flex items-center gap-4 px-4 py-2 text-[#e5e2e1]/60 hover:text-[#e5e2e1] transition-colors" href="#">
              <span className="material-symbols-outlined" data-icon="account_circle">account_circle</span>
              <span className="font-['Manrope'] tracking-widest uppercase text-[10px]">Account</span>
            </a>
            <a className="flex items-center gap-4 px-4 py-2 text-[#e5e2e1]/60 hover:text-[#e5e2e1] transition-colors" href="#">
              <span className="material-symbols-outlined" data-icon="help_center">help_center</span>
              <span className="font-['Manrope'] tracking-widest uppercase text-[10px]">Support</span>
            </a>
            <div className="flex items-center gap-3 mt-4 px-4 py-2 bg-surface-container-lowest rounded-md">
              <div className="w-8 h-8 rounded-full bg-primary-container text-on-primary-container flex items-center justify-center text-[10px] font-bold">EA</div>
              <div className="flex-1 min-w-0">
                <p className="text-[10px] font-bold truncate">ANALYST_01</p>
                <p className="text-[8px] text-primary/50 tracking-widest">CLEARANCE: ALPHA</p>
              </div>
              <span className="material-symbols-outlined text-sm text-primary/40">security</span>
            </div>
          </div>
        </aside>

        <main className="flex-1 flex flex-col relative h-screen bg-transparent overflow-hidden">
          {!backendAvailable && (
            <div className="absolute inset-x-0 top-0 z-50 mx-6 mt-6 rounded-lg border border-error/30 bg-error/10 p-3 text-[11px] uppercase tracking-widest text-error shadow-lg shadow-black/10">
              Serveur backend indisponible — démarre `uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000` puis réessaie.
            </div>
          )}
          <header className="flex justify-between items-center px-6 py-3 w-full bg-[#1c1b1b] border-b border-[#4d4635]/15 backdrop-blur-2xl sticky top-0 z-50">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_#f2ca50]" />
                <span className="font-['Manrope'] uppercase tracking-widest text-[10px] text-[#f2ca50]">Status: Active</span>
              </div>
              <div className="h-4 w-px bg-[#4d4635]/40" />
              <span className="font-['Manrope'] uppercase tracking-widest text-[10px] text-[#e5e2e1]">NLP: Encrypted</span>
            </div>
            <h2 className="text-[#f2ca50] font-black tracking-widest text-xs lg:text-sm">GSAM | PRIVATE INTELLIGENCE</h2>
            <div className="flex items-center gap-4">
              <button onClick={() => setActiveView('analytics')} title="Analytics" className="text-[#e5e2e1] hover:text-[#f2ca50] transition-colors">
                <span className="material-symbols-outlined text-[18px]">hub</span>
              </button>
              <button onClick={() => setActiveView('vault')} title="Vault" className="text-[#e5e2e1] hover:text-[#f2ca50] transition-colors">
                <span className="material-symbols-outlined text-[18px]">security</span>
              </button>
              <button onClick={() => {
                const name = prompt('Enter analyst name:', userName)
                if (name) setUserName(name)
              }} title="Settings" className="text-[#e5e2e1] hover:text-[#f2ca50] transition-colors">
                <span className="material-symbols-outlined text-[18px]">settings</span>
              </button>
            </div>
          </header>

          {activeView !== 'concierge' ? (
            <div className="flex-1 items-center justify-center px-6 py-8 text-on-surface-variant/60 text-sm uppercase tracking-widest">
              <div className="mx-auto mt-24 max-w-xl rounded-3xl border border-[#4d4635]/20 bg-[#141414]/80 p-10 text-center shadow-2xl shadow-black/20 backdrop-blur-xl">
                <p className="text-[13px] font-bold text-[#f2ca50] uppercase tracking-widest mb-3">{activeView.toUpperCase()}</p>
                <p className="text-sm leading-relaxed text-[#e5e2e1]/70">Cette vue est en cours de construction. Retourne dans Concierge pour lancer une nouvelle mission.</p>
              </div>
            </div>
          ) : (
            <div ref={chatCanvasRef} className="flex-1 overflow-y-auto px-6 py-8">
              <div className="max-w-5xl mx-auto space-y-8 pb-32">
                <div className="space-y-1">
                  <h3 className="text-3xl lg:text-4xl font-bold tracking-tighter">Welcome back, <span className="text-primary">{userName}</span>.</h3>
                  <p className="text-on-surface-variant text-sm tracking-wide opacity-80 uppercase tracking-widest">Protocol initiated. Agent awaiting vector instructions.</p>
                </div>

                <div className="flex justify-end">
                  <div className="bg-surface-container-low ghost-border p-4 max-w-[80%] rounded-xl rounded-tr-none">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="bg-primary-container/20 text-primary text-[9px] px-2 py-0.5 rounded-full font-bold tracking-widest uppercase">CLIENT PRIORITY</span>
                      <span className="text-[9px] text-on-surface-variant uppercase tracking-widest">14:21:05</span>
                    </div>
                    <p className="text-sm leading-relaxed text-on-surface/90">Understood. Initializing reconnaissance sequence. Establishing encrypted browser node.</p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded bg-primary flex items-center justify-center shrink-0">
                    <span className="material-symbols-outlined text-on-primary">terminal</span>
                  </div>
                  <div className="space-y-4 flex-1">
                    <div>
                      <h4 className="text-[11px] font-bold text-primary uppercase tracking-widest">GSAM INTELLIGENCE</h4>
                      <p className="text-sm text-on-surface/90 mt-1">Understood. Initializing reconnaissance sequence. Establishing encrypted browser node.</p>
                    </div>

                    <div id="step-tracker" className="space-y-3 pl-2 border-l border-outline-variant/20 ml-1">
                      {started
                        ? steps.map((step, index) => (
                            <div key={index} className="flex items-center gap-3">
                              {step.status === 'done' ? (
                                <>
                                  <span className="material-symbols-outlined text-[#f2ca50] text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                                  <p className="text-[11px] uppercase tracking-widest text-[#e5e2e1]/60">{step.name} {step.args ? `— ${step.args}` : ''}</p>
                                </>
                              ) : step.status === 'running' ? (
                                <>
                                  <div className="w-2 h-2 rounded-full bg-[#f2ca50] animate-ping mx-1 flex-shrink-0" />
                                  <p className="text-[11px] uppercase tracking-widest text-[#f2ca50] font-bold">{step.name}...</p>
                                </>
                              ) : (
                                <>
                                  <div className="w-2 h-2 rounded-full bg-[#e5e2e1]/20 mx-1 flex-shrink-0" />
                                  <p className="text-[11px] uppercase tracking-widest text-[#e5e2e1]/25">{step.name}</p>
                                </>
                              )}
                            </div>
                          ))
                        : null}
                    </div>

                    {resultData && (
                      <div className="bg-surface-container-high ghost-border p-6 rounded-xl shadow-2xl">
                        <div className="flex justify-between items-center mb-6">
                          <h4 className="text-[11px] font-bold text-primary uppercase tracking-widest">RESULT SET</h4>
                          <div className="flex gap-2">
                            <button onClick={exportCsv} className="text-[8px] bg-surface-container-lowest border border-outline-variant/30 px-2 py-1 rounded hover:border-primary/50 transition-all">EXPORT CSV</button>
                            <button onClick={exportPdf} className="text-[8px] bg-surface-container-lowest border border-outline-variant/30 px-2 py-1 rounded hover:border-primary/50 transition-all">EXPORT PDF</button>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          {Object.entries(resultData).map(([key, val]) => (
                            <div key={key} className="p-3 bg-background/50 rounded ghost-border">
                              <p className="text-[9px] text-on-surface-variant uppercase tracking-widest mb-1">{key.replace(/_/g, ' ')}</p>
                              <p className="text-sm font-bold text-primary break-all">{String(val)}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {chatEvents.map((event, idx) => {
                      if (event.type === 'user') {
                        return (
                          <div key={idx} className="flex justify-end">
                            <div className="bg-[#1c1b1b] border border-[#4d4635]/15 p-4 max-w-[80%] rounded-xl rounded-tr-none">
                              <div className="flex items-center gap-2 mb-2">
                                <span className="bg-[#f2ca50]/20 text-[#f2ca50] text-[9px] px-2 py-0.5 rounded-full font-bold tracking-widest uppercase">CLIENT PRIORITY</span>
                                <span className="text-[9px] text-[#e5e2e1]/50 uppercase tracking-widest">{event.time}</span>
                              </div>
                              <p className="text-sm leading-relaxed text-[#e5e2e1]/90">{event.text}</p>
                            </div>
                          </div>
                        )
                      }
                      if (event.type === 'result') {
                        const entries = Object.entries(event.data || {})
                        return (
                          <div key={idx} className="flex items-start gap-4">
                            <div className="w-10 h-10 rounded bg-primary flex items-center justify-center shrink-0">
                              <span className="material-symbols-outlined text-on-primary">terminal</span>
                            </div>
                            <div className="space-y-3 flex-1">
                              <h4 className="text-[11px] font-bold text-primary uppercase tracking-widest">GSAM INTELLIGENCE — RESULT</h4>
                              <div className="bg-surface-container-high ghost-border p-5 rounded-xl">
                                <div className="grid grid-cols-2 gap-3">
                                  {entries.map(([key, val]) => (
                                    <div key={key} className="p-3 bg-background/50 rounded ghost-border">
                                      <p className="text-[9px] text-on-surface-variant uppercase tracking-widest mb-1">{key.replace(/_/g, ' ')}</p>
                                      <p className="text-sm font-bold text-primary break-all">{String(val)}</p>
                                    </div>
                                  ))}
                                </div>
                                <div className="flex gap-2 mt-4">
                                  <button onClick={exportCsv} className="text-[8px] bg-surface-container-lowest border border-outline-variant/30 px-2 py-1 rounded hover:border-primary/50 transition-all">EXPORT CSV</button>
                                  <button onClick={exportPdf} className="text-[8px] bg-surface-container-lowest border border-outline-variant/30 px-2 py-1 rounded hover:border-primary/50 transition-all">EXPORT PDF</button>
                                </div>
                              </div>
                            </div>
                          </div>
                        )
                      }
                      if (event.type === 'abort') {
                        return (
                          <div key={idx} className="text-center py-4">
                            <span className="text-[#ffb4ab] text-[11px] uppercase tracking-widest font-bold">
                              ⊗ MISSION ABORTED — Agent loop terminated by operator.
                            </span>
                          </div>
                        )
                      }
                      return (
                        <div key={idx} className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="col-span-1 md:col-span-2 bg-[#0e0e0e] border border-[#4d4635]/15 p-5 rounded-xl border-l-2 border-l-[#f2ca50]/30">
                            <h4 className="text-[10px] text-[#f2ca50] font-bold uppercase tracking-widest mb-4">NLP Analysis Card</h4>
                            <div className="flex flex-wrap gap-2">
                              <span className="px-3 py-1 bg-[#2a2a2a] text-[10px] text-[#e5e2e1]/70 border border-[#4d4635]/30 rounded uppercase tracking-widest">Intent: {event.intent}</span>
                              <span className="px-3 py-1 bg-[#2a2a2a] text-[10px] text-[#e5e2e1]/70 border border-[#4d4635]/30 rounded uppercase tracking-widest">Entity: {event.entity}</span>
                              <span className="px-3 py-1 bg-[#2a2a2a] text-[10px] text-[#f2ca50] border border-[#f2ca50]/20 rounded uppercase tracking-widest">Subtask: {event.subtask}</span>
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="absolute bottom-0 left-0 w-full p-6 bg-gradient-to-t from-background via-background/90 to-transparent">
            <div className="max-w-5xl mx-auto space-y-4">
              <div className="flex items-center gap-4">
                <div className="flex-1 h-1 bg-surface-container-highest rounded-full overflow-hidden">
                  <div id="progress-fill" className="h-full gold-gradient" style={{ width: progressWidth, transition: 'width 0.6s ease' }} />
                </div>
                <span id="turn-label" className="text-[10px] font-bold text-primary tracking-widest uppercase">TURN {currentTurn} / 10</span>
              </div>
              <div className="bg-surface-container-low/60 backdrop-blur-3xl ghost-border rounded-xl p-2 flex items-center gap-2 shadow-2xl">
                <div className="flex items-center gap-4 pl-4">
                  <span className="material-symbols-outlined text-primary/60">terminal</span>
                </div>
                <input
                  ref={commandInputRef}
                  id="command-input"
                  value={command}
                  onChange={(e) => setCommand(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      handleCommand()
                    }
                  }}
                  className="flex-1 bg-transparent border-none focus:ring-0 text-sm font-['Manrope'] uppercase tracking-widest placeholder:text-on-surface-variant/40"
                  placeholder="AWAITING MISSION PARAMETERS..."
                  type="text"
                />
                <div className="flex items-center gap-2 pr-2">
                  <button id="command-btn" type="button" onClick={handleCommand} className="bg-gradient-to-r from-[#f2ca50] via-[#eac249] to-[#d4af37] text-[#3d2f00] font-black px-6 py-2.5 rounded-lg flex items-center gap-2 hover:opacity-90 active:scale-95 transition-all shadow-2xl shadow-[#f2ca50]/35 border border-[#f2ca50]/20">
                    <span className="font-['Manrope'] text-[10px] tracking-widest uppercase">COMMAND</span>
                    <span className="material-symbols-outlined text-sm">bolt</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>

        <aside className="hidden xl:flex flex-col w-96 bg-surface-container-lowest border-l border-[#4d4635]/15 z-40 relative">
          <div className="p-4 border-b border-[#4d4635]/15 flex items-center justify-between bg-surface-container-low">
            <h3 className="text-[10px] font-bold text-primary tracking-widest uppercase">Live Browser Node</h3>
            <div className="flex gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-error/40" />
              <div className="w-1.5 h-1.5 rounded-full bg-primary/40" />
              <div className="w-1.5 h-1.5 rounded-full bg-on-surface/40" />
            </div>
          </div>
          <div className="p-4 bg-background/50 flex items-center gap-2">
            <div className="flex-1 bg-surface-container-highest px-3 py-1.5 rounded ghost-border flex items-center gap-2 truncate">
              <span className="material-symbols-outlined text-[14px] text-primary">lock</span>
              <span id="live-url" className="text-[10px] text-on-surface/70 truncate tracking-wide">{currentUrl || 'Awaiting connection...'}</span>
            </div>
            <button onClick={() => currentUrl && window.open(currentUrl, '_blank')} disabled={!currentUrl} className="p-1.5 hover:bg-surface-container-highest rounded text-on-surface-variant transition-colors disabled:opacity-30">
              <span className="material-symbols-outlined text-[18px]">open_in_new</span>
            </button>
          </div>
          <div className="flex-1 overflow-y-auto bg-surface-container-low/40 relative group">
            <div
              className="relative m-4 bg-black rounded-lg border border-outline-variant/20 shadow-2xl"
              style={{
                resize: 'both',
                overflow: 'auto',
                minWidth: '280px',
                minHeight: '420px',
                maxWidth: '100%',
                maxHeight: '80vh'
              }}
            >
              {screenshot ? (
                <img id="live-screenshot" className="w-full h-full object-cover" alt="Capture navigateur" src={screenshot} />
              ) : (
                <div className="flex items-center justify-center h-full text-on-surface-variant text-xs">
                  En attente du navigateur...
                </div>
              )}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="p-4 bg-black/80 backdrop-blur-xl border border-primary/20 rounded-xl text-center max-w-[80%]">
                  <span className="material-symbols-outlined text-primary text-3xl mb-2 animate-pulse">visibility</span>
                  <p className="text-[10px] font-bold text-primary uppercase tracking-widest">Active Monitoring</p>
                </div>
              </div>
              <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/5 to-transparent h-20 w-full animate-scan pointer-events-none" />
            </div>
            <div className="px-4 pb-8 space-y-3">
              <h5 className="text-[9px] font-bold text-on-surface-variant uppercase tracking-widest">Sequence Log</h5>
              <div id="sequence-log" className="space-y-2">
                {logs.map((log, index) => (
                  <div
                    key={index}
                    className={`text-[10px] p-2 bg-surface-container-lowest rounded border-l-2 font-mono ${
                      log.faded
                        ? 'border-on-surface/20 opacity-50'
                        : log.active
                        ? 'border-[#f2ca50] bg-[#f2ca50]/5'
                        : 'border-primary/40'
                    }`}
                  >
                    <span className="text-primary/60">[{log.time}]</span> {log.text}
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-surface-container-high p-2 rounded-full shadow-2xl border border-outline-variant/40">
            <button onClick={() => handleBrowserNav('back')} disabled={agentStatus !== 'executing'} className="w-10 h-10 flex items-center justify-center rounded-full bg-surface-container-lowest text-on-surface-variant hover:text-primary transition-all disabled:opacity-30">
              <span className="material-symbols-outlined text-sm">arrow_back</span>
            </button>
            <button onClick={() => handleBrowserNav('reload')} disabled={agentStatus !== 'executing'} className="w-10 h-10 flex items-center justify-center rounded-full bg-primary text-on-primary shadow-lg shadow-primary/20 transition-all disabled:opacity-30">
              <span className="material-symbols-outlined text-sm">refresh</span>
            </button>
            <button onClick={() => handleBrowserNav('forward')} disabled={agentStatus !== 'executing'} className="w-10 h-10 flex items-center justify-center rounded-full bg-surface-container-lowest text-on-surface-variant hover:text-primary transition-all disabled:opacity-30">
              <span className="material-symbols-outlined text-sm">arrow_forward</span>
            </button>
          </div>
        </aside>
      </div>

      {toastVisible && (
        <div className="fixed top-8 right-8 z-[100] transition-transform duration-500 ease-out flex items-center gap-4 bg-surface-container-high border-l-4 border-primary p-5 rounded shadow-2xl max-w-sm">
          <div className="p-2 rounded-full bg-primary/10 text-primary">
            <span className="material-symbols-outlined">task_alt</span>
          </div>
          <div>
            <p className="text-xs font-bold uppercase tracking-widest">MISSION COMPLETE</p>
            <p className="text-[10px] text-on-surface-variant mt-1">Protocol GSAM-9 has finished archival.</p>
          </div>
        </div>
      )}

      {showModal && (
        <div id="safety-modal" className="fixed inset-0 z-[200] bg-black/70 backdrop-blur-sm flex items-center justify-center">
          <div className="bg-[#2a2a2a] max-w-md w-full mx-4 rounded-xl p-8 border border-[#4d4635]/15 shadow-2xl shadow-black">
            <div className="text-[#f2ca50] text-5xl text-center mb-4 material-symbols-outlined">gpp_maybe</div>
            <h3 className="text-[#f2ca50] text-[11px] font-bold uppercase tracking-widest text-center mb-3">ACTION REQUIRES CONFIRMATION</h3>
            <p id="safety-explanation" className="text-[#e5e2e1]/70 text-sm text-center leading-relaxed mb-8">
              {safetyMessage || 'The agent is about to perform a sensitive action. This cannot be undone. Confirm to proceed.'}
            </p>
            <div className="flex gap-3 justify-center">
              <button id="safety-abort-btn" onClick={handleAbort} className="text-[#ffb4ab] border border-[#ffb4ab]/20 px-6 py-2.5 rounded-lg text-[10px] uppercase tracking-widest font-bold hover:bg-[#ffb4ab]/10 transition-colors">
                ABORT
              </button>
              <button id="safety-confirm-btn" onClick={handleConfirm} className="bg-gradient-to-r from-[#f2ca50] via-[#eac249] to-[#d4af37] text-[#3d2f00] px-6 py-2.5 rounded-lg text-[10px] uppercase tracking-widest font-bold hover:opacity-90 transition-opacity shadow-2xl shadow-[#f2ca50]/35 border border-[#f2ca50]/20">
                CONFIRM
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
""",

    'backend/.env.example': "# Copy to backend/.env and fill in your keys\n\n# Required: Gemini API key for Gemini provider\nGEMINI_API_KEY=your_gemini_api_key_here\n\n# Optional: Use Claude Haiku as a fallback vision provider\n# PROVIDER=claude\n# ANTHROPIC_API_KEY=your_anthropic_api_key_here\n\n# Default provider\nPROVIDER=gemini\n",

    'requirements.txt': "fastapi>=0.111.0\nuvicorn[standard]>=0.29.0\npython-dotenv>=1.0.0\npydantic>=2.0.0\nPillow>=10.0.0\nplaywright>=1.44.0\ngoogle-generativeai>=0.7.0\nanthropic>=0.28.0\n"
}

for path, content in files.items():
    write(path, content)
print('rewrite complete')
"""
}