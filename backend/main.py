import asyncio
import json
import os
import threading
import time
from typing import Optional

if os.name == 'nt':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent import run_agent
from backend.auth_db import (
    close_agent_session,
    create_agent,
    create_agent_session,
    create_token,
    delete_agent,
    find_user_by_id,
    get_agent_by_id,
    get_latest_running_session,
    get_session_for_user,
    get_session_messages_for_user,
    get_user_agents,
    get_user_sessions,
    login_user,
    mark_agent_session_running,
    public_agent,
    public_session,
    public_user,
    register_user,
    save_agent_message,
    save_screenshot,
    update_agent,
    get_last_screenshot,
    update_session_page_state,
    verify_token,
)
from backend.mcp import MCPContext
from backend.memory import archive_and_reset, get_active_session

app = FastAPI(title='GSAM Agent Backend')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:5174', 'http://127.0.0.1:5174', 'http://localhost:8000', 'http://127.0.0.1:8000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class RunRequest(BaseModel):
    task: str
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    stale_browser: bool = False
    skip_anti_bot: bool = False
    show_browser: bool = False

class FeedbackRequest(BaseModel):
    message: str

class AgentCreateRequest(BaseModel):
    name: str
    description: str
    systemContext: str

class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    systemContext: Optional[str] = None

class AuthRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(AuthRequest):
    username: str


async def get_current_user(authorization: str = Header(default='')):
    if not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Authentification requise')

    token = authorization.removeprefix('Bearer ').strip()
    try:
        payload = verify_token(token)
        user = find_user_by_id(payload['sub'])
    except Exception:
        raise HTTPException(status_code=401, detail='Session invalide ou expiree')

    if not user:
        raise HTTPException(status_code=401, detail='Utilisateur introuvable')
    return user

class AgentState:
    task: Optional[asyncio.Task] = None
    abort_event: Optional[asyncio.Event] = None
    confirm_event: Optional[asyncio.Event] = None
    context: Optional[MCPContext] = None
    mongo_session: Optional[dict] = None
    mongo_user: Optional[dict] = None
    feedback_queue: list = []
    user_reply_event: Optional[threading.Event] = None
    lock: asyncio.Lock = asyncio.Lock()

state = AgentState()


def sse_format(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode('utf-8')


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.post('/auth/register')
async def register(body: RegisterRequest):
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail='Le mot de passe doit contenir au moins 8 caracteres')
    try:
        user = register_user(body.username, body.email, body.password)
        token = create_token(user)
        return {'user': public_user(user), 'token': token}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post('/auth/login')
async def login(body: AuthRequest):
    try:
        user = login_user(body.email, body.password)
        token = create_token(user)
        return {'user': public_user(user), 'token': token}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


@app.get('/auth/me')
async def me(user=Depends(get_current_user)):
    return {'user': public_user(user)}


@app.get('/sessions')
async def list_sessions(user=Depends(get_current_user)):
    return {'sessions': get_user_sessions(user)}


@app.get('/sessions/{session_id}')
async def get_session_messages(session_id: str, user=Depends(get_current_user)):
    try:
        session, messages = get_session_messages_for_user(user, session_id)
        return {'session': session, 'messages': messages}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get('/status')
async def get_status():
    async with state.lock:
        running = bool(state.task and not state.task.done())
        return {'running': running, 'status': 'executing' if running else 'idle'}


@app.get('/sessions/{session_id}/screenshots')
async def get_session_screenshots_endpoint(session_id: str, user=Depends(get_current_user)):
    session = get_session_for_user(user, session_id)
    if not session:
        raise HTTPException(status_code=404, detail='Session introuvable')
    from backend.auth_db import get_session_screenshots
    screenshots = get_session_screenshots(session)
    return {'screenshots': screenshots}


@app.get('/agents')
async def list_agents(user=Depends(get_current_user)):
    return {'agents': [public_agent(agent) for agent in get_user_agents(user['id'])]}


@app.post('/agents')
async def create_agent_endpoint(body: AgentCreateRequest, user=Depends(get_current_user)):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail='Le nom est obligatoire')
    if not body.description.strip():
        raise HTTPException(status_code=400, detail='La description est obligatoire')
    if not body.systemContext.strip():
        raise HTTPException(status_code=400, detail='Le contexte systeme est obligatoire')
    agent = create_agent(user['id'], body.name, body.description, body.systemContext)
    return {'agent': public_agent(agent)}


@app.get('/agents/{agent_id}')
async def get_agent_endpoint(agent_id: str, user=Depends(get_current_user)):
    agent = get_agent_by_id(agent_id, user['id'])
    if not agent:
        raise HTTPException(status_code=404, detail='Agent introuvable')
    return {'agent': public_agent(agent)}


@app.patch('/agents/{agent_id}')
async def update_agent_endpoint(agent_id: str, body: AgentUpdateRequest, user=Depends(get_current_user)):
    fields = body.dict(exclude_unset=True)
    for key in ('name', 'description', 'systemContext'):
        if key in fields and not str(fields[key]).strip():
            raise HTTPException(status_code=400, detail=f'{key} ne peut pas etre vide')
    agent = update_agent(agent_id, user['id'], fields)
    if not agent:
        raise HTTPException(status_code=404, detail='Agent introuvable')
    return {'agent': public_agent(agent)}


@app.delete('/agents/{agent_id}')
async def delete_agent_endpoint(agent_id: str, user=Depends(get_current_user)):
    if not delete_agent(agent_id, user['id']):
        raise HTTPException(status_code=404, detail='Agent introuvable')
    return {'status': 'deleted', 'agent_id': agent_id}


@app.get('/context')
async def get_context():
    async with state.lock:
        if state.context:
            return {
                'context': state.context.get_context_summary(),
                'active_session': get_active_session(),
            }
        return {'context': None, 'active_session': get_active_session()}


@app.get('/session')
async def get_session():
    return {'active_session': get_active_session()}


@app.post('/run')
async def run(request: Request, body: RunRequest, user=Depends(get_current_user)):
    print(f"BACKEND: Received /run request with task: {body.task}")
    async with state.lock:
        if state.task and not state.task.done():
            print("BACKEND: Agent already running, rejecting request")
            raise HTTPException(status_code=409, detail='Agent is already running')
        state.abort_event = asyncio.Event()
        state.confirm_event = asyncio.Event()
        state.feedback_queue = []
        state.user_reply_event = threading.Event()

    queue: asyncio.Queue = asyncio.Queue()
    selected_agent = None
    start_time = time.time()
    if body.session_id:
        mongo_session = get_session_for_user(user, body.session_id)
        if not mongo_session:
            raise HTTPException(status_code=404, detail='Session introuvable')
        mark_agent_session_running(mongo_session)
        if mongo_session.get('agentId'):
            selected_agent = {
                'name': mongo_session.get('agentName') or '',
                'description': mongo_session.get('agentDescription') or '',
                'systemContext': mongo_session.get('agentSystemContext') or '',
            }
    else:
        if body.agent_id:
            selected_agent = get_agent_by_id(body.agent_id, user['id'])
            if not selected_agent:
                raise HTTPException(status_code=404, detail='Agent introuvable')
        mongo_session = create_agent_session(user, body.task, agent_id=body.agent_id if selected_agent else None)
    save_agent_message(mongo_session, user, 'user', body.task)

    current_url_holder = ['']  # mutable container to track current URL in closure

    async def send_event(data: dict):
        event_type = data.get('type')
        if event_type == 'url':
            current_url_holder[0] = data.get('value', '')
        if event_type == 'step':
            generated_by = next(
                (tool for tool in ['langchain', 'playwright', 'rpa', 'gemini', 'computer_use']
                 if tool in f"{data.get('name', '')} {data.get('args', '')}".lower()),
                None,
            )
            step_content = json.dumps({
                'action': data.get('name'),
                'args': data.get('args'),
                'status': data.get('status'),
            }, ensure_ascii=False)
            save_agent_message(
                mongo_session,
                user,
                'agent',
                step_content,
                'action',
                generated_by,
            )
        elif event_type == 'url':
            update_session_page_state(mongo_session, url=data.get('value'))
        elif event_type == 'screenshot':
            save_screenshot(mongo_session, user, data.get('data', ''), current_url_holder[0])
            update_session_page_state(mongo_session, screenshot_b64=data.get('data'))
        elif event_type == 'result':
            save_agent_message(mongo_session, user, 'agent', json.dumps(data.get('data', {})), 'result')
        elif event_type == 'ask_user':
            save_agent_message(mongo_session, user, 'agent', data.get('question', 'Question agent'), 'text')
        elif event_type == 'error':
            save_agent_message(mongo_session, user, 'agent', data.get('message', 'Erreur agent'), 'error')
        await queue.put(data)

    async def set_context(ctx: MCPContext):
        async with state.lock:
            state.context = ctx

    def context_callback(ctx: MCPContext):
        asyncio.create_task(set_context(ctx))

    async def agent_task():
        try:
            print(f"BACKEND: Starting agent with task: {body.task}")
            await run_agent(
                task=body.task,
                send_event=send_event,
                abort_event=state.abort_event,
                confirmation_event=state.confirm_event,
                context_callback=context_callback,
                stale_browser=body.stale_browser,
                skip_anti_bot=body.skip_anti_bot,
                feedback_queue=state.feedback_queue,
                user_reply_event=state.user_reply_event,
                show_browser=body.show_browser,
                agent_context=selected_agent.get('systemContext') if selected_agent else None,
                agent_name=selected_agent.get('name') if selected_agent else None,
                agent_description=selected_agent.get('description') if selected_agent else None,
                agent_id=str(selected_agent.get('id') or mongo_session.get('agentId')) if selected_agent else None,
                user_id=str(user['id']),
            )
            close_agent_session(mongo_session, 'completed')
            print("BACKEND: Agent completed successfully")
            elapsed_time = time.time() - start_time
            await queue.put({
                'type': 'execution_time',
                'elapsed_seconds': round(elapsed_time, 2),
                'elapsed_formatted': f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
            })
            await queue.put({'type': 'done'})
        except asyncio.CancelledError:
            print("BACKEND: Agent cancelled")
            close_agent_session(mongo_session, 'failed')
            await queue.put({'type': 'log', 'message': 'Agent canceled.'})
            await queue.put({'type': 'done'})
        except Exception as exc:
            print(f"BACKEND: Agent error: {exc}")
            close_agent_session(mongo_session, 'failed', {'error': str(exc)})
            await queue.put({'type': 'error', 'message': str(exc)})

    async with state.lock:
        state.context = None
        state.mongo_session = mongo_session
        state.mongo_user = user
        state.task = asyncio.create_task(agent_task())
        print("BACKEND: Agent task created")

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
                        print("BACKEND: Client disconnected")
                        break
                    
                    if not done:
                        print("BACKEND: Sending keepalive")
                        yield sse_format({'type': 'log', 'message': 'keepalive'})
                        continue
                    
                    if get_task in done:
                        event = get_task.result()
                        print(f"BACKEND: Sending event: {event['type']}")
                        yield sse_format(event)
                        if event.get('type') == 'error':
                            print("BACKEND: Stream ending on error")
                            break
                except asyncio.CancelledError:
                    print("BACKEND: Stream cancelled")
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
                state.context = None
                state.mongo_session = None
                state.mongo_user = None
                state.feedback_queue = []  # Reset feedback queue on cleanup
                print("BACKEND: Stream cleanup done")

    print("BACKEND: Returning event stream")
    async def stream_with_session():
        yield sse_format({'type': 'session_started', 'session_id': mongo_session['sessionId']})
        async for event in event_stream():
            yield event

    return StreamingResponse(stream_with_session(), media_type='text/event-stream')


@app.get('/session/restore')
async def restore_session(user=Depends(get_current_user)):
    """Return last session with messages, lastUrl, lastScreenshot for frontend restore."""
    session = get_latest_running_session(user)
    if not session:
        return {'session': None, 'messages': []}
    try:
        pub_session, messages = get_session_messages_for_user(user, session['sessionId'])
    except ValueError:
        return {'session': None, 'messages': [], 'lastScreenshot': None}
    last_screenshot = get_last_screenshot(session)
    return {'session': pub_session, 'messages': messages, 'lastScreenshot': last_screenshot}


@app.post('/reset')
async def reset(user=Depends(get_current_user)):
    """Archive the current conversation and start a clean slate."""
    async with state.lock:
        if state.task and not state.task.done():
            if state.abort_event:
                state.abort_event.set()
            state.task.cancel()
            state.task = None
        state.feedback_queue = []
        state.context = None
    archive_and_reset()
    return {'status': 'reset', 'message': 'Conversation archived. Ready for a new topic.'}


@app.post('/abort')
async def abort(user=Depends(get_current_user)):
    async with state.lock:
        if not state.task or state.task.done():
            raise HTTPException(status_code=404, detail='No active agent to abort')
        if state.abort_event and not state.abort_event.is_set():
            state.abort_event.set()
        state.task.cancel()
    return {'status': 'aborting'}

@app.post('/confirm')
async def confirm(user=Depends(get_current_user)):
    async with state.lock:
        if state.confirm_event and not state.confirm_event.is_set():
            state.confirm_event.set()
            return {'status': 'confirmed'}
    return {'status': 'no_pending_confirmation'}


@app.post('/feedback')
async def feedback(body: FeedbackRequest, user=Depends(get_current_user)):
    async with state.lock:
        if state.mongo_session and state.mongo_user and str(state.mongo_user['id']) == str(user['id']):
            try:
                session_exists = get_session_for_user(user, state.mongo_session.get('sessionId', ''))
                if session_exists:
                    save_agent_message(state.mongo_session, user, 'user', body.message)
            except Exception:
                pass
        state.feedback_queue.append(body.message)
        if state.user_reply_event:
            state.user_reply_event.set()
    return {'status': 'feedback_queued', 'message': body.message}


@app.post('/browser/{action}')
async def browser_action(action: str):
    valid = {'back', 'forward', 'reload'}
    if action not in valid:
        raise HTTPException(status_code=400, detail='Invalid browser action')
    async with state.lock:
        if not state.task or state.task.done():
            return {'status': 'no_active_agent'}
    return {'status': 'accepted', 'action': action}


# ========== WORKFLOWS ENDPOINTS ==========

from backend.workflow_manager import (
    get_user_workflows,
    get_workflow_by_id,
    deactivate_workflow,
    activate_workflow,
    delete_workflow,
)


@app.get('/workflows')
async def list_workflows(user=Depends(get_current_user)):
    """Liste tous les workflows de l'utilisateur"""
    workflows = get_user_workflows(user['id'])
    return {'workflows': workflows}


@app.get('/workflows/{workflow_id}')
async def get_workflow(workflow_id: str, user=Depends(get_current_user)):
    """Récupère les détails d'un workflow"""
    workflow = get_workflow_by_id(workflow_id, user['id'])
    if not workflow:
        raise HTTPException(status_code=404, detail='Workflow introuvable')
    return {'workflow': workflow}


@app.post('/workflows/{workflow_id}/activate')
async def activate_workflow_endpoint(workflow_id: str, user=Depends(get_current_user)):
    """Active un workflow"""
    activate_workflow(workflow_id, user['id'])
    return {'status': 'activated'}


@app.post('/workflows/{workflow_id}/deactivate')
async def deactivate_workflow_endpoint(workflow_id: str, user=Depends(get_current_user)):
    """Désactive un workflow"""
    deactivate_workflow(workflow_id, user['id'])
    return {'status': 'deactivated'}


@app.delete('/workflows/{workflow_id}')
async def delete_workflow_endpoint(workflow_id: str, user=Depends(get_current_user)):
    """Supprime un workflow"""
    delete_workflow(workflow_id, user['id'])
    return {'status': 'deleted'}


@app.delete('/sessions/{session_id}')
async def delete_session_endpoint(session_id: str, user=Depends(get_current_user)):
    """Supprime une session et tous ses messages"""
    from backend.auth_db import delete_session_completely
    success = delete_session_completely(session_id, user['id'])
    if not success:
        raise HTTPException(status_code=404, detail='Session introuvable')
    return {'status': 'deleted', 'session_id': session_id}
