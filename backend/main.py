import asyncio
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
from backend.mcp import MCPContext

app = FastAPI(title='GSAM Agent Backend')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:8000', 'http://127.0.0.1:8000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class RunRequest(BaseModel):
    task: str
    stale_browser: bool = False
    skip_anti_bot: bool = False

class AgentState:
    task: Optional[asyncio.Task] = None
    abort_event: Optional[asyncio.Event] = None
    confirm_event: Optional[asyncio.Event] = None
    context: Optional[MCPContext] = None
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


@app.get('/context')
async def get_context():
    async with state.lock:
        if state.context:
            return {'context': state.context.get_context_summary()}
        return {'context': None}


@app.post('/run')
async def run(request: Request, body: RunRequest):
    print(f"BACKEND: Received /run request with task: {body.task}")
    async with state.lock:
        if state.task and not state.task.done():
            print("BACKEND: Agent already running, rejecting request")
            raise HTTPException(status_code=409, detail='Agent is already running')
        state.abort_event = asyncio.Event()
        state.confirm_event = asyncio.Event()

    queue: asyncio.Queue = asyncio.Queue()

    async def send_event(data: dict):
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
            )
            print("BACKEND: Agent completed successfully")
        except asyncio.CancelledError:
            print("BACKEND: Agent cancelled")
            await queue.put({'type': 'log', 'message': 'Agent canceled.'})
            await queue.put({'type': 'done'})
        except Exception as exc:
            print(f"BACKEND: Agent error: {exc}")
            await queue.put({'type': 'error', 'message': str(exc)})

    async with state.lock:
        state.context = None
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
                        if event.get('type') in {'done', 'error'}:
                            print("BACKEND: Stream ending")
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
                print("BACKEND: Stream cleanup done")

    print("BACKEND: Returning event stream")
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
