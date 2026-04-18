import asyncio
import base64
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional
import io
from urllib.parse import quote_plus

if os.name == 'nt':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass

from dotenv import load_dotenv
from PIL import Image
from playwright.sync_api import sync_playwright
from pydantic import BaseModel
from google import genai

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

LANGCHAIN_GOOGLE_AVAILABLE = False
ChatGoogleGenerativeAI = None
HumanMessage = None
SystemMessage = None
PydanticOutputParser = None
PromptTemplate = None
Tool = None
AgentExecutor = None
initialize_agent = None
AgentType = None
try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers.pydantic import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain.agents import Tool, AgentExecutor, initialize_agent
    from langchain.agents.agent_types import AgentType
    LANGCHAIN_GOOGLE_AVAILABLE = True
except Exception:
    pass

from backend.mcp import MCPContext
from backend.nlp import analyze_task
from backend.rpa import RPAController

PROVIDER = os.getenv('PROVIDER', 'gemini').lower()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PLAYWRIGHT_STALE = os.getenv('PLAYWRIGHT_STALE', 'false').lower() in ('1', 'true', 'yes')
PLAYWRIGHT_SKIP_ANTI_BOT = os.getenv('PLAYWRIGHT_SKIP_ANTI_BOT', 'false').lower() in ('1', 'true', 'yes')
PLAYWRIGHT_USER_DATA_DIR = os.getenv('PLAYWRIGHT_USER_DATA_DIR', os.path.join(os.path.dirname(__file__), '.playwright_profile'))
PLAYWRIGHT_USER_AGENT = os.getenv(
    'PLAYWRIGHT_USER_AGENT',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
)
PLAYWRIGHT_EXTRA_ARGS = [arg.strip() for arg in os.getenv('PLAYWRIGHT_EXTRA_ARGS', '').split(',') if arg.strip()]

if PROVIDER == 'gemini' and (not GEMINI_API_KEY or GEMINI_API_KEY == 'your_gemini_api_key'):
    raise RuntimeError(
        'GEMINI_API_KEY is not set. Create backend/.env with GEMINI_API_KEY=your_real_key'
    )

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

DEFAULT_GEMINI_MODELS = [
    'models/gemini-2.0-flash',
]

def _normalize_gemini_model_name(model_name: str) -> str:
    if model_name.startswith('publishers/google/models/'):
        parts = model_name.split('/')
        return 'models/' + '/'.join(parts[3:])
    return model_name


def _get_available_gemini_models(max_models: int = 50) -> List[str]:
    if not client:
        return []
    try:
        response = client.models.list(config={'page_size': max_models})
        available = []
        for model in response:
            name = getattr(model, 'name', None) or getattr(model, 'display_name', None)
            if not name:
                continue
            normalized = _normalize_gemini_model_name(name)
            if normalized.startswith('models/gemini-'):
                available.append(normalized)
        return available
    except Exception:
        return []


def _create_gemini_model(model_name: str):
    return client.models


def _create_langchain_gemini_model():
    return ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        api_key=GEMINI_API_KEY,
        temperature=0.15,
        max_tokens=1024,
        streaming=False,
    )


def _create_langchain_agent(llm: Any, tools: List[Any]) -> Optional[Any]:
    if not initialize_agent or not AgentType:
        return None
    try:
        return initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            max_iterations=1,
        )
    except Exception:
        return None


def _build_langchain_tools(rpa: RPAController, loop: asyncio.AbstractEventLoop, send_event) -> List[Any]:
    def navigate_tool(url: str) -> str:
        action = {'action': 'navigate', 'url': url}
        result = rpa.navigate(url)
        return json.dumps(action)

    def click_tool(selector: str) -> str:
        action = {'action': 'click', 'selector': selector}
        result = rpa.click(selector)
        return json.dumps(action)

    def type_tool(selector: str, text: str) -> str:
        action = {'action': 'type', 'selector': selector, 'text': text}
        result = rpa.type_text(selector, text)
        return json.dumps(action)

    def scroll_tool(direction: str = 'down') -> str:
        action = {'action': 'scroll', 'direction': direction}
        result = rpa.scroll(600 if direction == 'down' else -600)
        return json.dumps(action)

    def extract_tool() -> str:
        page_text = rpa.extract_text(max_chars=3000)
        action = {'action': 'extract', 'data': {'text': page_text}}
        return json.dumps(action)

    def screenshot_tool() -> str:
        screenshot = rpa.take_screenshot()
        action = {'action': 'screenshot', 'data': 'captured'}
        return json.dumps(action)

    return [
        Tool(name='navigate', func=navigate_tool, description='Navigate the browser to a URL.', return_direct=True),
        Tool(name='click', func=click_tool, description='Click a page element by selector or visible text.', return_direct=True),
        Tool(name='type', func=type_tool, description='Type text into an input field by selector.', return_direct=True),
        Tool(name='scroll', func=scroll_tool, description='Scroll the page vertically.', return_direct=True),
        Tool(name='extract', func=extract_tool, description='Extract visible page text as structured data.', return_direct=True),
        Tool(name='screenshot', func=screenshot_tool, description='Capture a screenshot and send it to the UI.', return_direct=True),
    ]


def _format_react_prompt(task: str, page_url: str, page_text: str, iteration: int, context: Dict[str, Any]) -> str:
    summary_json = json.dumps(context, ensure_ascii=False)
    return f"""You are GSAM, a precise web navigation AI agent.
Your current task: {task}
Iteration: {iteration}/10
Current URL: {page_url}

Current page text preview:
{page_text or '(empty page)'}

Persistent context:
{summary_json}

You have the following tools available:
- navigate(url)
- click(selector)
- type(selector, text)
- scroll(direction)
- extract()
- screenshot()

Choose exactly one tool call to make next. Return only the tool invocation in JSON format or a single JSON object with action and parameters. Do not add any explanation or markdown.
"""


def _parse_agent_action(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    raw = re.sub(r'```(?:json)?\s*', '', str(raw)).strip().rstrip('`').strip()
    if ACTION_PARSER:
        try:
            return ACTION_PARSER.parse(raw).dict()
        except Exception:
            pass
    parsed = _parse_json(raw)
    if not parsed:
        return {}
    try:
        return BrowserAction.parse_obj(parsed).dict()
    except Exception:
        return parsed


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
            return texts.join(' ').replace(/\\s+/g, ' ');
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


class BrowserAction(BaseModel):
    action: str
    url: Optional[str] = None
    selector: Optional[str] = None
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None


ACTION_PARSER = None
ACTION_PROMPT = None
START_URL_PROMPT = None
if PydanticOutputParser and PromptTemplate:
    ACTION_PARSER = PydanticOutputParser(pydantic_object=BrowserAction)
    ACTION_PROMPT = PromptTemplate.from_template(
        """You are GSAM, a precise web navigation AI agent.
Your current task: {task}
Iteration: {iteration} of 10
Current URL: {page_url}

Page content preview:
{page_text}

{format_instructions}

Respond now with only the JSON object matching the schema exactly."""
    )
    START_URL_PROMPT = PromptTemplate.from_template(
        """Given this web task: \"{task}\"
What is the best URL to start at? Reply with ONLY the URL, nothing else.
Examples: https://www.google.com/search?q=... or https://www.amazon.com
If unsure, reply: https://www.google.com"""
    )


def _normalize_target_url(task: str) -> str:
    value = task.strip()
    if not value:
        return 'https://www.google.com'
    if value.startswith('http://') or value.startswith('https://'):
        return value
    if re.match(r'^[\w-]+(\.[\w-]+)+(\:\d+)?(/.*)?$', value):
        return f'https://{value}'
    return 'https://www.google.com'


def _create_google_search_url(task: str) -> str:
    query = task.strip()
    if not query:
        return 'https://www.google.com'
    query = quote_plus(query)
    return f'https://www.google.com/search?q={query}'


def _build_playwright_args(skip_anti_bot: bool) -> List[str]:
    args = ['--no-sandbox']
    if skip_anti_bot:
        args.extend([
            '--disable-blink-features=AutomationControlled',
            '--disable-features=IsolateOrigins,site-per-process',
            '--disable-infobars',
            '--disable-dev-shm-usage',
            '--no-first-run',
            '--no-default-browser-check',
        ])
    args.extend(PLAYWRIGHT_EXTRA_ARGS)
    return args


def _create_playwright_browser_page(playwright, stale: bool, skip_anti_bot: bool):
    launch_args = _build_playwright_args(skip_anti_bot)
    if stale:
        browser = playwright.chromium.launch_persistent_context(
            user_data_dir=PLAYWRIGHT_USER_DATA_DIR,
            headless=True,
            args=launch_args,
            viewport={'width': 1280, 'height': 800},
            user_agent=PLAYWRIGHT_USER_AGENT if skip_anti_bot else None,
            locale='en-US',
            ignore_https_errors=skip_anti_bot,
        )
        page = browser.pages[0] if browser.pages else browser.new_page()
        return browser, page

    browser = playwright.chromium.launch(headless=True, args=launch_args)
    context = browser.new_context(
        viewport={'width': 1280, 'height': 800},
        user_agent=PLAYWRIGHT_USER_AGENT if skip_anti_bot else None,
        locale='en-US',
        extra_http_headers={'accept-language': 'en-US,en;q=0.9'} if skip_anti_bot else None,
        ignore_https_errors=skip_anti_bot,
    )
    page = context.new_page()
    return browser, page


def _task_requires_skip_anti_bot(task: str) -> bool:
    query = task.lower()
    keywords = [
        'anti-bot',
        'anti bot',
        'anti-scraping',
        'anti scraping',
        'captcha',
        'bot detection',
        'bot-detection',
        'cloudflare',
        'contourner bot',
        'contourner anti bot',
        'contourner anti-bot',
        'contourner captcha',
        'bypass bot',
        'bypass captcha',
        'scraping protection',
        'protection anti-bot',
        'anti bot protection',
        'anti bot challenge',
    ]
    return any(keyword in query for keyword in keywords)


def _apply_anti_bot_page_settings(page) -> None:
    try:
        page.add_init_script(
            '''() => {
                try {
                    Object.defineProperty(navigator, 'webdriver', {get: () => false, configurable: true});
                    window.chrome = { runtime: {} };
                    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => {
                        if (parameters.name === 'notifications') {
                            return Promise.resolve({ state: Notification.permission });
                        }
                        return originalQuery(parameters);
                    };
                } catch (e) {
                    // ignore anti-bot shim failures
                }
            }'''
        )
    except Exception:
        pass


def _format_action_prompt(task: str, page_url: str, page_text: str, iteration: int) -> str:
    return f"""You are GSAM, a precise web navigation AI agent controlling a real browser.

TASK: {task}
ITERATION: {iteration}/10
CURRENT URL: {page_url}

PAGE TEXT (first 2000 chars):
{page_text or '(empty page)'}

STRICT RULES:
- If you are on google.com and have NOT searched yet: type in the search box
  selector for Google search input: input[name='q']
- After typing in Google, ALWAYS follow with a navigate action using the full search URL
  Example: {{"action":"navigate","url":"https://www.google.com/search?q=your+query"}}
- If you see search results: click on the most relevant result link
- If you are on the target page and have found the answer: extract the data
- If the task is fully complete: use done action with a clear summary
- NEVER repeat the same action twice in a row
- NEVER stay on google.com for more than 2 iterations without navigating away
- NEVER return empty or null — always return a valid JSON action

AVAILABLE ACTIONS — respond with ONLY one raw JSON object, no markdown, no explanation:
{{"action":"navigate","url":"https://..."}}
{{"action":"type","selector":"CSS_SELECTOR","text":"text to type"}}
{{"action":"click","selector":"CSS_SELECTOR_OR_TEXT"}}
{{"action":"scroll","direction":"down"}}
{{"action":"extract","data":{{"field":"value found"}}}}
{{"action":"done","summary":"what was accomplished"}}

RESPOND NOW WITH ONLY THE JSON:"""


def _send_event_sync(loop: asyncio.AbstractEventLoop, send_event, data: dict) -> None:
    try:
        future = asyncio.run_coroutine_threadsafe(send_event(data), loop)
        future.result(timeout=10)
    except Exception:
        pass


def _ask_gemini_sync(screenshot_base64: str, prompt: str) -> Dict[str, Any]:
    raw = None
    last_error = None
    quota_keywords = ['429', 'quota', 'rate-limit', 'rate limit', 'free_tier']

    # --- LangChain path (primary) with multimodal image support ---
    if LANGCHAIN_GOOGLE_AVAILABLE and ChatGoogleGenerativeAI and HumanMessage:
        try:
            image_bytes = base64.b64decode(screenshot_base64)
            image = Image.open(io.BytesIO(image_bytes))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            image_b64_jpeg = base64.b64encode(buffer.getvalue()).decode('utf-8')

            model = ChatGoogleGenerativeAI(
                model='gemini-2.0-flash',
                api_key=GEMINI_API_KEY,
                temperature=0.1,
                max_tokens=512,
            )
            message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64_jpeg}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ])
            result = model.invoke([message])
            raw = result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if any(k in err_str for k in quota_keywords):
                return {'error': str(e), 'quota_exceeded': True}

    # --- Native Gemini fallback (if LangChain failed) ---
    if not raw:
        image_bytes = base64.b64decode(screenshot_base64)
        image = Image.open(io.BytesIO(image_bytes))
        models_to_try = _get_available_gemini_models() or DEFAULT_GEMINI_MODELS
        if models_to_try != DEFAULT_GEMINI_MODELS:
            models_to_try = [m for m in models_to_try if m.startswith('models/gemini-')]
            if not models_to_try:
                models_to_try = DEFAULT_GEMINI_MODELS
        for model_name in models_to_try:
            try:
                current_model = _create_gemini_model(model_name)
                response = current_model.generate_content(
                    model=model_name,
                    contents=[prompt, image]
                )
                raw = response.text if hasattr(response, 'text') else str(response)
                if raw:
                    break
            except Exception as e:
                err_str = str(e).lower()
                last_error = e
                if any(k in err_str for k in quota_keywords):
                    return {'error': str(e), 'quota_exceeded': True}
                if 'not found' in err_str or '404' in err_str:
                    continue
                continue

    if not raw:
        return {'error': f'All models failed. Last error: {last_error}'}

    raw = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()
    if ACTION_PARSER:
        try:
            return ACTION_PARSER.parse(raw).dict()
        except Exception:
            pass
    result = _parse_json(raw)
    if not result:
        return {'error': f'Invalid JSON from model: {raw}'}
    try:
        return BrowserAction.parse_obj(result).dict()
    except Exception:
        return result


def _ask_vision_claude(screenshot_base64: str, prompt: str) -> Dict[str, Any]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=512,
            messages=[{
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
            }]
        )
        raw = message.content[0].text
        raw = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()
        if ACTION_PARSER:
            try:
                return ACTION_PARSER.parse(raw).dict()
            except Exception:
                pass
        parsed = _parse_json(raw)
        return parsed or {'error': f'Invalid JSON: {raw}'}
    except Exception as e:
        return {'error': str(e)}


def _ask_vision_groq(screenshot_base64: str, prompt: str) -> Dict[str, Any]:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            max_tokens=512,
            temperature=0.1,
        )
        raw = response.choices[0].message.content
        raw = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()
        if ACTION_PARSER:
            try:
                return ACTION_PARSER.parse(raw).dict()
            except Exception:
                pass
        parsed = _parse_json(raw)
        return parsed or {'error': f'Invalid JSON: {raw}'}
    except Exception as e:
        return {'error': str(e)}


def _run_playwright_sync(
    task: str,
    loop: asyncio.AbstractEventLoop,
    send_event,
    mcp_context: MCPContext,
    abort_event: Optional[asyncio.Event] = None,
    confirmation_event: Optional[asyncio.Event] = None,
    start_url: str = 'https://www.google.com',
    stale_browser: bool = False,
    skip_anti_bot: bool = False,
) -> None:
    def _run_legacy_loop(page):
        # Fallback behavior using the existing native model loop when LangChain is unavailable.
        url = page.url
        screenshot_bytes = page.screenshot(full_page=False)
        screenshot_base64 = _to_base64_png(screenshot_bytes)
        _send_event_sync(loop, send_event, {'type': 'screenshot', 'data': screenshot_base64})
        _send_event_sync(loop, send_event, {'type': 'url', 'value': url})

        def _is_google_search_page(url: str) -> bool:
            return 'google.com/search' in url or 'bing.com/search' in url

        def _click_first_search_result(page) -> bool:
            try:
                if page.locator('a h3').count() > 0:
                    page.locator('a h3').first.click(timeout=8000)
                    page.wait_for_load_state('domcontentloaded', timeout=10000)
                    return True
            except Exception:
                pass
            try:
                if page.locator('div#search a h3').count() > 0:
                    page.locator('div#search a h3').first.click(timeout=8000)
                    page.wait_for_load_state('domcontentloaded', timeout=10000)
                    return True
            except Exception:
                pass
            return False

        def _force_google_query(task: str, page) -> bool:
            try:
                if 'google.com' in page.url and 'search' not in page.url:
                    page.goto(_create_google_search_url(task), timeout=20000)
                    page.wait_for_load_state('domcontentloaded', timeout=10000)
                    return True
            except Exception:
                pass
            return False

        for iteration in range(10):
            _send_event_sync(loop, send_event, {
                'type': 'iteration',
                'current': iteration + 1,
                'total': 10
            })

            if abort_event and abort_event.is_set():
                _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Abort requested during execution.'})
                return

            if _is_google_search_page(page.url):
                if _click_first_search_result(page):
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Auto-clicked first search result.'})
                    _send_event_sync(loop, send_event, {'type': 'url', 'value': page.url})
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': 'NAVIGATE', 'args': 'first search result', 'status': 'done'})
                    continue

            if _force_google_query(task, page):
                _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Auto-navigated to Google search results.'})
                _send_event_sync(loop, send_event, {'type': 'url', 'value': page.url})
                _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SEARCH', 'args': page.url, 'status': 'done'})
                continue

            screenshot_bytes = page.screenshot(full_page=False)
            screenshot_base64 = _to_base64_png(screenshot_bytes)
            page_text = _get_page_text(page, max_chars=2000)
            _send_event_sync(loop, send_event, {'type': 'screenshot', 'data': screenshot_base64})

            if abort_event and abort_event.is_set():
                _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Abort detected after screenshot.'})
                return

            _send_event_sync(loop, send_event, {
                'type': 'thinking',
                'message': f'Analyzing screenshot and page state — iteration {iteration + 1}/10.'
            })

            system_prompt = _format_action_prompt(
                task=task,
                page_url=page.url,
                page_text=page_text,
                iteration=iteration + 1
            )

            if PROVIDER == 'claude' and ANTHROPIC_API_KEY:
                action = _ask_vision_claude(screenshot_base64, system_prompt)
            elif PROVIDER == 'groq' and GROQ_API_KEY:
                action = _ask_vision_groq(screenshot_base64, system_prompt)
            else:
                action = _ask_gemini_sync(screenshot_base64, system_prompt)

            if action and action.get('quota_exceeded'):
                _send_event_sync(loop, send_event, {
                    'type': 'error',
                    'message': 'Quota exceeded or rate limit reached. Check provider keys and plan.'
                })
                return
            if action and 'error' in action:
                _send_event_sync(loop, send_event, {
                    'type': 'log',
                    'message': f'Vision model error: {action["error"]}'
                })
            if not action or 'action' not in action:
                retry_prompt = system_prompt + "\n\nIMPORTANT: You MUST respond with ONLY a raw JSON object. No markdown, no explanation."
                if PROVIDER == 'claude' and ANTHROPIC_API_KEY:
                    action = _ask_vision_claude(screenshot_base64, retry_prompt)
                elif PROVIDER == 'groq' and GROQ_API_KEY:
                    action = _ask_vision_groq(screenshot_base64, retry_prompt)
                else:
                    action = _ask_gemini_sync(screenshot_base64, retry_prompt)
                if action and action.get('quota_exceeded'):
                    _send_event_sync(loop, send_event, {
                        'type': 'error',
                        'message': 'Quota exceeded or rate limit reached. Check provider keys and plan.'
                    })
                    return
                if action and 'error' in action:
                    _send_event_sync(loop, send_event, {
                        'type': 'log',
                        'message': f'Vision model error (retry): {action["error"]}'
                    })
            if not action or 'action' not in action:
                _send_event_sync(loop, send_event, {
                    'type': 'log',
                    'message': f'Iteration {iteration+1}: Gemini returned no valid action, skipping.'
                })
                time.sleep(1)
                continue

            name = action.get('action')
            name_upper = name.upper() if name else 'UNKNOWN'
            args = ''
            if name == 'navigate':
                args = action.get('url', '')
            elif name == 'click':
                args = action.get('selector', '')
            elif name == 'type':
                args = f"{action.get('selector', '')} | {action.get('text', '')}"
            elif name == 'scroll':
                args = action.get('direction', '')
            elif name == 'extract':
                args = json.dumps(action.get('data', {}))
            elif name == 'done':
                args = action.get('summary', '')

            action_text = (name + ' ' + args).lower() if name else args.lower()
            needs_confirm = any(word in action_text for word in {'submit', 'purchase', 'delete', 'send', 'confirm', 'pay'})
            if needs_confirm:
                _send_event_sync(loop, send_event, {
                    'type': 'safety',
                    'explanation': f'The agent is about to: {name_upper} — {args}. '
                                   f'This action may be irreversible. Confirm to proceed.'
                })
                if confirmation_event:
                    confirmation_event.clear()
                    deadline = time.time() + 30
                    while time.time() < deadline:
                        if abort_event and abort_event.is_set():
                            _send_event_sync(loop, send_event, {
                                'type': 'log', 'message': 'Aborted during safety wait.'
                            })
                            return
                        if confirmation_event.is_set():
                            _send_event_sync(loop, send_event, {
                                'type': 'log', 'message': 'Safety confirmed by operator.'
                            })
                            break
                        time.sleep(0.5)
                    else:
                        _send_event_sync(loop, send_event, {
                            'type': 'log',
                            'message': f'Safety timeout — skipping action: {name_upper}'
                        })
                        continue
                else:
                    time.sleep(2)

            _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'running'})

            try:
                if abort_event and abort_event.is_set():
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Abort requested before action.'})
                    return
                if name == 'navigate':
                    try:
                        page.goto(action.get('url', ''), timeout=20000)
                        page.wait_for_load_state('domcontentloaded', timeout=10000)
                    except Exception as nav_err:
                        _send_event_sync(loop, send_event, {
                            'type': 'log',
                            'message': f'Navigation timeout: {nav_err}'
                        })
                elif name == 'click':
                    selector = action.get('selector', '')
                    try:
                        page.click(selector, timeout=5000)
                    except Exception:
                        try:
                            page.get_by_text(selector, exact=False).first.click(timeout=5000)
                        except Exception:
                            try:
                                page.get_by_role('button', name=selector).click(timeout=5000)
                            except Exception as fallback_err:
                                _send_event_sync(loop, send_event, {
                                    'type': 'log',
                                    'message': f'Click failed on "{selector}": {fallback_err}'
                                })
                elif name == 'type':
                    selector = action.get('selector', '')
                    text = action.get('text') or ''
                    try:
                        page.click(selector, timeout=5000)
                        page.fill(selector, text)
                        page.keyboard.press('Enter')
                        page.wait_for_load_state('domcontentloaded', timeout=8000)
                    except Exception:
                        typed = False
                        try:
                            page.get_by_placeholder(selector).fill(text)
                            typed = True
                        except Exception:
                            pass
                        if not typed:
                            try:
                                page.locator('[name="q"]:visible').first.fill(text)
                                typed = True
                            except Exception:
                                pass
                        if not typed:
                            try:
                                page.locator('input[type="search"]:visible').first.fill(text)
                                typed = True
                            except Exception:
                                pass
                        if not typed:
                            try:
                                page.locator('input[type="text"]:visible').first.fill(text)
                                typed = True
                            except Exception:
                                pass
                        if not typed:
                            try:
                                page.locator('textarea:visible').first.fill(text)
                                typed = True
                            except Exception as type_err:
                                _send_event_sync(loop, send_event, {
                                    'type': 'log',
                                    'message': f'Type failed on "{selector}": {type_err}'
                                })
                        if typed:
                            try:
                                page.keyboard.press('Enter')
                                page.wait_for_load_state('domcontentloaded', timeout=8000)
                            except Exception:
                                pass
                elif name == 'scroll':
                    page.evaluate('window.scrollBy(0, 600)')
                elif name == 'extract':
                    _send_event_sync(loop, send_event, {'type': 'result', 'data': action.get('data', {})})
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'done'})
                    return
                elif name == 'done':
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'done'})
                    return
                else:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': f'Unknown action: {name}'})
            except Exception as exc:
                _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'error'})
                _send_event_sync(loop, send_event, {'type': 'error', 'message': str(exc)})
                return

            _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'done'})
            time.sleep(4)

        title = page.title()
        _send_event_sync(loop, send_event, {'type': 'result', 'data': {'page_title': title}})

    with sync_playwright() as playwright:
        browser, page = _create_playwright_browser_page(
            playwright,
            stale=stale_browser or PLAYWRIGHT_STALE,
            skip_anti_bot=skip_anti_bot or PLAYWRIGHT_SKIP_ANTI_BOT,
        )
        if skip_anti_bot or PLAYWRIGHT_SKIP_ANTI_BOT:
            _apply_anti_bot_page_settings(page)

        try:
            try:
                page.goto(start_url, timeout=20000)
                page.wait_for_load_state('domcontentloaded', timeout=10000)
            except Exception as nav_err:
                _send_event_sync(loop, send_event, {
                    'type': 'log',
                    'message': f'Navigation timeout: {nav_err}'
                })
            url = page.url
            mcp_context.update_state(current_url=url, iteration=0, status='starting')
            _send_event_sync(loop, send_event, {'type': 'url', 'value': url})
            _send_event_sync(loop, send_event, {'type': 'log', 'message': f'Starting agent with LangChain: {LANGCHAIN_GOOGLE_AVAILABLE}'})

            if abort_event and abort_event.is_set():
                _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Abort requested before execution.'})
                return

            if LANGCHAIN_GOOGLE_AVAILABLE and ChatGoogleGenerativeAI and Tool and AgentType:
                rpa = RPAController(page, mcp_context, loop, send_event)
                llm = _create_langchain_gemini_model()
                tools = _build_langchain_tools(rpa, loop, send_event)
                agent_executor = _create_langchain_agent(llm, tools)

                if agent_executor is None:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': 'LangChain agent initialization failed, using fallback.'})
                    _run_legacy_loop(page)
                    return

                for iteration in range(10):
                    mcp_context.update_state(current_url=page.url, iteration=iteration + 1, status='running')
                    _send_event_sync(loop, send_event, {
                        'type': 'iteration',
                        'current': iteration + 1,
                        'total': 10
                    })

                    if abort_event and abort_event.is_set():
                        _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Abort requested during execution.'})
                        return

                    _send_event_sync(loop, send_event, {
                        'type': 'thinking',
                        'message': f'Analyzing browser state — iteration {iteration + 1}/10.'
                    })

                    page_text = rpa.get_page_text(max_chars=2000)
                    screenshot_base64 = rpa.take_screenshot()

                    prompt = _format_react_prompt(
                        task=task,
                        page_url=page.url,
                        page_text=page_text,
                        iteration=iteration + 1,
                        context=mcp_context.get_context_summary(),
                    )

                    try:
                        raw_response = agent_executor.invoke(input=prompt)
                        raw = raw_response.output if hasattr(raw_response, 'output') else str(raw_response)
                    except Exception as exc:
                        _send_event_sync(loop, send_event, {'type': 'error', 'message': f'LangChain invocation failed: {exc}'})
                        _run_legacy_loop(page)
                        return

                    action = _parse_agent_action(raw)
                    if not action or 'action' not in action:
                        _send_event_sync(loop, send_event, {'type': 'log', 'message': 'LangChain returned no valid action, retrying with fallback model.'})
                        _run_legacy_loop(page)
                        return

                    if action.get('action') == 'extract':
                        _send_event_sync(loop, send_event, {'type': 'result', 'data': action.get('data', {})})
                        _send_event_sync(loop, send_event, {'type': 'step', 'name': 'EXTRACT', 'args': json.dumps(action.get('data', {})), 'status': 'done'})
                        return
                    if action.get('action') == 'done':
                        _send_event_sync(loop, send_event, {'type': 'step', 'name': 'DONE', 'args': action.get('summary', ''), 'status': 'done'})
                        return

                    name = action.get('action')
                    args = ''
                    if name == 'navigate':
                        args = action.get('url', '')
                    elif name == 'click':
                        args = action.get('selector', '')
                    elif name == 'type':
                        args = f"{action.get('selector', '')} | {action.get('text', '')}"
                    elif name == 'scroll':
                        args = action.get('direction', '')
                    elif name == 'extract':
                        args = json.dumps(action.get('data', {}))

                    _send_event_sync(loop, send_event, {'type': 'step', 'name': name.upper() if name else 'UNKNOWN', 'args': args, 'status': 'running'})
                    mcp_context.add_action(name or 'unknown', {'args': args})

                    time.sleep(4)

                title = page.title()
                _send_event_sync(loop, send_event, {'type': 'result', 'data': {'page_title': title}})
            else:
                _send_event_sync(loop, send_event, {'type': 'log', 'message': 'LangChain not available, using native fallback.'})
                _run_legacy_loop(page)
        except Exception as exc:
            _send_event_sync(loop, send_event, {'type': 'error', 'message': str(exc)})
        finally:
            browser.close()


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
        search_intent = re.search(
            r'\b(search|find|look for|lookup|look up|what is|who is|how to|where is|details|information|recherche|cherche|trouver|scan|search for|find the|list the|find all|discover|crawl|audit)\b',
            task,
            re.IGNORECASE
        )
        direct_navigation = re.search(
            r'\b(go to|visit|open|navigate to|browse to|access|launch|open website|visit site|check the site)\b',
            task,
            re.IGNORECASE
        )
        if direct_navigation and not search_intent:
            return f'https://{domain}'
        return ''

    return ''


async def _get_start_url(task: str) -> str:
    extracted_url = _extract_start_url(task)
    if extracted_url:
        return extracted_url

    # Automatic search start: use Google Search for query-like tasks.
    return _create_google_search_url(task)


async def run_agent(
    task: str,
    send_event,
    abort_event: Optional[asyncio.Event] = None,
    confirmation_event: Optional[asyncio.Event] = None,
    stale_browser: bool = False,
    skip_anti_bot: bool = False,
    context_callback: Optional[Callable[[MCPContext], None]] = None,
) -> MCPContext:
    nlp_result = await analyze_task(task)
    intent = nlp_result.get('intent', 'DEEP_SWEEP')
    entity = nlp_result.get('entity', 'TARGET')
    subtasks = nlp_result.get('subtasks', ['Analyze', 'Execute', 'Report'])

    mcp_context = MCPContext(task=task, intent=intent, entity=entity, subtasks=subtasks)
    if context_callback:
        context_callback(mcp_context)
    await send_event({'type': 'nlp', 'intent': intent, 'entity': entity, 'subtasks': subtasks})
    await send_event({'type': 'context', 'context': mcp_context.get_context_summary()})
    if abort_event and abort_event.is_set():
        await send_event({'type': 'log', 'message': 'Abort requested before starting.'})
        await send_event({'type': 'done'})
        return

    start_url = await _get_start_url(task)
    auto_skip = _task_requires_skip_anti_bot(task)
    if auto_skip and not skip_anti_bot:
        skip_anti_bot = True
        await send_event({'type': 'log', 'message': 'Anti-bot mode enabled automatically based on task text.'})

    await send_event({'type': 'log', 'message': f'Starting at: {start_url}'})

    loop = asyncio.get_running_loop()
    try:
        await asyncio.to_thread(
            _run_playwright_sync,
            task,
            loop,
            send_event,
            mcp_context,
            abort_event,
            confirmation_event,
            start_url,
            stale_browser,
            skip_anti_bot,
        )
    except Exception as exc:
        await send_event({'type': 'error', 'message': str(exc)})
    finally:
        await send_event({'type': 'done'})
    return mcp_context
