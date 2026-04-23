import asyncio
import base64
import json
import os
import random
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
    from langchain_core.tools import Tool
    LANGCHAIN_GOOGLE_AVAILABLE = True
except ImportError as _lc_import_err:
    import logging as _logging
    _logging.getLogger(__name__).warning('LangChain not available, using native fallback: %s', _lc_import_err)

AgentExecutor = None
initialize_agent = None
AgentType = None

from backend.mcp import MCPContext
from backend.nlp import analyze_task
from backend.rpa import RPAController
from backend.memory import append_conversation, get_conversation_history, get_memory_context, save_session, archive_and_reset

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
    'models/gemini-2.5-flash',
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
    except OSError as _models_err:
        import logging as _logging
        _logging.getLogger(__name__).warning('Failed to list Gemini models: %s', _models_err)
        return []


def _create_gemini_model(model_name: str):
    return client.models


def _create_langchain_gemini_model():
    return ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
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


def _format_react_prompt(task: str, page_url: str, page_text: str, iteration: int, context: Dict[str, Any], memory_context: str = '') -> str:
    summary_json = json.dumps(context, ensure_ascii=False)

    conversation_memory = context.get('conversation_memory', [])
    recent_history = ""
    if conversation_memory:
        recent = conversation_memory[-20:]
        recent_history = "\n".join([f"- {item.get('type', 'unknown')}: {item.get('message', item.get('question', 'N/A'))}" for item in recent])

    user_correction = ""
    if conversation_memory:
        corrections = [item for item in conversation_memory if item.get('type') == 'user_feedback']
        if corrections:
            user_correction = f"⚠️ {corrections[-1].get('message', '')}"

    memory_section = f"{memory_context}\n\n" if memory_context else ""

    return f"""# GSAM — Web Navigation Agent
{memory_section}## TASK: {task}
## CURRENT STATE: iteration {iteration}, url: {page_url}, page content: {page_text[:1000]}...
## RECENT ACTIONS: last 3 from MCP history
## CONVERSATION HISTORY: {recent_history or 'None'}
## USER CORRECTION (if any): {user_correction}
## AVAILABLE TOOLS: navigate, click, type, fill_form, scroll, extract, ask_user, done

You have the following tools available:
- navigate(url)
- click(selector)
- type(selector, text)
- scroll(direction)
- extract()
- screenshot()

RULES:
- Review the CONVERSATION HISTORY to find relations between the current task and previous questions/searches. Continue related research accordingly.
- If there are unfinished tasks or follow-ups from previous conversations, prioritize completing them.
- Do not click on select elements or dropdowns unless specifically needed.
- If you need user input for forms, use ask_user.
- Avoid repetitive actions.
- RELEVANCE CHECK BEFORE done: Before using the done action, verify that the extracted or found data actually answers the user's task. If the page content is empty, off-topic, or does not match the task intent, do NOT use done — use ask_user to explain what was found and ask for clarification.
- EMPTY OR IRRELEVANT RESULTS: If after a search or data extraction the results are empty, unrelated to the task, or clearly wrong, do NOT invent a continuation or retry the same action. Immediately use ask_user to explain what happened and ask the user how to proceed.
- If the task is to read or extract information, use extract.

Choose exactly one tool call to make next. Return only the tool invocation in JSON format or a single JSON object with action and parameters. Do not add any explanation or markdown.

Persistent context:
{summary_json}
"""


def _parse_agent_action(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    raw = re.sub(r'```(?:json)?\s*', '', str(raw)).strip().rstrip('`').strip()
    if ACTION_PARSER:
        try:
            return ACTION_PARSER.parse(raw).dict()
        except ValueError:
            pass
    parsed = _parse_json(raw)
    if not parsed:
        return {}
    try:
        return BrowserAction.parse_obj(parsed).dict()
    except ValueError:
        return parsed


def _to_base64_png(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _human_type(page, selector: str, text: str) -> None:
    """Type text character by character with human-like delays."""
    try:
        element = page.locator(selector).first
        element.scroll_into_view_if_needed()
        element.click()
        element.fill('')  # Clear the field
        for char in text:
            page.keyboard.type(char)
            time.sleep(random.randint(40, 120) / 1000.0)
    except Exception as e:
        # Fallback to direct fill if human typing fails
        try:
            page.fill(selector, text)
        except Exception:
            raise e


def _fill_form_field(page, selector: str, value: str) -> None:
    """Fill a form field using multiple strategies."""
    strategies = [
        lambda: page.locator(selector).first,
        lambda: page.get_by_placeholder(selector),
        lambda: page.get_by_label(selector),
        lambda: page.locator('input[type="text"]:visible').first,
        lambda: page.locator('input[type="email"]:visible').first,
        lambda: page.locator('input[type="password"]:visible').first,
        lambda: page.locator('textarea:visible').first,
    ]
    
    for strategy in strategies:
        try:
            element = strategy()
            element.scroll_into_view_if_needed()
            element.click()
            element.fill('')  # Clear first
            _human_type(page, selector, value)
            return
        except (OSError, RuntimeError):
            continue
    raise ValueError(f"Could not fill form field with selector: {selector}")


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
    except (OSError, RuntimeError):
        return ''


def _get_form_fields(page) -> List[Dict[str, str]]:
    """Extract visible form fields with their labels, placeholders and selectors."""
    try:
        return page.evaluate("""() => {
            const fields = [];
            const inputs = document.querySelectorAll('input:not([type=hidden]):not([type=submit]):not([type=reset]):not([type=button]), textarea, select');
            inputs.forEach((el, i) => {
                const id = el.id || el.name || '';
                let label = '';
                if (id) {
                    const lbl = document.querySelector(`label[for='${id}']`);
                    if (lbl) label = lbl.textContent.trim();
                }
                if (!label) {
                    const parent = el.closest('label');
                    if (parent) label = parent.textContent.trim();
                }
                if (!label) label = el.getAttribute('aria-label') || '';
                const placeholder = el.getAttribute('placeholder') || '';
                const type = el.getAttribute('type') || el.tagName.toLowerCase();
                const selector = el.name ? `[name='${el.name}']` : (el.id ? `#${el.id}` : `${el.tagName.toLowerCase()}:nth-of-type(${i+1})`);
                fields.push({ label: label || placeholder || type, placeholder, type, selector });
            });
            return fields;
        }""")
    except (OSError, RuntimeError):
        return []


def _detect_captcha(page) -> bool:
    """Detect if the page has a CAPTCHA."""
    try:
        captcha_indicators = [
            'captcha', 'verification', 'veuillez saisir', 'enter the code',
            'letters and numbers', 'security code', 'recaptcha', 'hcaptcha',
            'prove you are human', 'anti-bot', 'bot detection'
        ]
        page_text = _get_page_text(page, max_chars=5000).lower()
        for indicator in captcha_indicators:
            if indicator in page_text:
                return True
        # Check for common CAPTCHA elements
        captcha_selectors = [
            '.recaptcha', '#recaptcha', '.hcaptcha', '#hcaptcha',
            '[class*="captcha"]', '[id*="captcha"]'
        ]
        for selector in captcha_selectors:
            if page.locator(selector).count() > 0:
                return True
        return False
    except (OSError, RuntimeError):
        return False


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
    fields: Optional[List[Dict[str, str]]] = None
    submit_selector: Optional[str] = None


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


def _extract_search_query(task: str) -> str:
    """
    Extract the actual search keyword(s) from a natural language task sentence.
    E.g. "cherche canva" -> "canva"
         "search for python tutorials" -> "python tutorials"
    """
    strip_patterns = [
        # English
        r'^search\s+for\s+',
        r'^search\s+',
        r'^find\s+(?:the\s+|a\s+|an\s+)?',
        r'^look\s+up\s+',
        r'^look\s+for\s+',
        r'^google\s+',
        r'^lookup\s+',
        r'^what\s+is\s+',
        r'^who\s+is\s+',
        r'^how\s+to\s+',
        r'^where\s+is\s+',
        r'^get\s+(?:me\s+)?(?:information\s+(?:about|on)\s+|info\s+(?:about|on)\s+)?',
        r'^show\s+me\s+',
        r'^discover\s+',
        r'^explore\s+',
        # French
        r'^cherche[rz]?\s+(?:sur\s+google\s+)?',
        r'^recherche[rz]?\s+',
        r'^trouve[rz]?\s+(?:moi\s+)?',
        r'^trouver\s+',
        r'^cherchez?\s+',
        r'^lance\s+une\s+recherche\s+(?:sur\s+|pour\s+)?',
        r'^fais?\s+une\s+recherche\s+(?:sur\s+|pour\s+)?',
        r'^donne\s+moi\s+(?:des\s+)?(?:information[s]?\s+(?:sur|à\s+propos\s+de)\s+)?',
        r'^infos?\s+sur\s+',
        r'^quel\s+(?:est|sont)\s+',
        r'^qui\s+est\s+',
        r'^où\s+(?:est|se\s+trouve)\s+',
    ]
    query = task.strip()
    lower = query.lower()
    for pattern in strip_patterns:
        m = re.match(pattern, lower)
        if m:
            query = query[m.end():].strip()
            break
    query = query.rstrip('?.!,;')
    return query if query else task.strip()


def _create_google_search_url(task: str) -> str:
    if not task.strip():
        return 'https://www.google.com'
    query = _extract_search_query(task)
    return f'https://www.google.com/search?q={quote_plus(query)}'


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


def _create_playwright_browser_page(playwright, stale: bool, skip_anti_bot: bool, headless: bool = True):
    launch_args = _build_playwright_args(skip_anti_bot)
    if stale:
        browser = playwright.chromium.launch_persistent_context(
            user_data_dir=PLAYWRIGHT_USER_DATA_DIR,
            headless=headless,
            args=launch_args,
            viewport={'width': 1280, 'height': 800},
            user_agent=PLAYWRIGHT_USER_AGENT if skip_anti_bot else None,
            locale='en-US',
            ignore_https_errors=skip_anti_bot,
        )
        page = browser.pages[0] if browser.pages else browser.new_page()
        return browser, page

    browser = playwright.chromium.launch(headless=headless, args=launch_args)
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
    except (OSError, RuntimeError):
        pass


def _format_action_prompt(task: str, page_url: str, page_text: str, iteration: int, conversation_history: List[Dict] = None, form_fields: List[Dict] = None, memory_context: str = '') -> str:
    recent_history = ""
    if conversation_history:
        recent = conversation_history[-20:]
        recent_history = "\n".join([f"- {item.get('type', 'unknown')}: {item.get('message', item.get('question', 'N/A'))}" for item in recent])

    user_correction = ""
    if conversation_history:
        corrections = [item for item in conversation_history if item.get('type') == 'user_feedback']
        if corrections:
            user_correction = f"⚠️ {corrections[-1].get('message', '')}"

    fields_section = ""
    if form_fields:
        lines = [f"  - label='{f['label']}' placeholder='{f['placeholder']}' type='{f['type']}' selector='{f['selector']}'" for f in form_fields]
        fields_section = "## FORM FIELDS DETECTED ON PAGE:\n" + "\n".join(lines) + "\n"

    memory_section = f"{memory_context}\n\n" if memory_context else ""

    return f"""# GSAM — Web Navigation Agent
{memory_section}## TASK: {task}
## CURRENT STATE: iteration {iteration}, url: {page_url}
## PAGE CONTENT: {page_text[:800]}...
## CONVERSATION HISTORY: {recent_history or 'None'}
## USER CORRECTION (if any): {user_correction}
{fields_section}## RULES:
*** CRITICAL: NEVER invent, guess, or hallucinate values for any form field. If the user has not explicitly provided a value, use ask_user. ***
- If FORM FIELDS are detected AND user has NOT provided values in CONVERSATION HISTORY: use ask_user listing all field labels.
- If FORM FIELDS are detected AND user already provided values: use fill_form with exactly those user-provided values. Never substitute your own.
- If you cannot match a field label to the user's answer: use ask_user for that specific missing field. Do NOT fill it with fake data.
- If you are stuck or uncertain: use ask_user to ask the user for guidance instead of retrying the same action.
- Do not click on select elements or dropdowns unless specifically needed for the task.
- For login forms, ask for credentials — never invent them.
- If CAPTCHA is detected: use ask_user.
- After user provides values: use fill_form with those exact values.
- If the page has the information requested: use extract.
- RELEVANCE CHECK BEFORE done: Before using the done action, verify that the extracted or found data actually answers the user's task. If the page content is empty, off-topic, or does not match the task intent, do NOT use done — use ask_user to explain what was found and ask for clarification.
- EMPTY OR IRRELEVANT RESULTS: If after a search or data extraction the results are empty, unrelated to the task, or clearly wrong, do NOT invent a continuation or retry the same action. Immediately use ask_user to explain what happened and ask the user how to proceed.
- If task is complete AND results are relevant: use done.
- Avoid repetitive actions; think step by step.
- NEVER return empty/null — always return valid JSON.
## RESPOND NOW: only JSON

AVAILABLE ACTIONS:
{{"action":"navigate","url":"https://..."}}
{{"action":"click","selector":"CSS_SELECTOR_OR_TEXT"}}
{{"action":"type","selector":"CSS_SELECTOR","text":"text to type"}}
{{"action":"fill_form","fields":[{{"selector":"...","value":"..."}}],"submit_selector":"..."}}
{{"action":"scroll","direction":"down"}}
{{"action":"extract","data":{{"field":"value found"}}}}
{{"action":"ask_user","question":"Please provide values for the following fields: FIELD1, FIELD2, ..."}}
{{"action":"done","summary":"what was accomplished"}}

RESPOND NOW WITH ONLY THE JSON:"""


def _send_event_sync(loop: asyncio.AbstractEventLoop, send_event, data: dict) -> None:
    try:
        future = asyncio.run_coroutine_threadsafe(send_event(data), loop)
        future.result(timeout=10)
    except (RuntimeError, TimeoutError, OSError):
        pass


def _ask_gemini_sync(screenshot_base64: str, prompt: str) -> Dict[str, Any]:
    raw = None
    last_error = None
    quota_keywords = ['429', 'quota', 'rate-limit', 'rate limit', 'free_tier']

    # --- LangChain path (primary) with multimodal image support ---
    if LANGCHAIN_GOOGLE_AVAILABLE and ChatGoogleGenerativeAI and HumanMessage:
        try:
            image_bytes = base64.b64decode(screenshot_base64)
            with io.BytesIO(image_bytes) as img_buf:
                image = Image.open(img_buf)
                image.load()
            with io.BytesIO() as out_buf:
                image.save(out_buf, format='JPEG', quality=85)
                image_b64_jpeg = base64.b64encode(out_buf.getvalue()).decode('utf-8')

            model = ChatGoogleGenerativeAI(
                model='gemini-2.5-flash',
                api_key=GEMINI_API_KEY,
                temperature=0.1,
                max_tokens=512,
                timeout=30,
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
        except (OSError, RuntimeError, ValueError) as e:
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
        except ValueError:
            pass
    result = _parse_json(raw)
    if not result:
        return {'error': f'Invalid JSON from model: {raw}'}
    try:
        return BrowserAction.parse_obj(result).dict()
    except ValueError:
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
            except ValueError:
                pass
        parsed = _parse_json(raw)
        return parsed or {'error': f'Invalid JSON: {raw}'}
    except (OSError, RuntimeError, ValueError) as e:
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
            except ValueError:
                pass
        parsed = _parse_json(raw)
        return parsed or {'error': f'Invalid JSON: {raw}'}
    except (OSError, RuntimeError, ValueError) as e:
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
    feedback_queue: Optional[List[str]] = None,
    user_reply_event: Optional[Any] = None,
    show_browser: bool = False,
) -> None:
    def _run_legacy_loop(page):
        nonlocal task  # allow updating task with new user instructions
        url = page.url
        screenshot_bytes = page.screenshot(full_page=False)
        screenshot_base64 = _to_base64_png(screenshot_bytes)
        _send_event_sync(loop, send_event, {'type': 'screenshot', 'data': screenshot_base64})
        _send_event_sync(loop, send_event, {'type': 'url', 'value': url})

        # Load full persistent history so agent remembers all previous conversations
        conversation_history: List[Dict] = list(get_conversation_history(100))  # Last 100 messages
        memory_ctx = get_memory_context(task)

        # --- Stuck-loop detection state ---
        _last_state: Dict[str, Any] = {'url': None, 'text_hash': None, 'count': 0}

        def _persist(entry: Dict) -> None:
            """Append to RAM + disk simultaneously."""
            conversation_history.append(entry)
            role = entry.get('type', 'agent')
            msg = entry.get('message') or entry.get('question') or ''
            append_conversation(role, msg, task=task)

        _DIRECT_ACTION_RE = re.compile(
            r'^(click|clique|cliquer|appuie|appuyer|press|tap|scroll|scrolle|'
            r'type|tape|remplis|fill|submit|soumet|navigate|navigue|'
            r'ouvre|open|ferme|close|telecharge|download|copie|copy|'
            r'selectionne|select|coche|zoom)',
            re.IGNORECASE
        )

        def _is_direct_action(instruction: str) -> bool:
            return bool(_DIRECT_ACTION_RE.match(instruction.strip()))

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
            if _is_direct_action(task):
                return False
            try:
                if 'google.com' in page.url and 'search' not in page.url:
                    page.goto(_create_google_search_url(task), timeout=20000)
                    page.wait_for_load_state('domcontentloaded', timeout=10000)
                    return True
            except Exception:
                pass
            return False

        for iteration in range(20):
            _send_event_sync(loop, send_event, {
                'type': 'iteration',
                'current': iteration + 1,
                'total': 20
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
            form_fields = _get_form_fields(page)
            has_captcha = _detect_captcha(page)
            _send_event_sync(loop, send_event, {'type': 'screenshot', 'data': screenshot_base64})

            # --- Stuck-loop detection ---
            _cur_text_hash = hash(page_text[:500])
            if page.url == _last_state['url'] and _cur_text_hash == _last_state['text_hash']:
                _last_state['count'] += 1
            else:
                _last_state.update({'url': page.url, 'text_hash': _cur_text_hash, 'count': 1})
            if _last_state['count'] > 3:
                _last_state['count'] = 0
                stuck_q = (
                    f"I seem to be stuck on {page.url} after several iterations "
                    f"without making progress. What should I do next? "
                    f"You can tell me to navigate somewhere else, try a different approach, or stop."
                )
                _persist({'type': 'agent_question', 'question': stuck_q})
                _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': stuck_q})
                if user_reply_event:
                    user_reply_event.clear()
                deadline = time.time() + 300
                while time.time() < deadline:
                    if abort_event and abort_event.is_set():
                        return
                    if feedback_queue:
                        user_feedback = feedback_queue.pop(0)
                        _persist({'type': 'user_feedback', 'message': user_feedback})
                        _send_event_sync(loop, send_event, {'type': 'log', 'message': f'User guidance: {user_feedback}'})
                        break
                    time.sleep(0.3)
                continue

            if abort_event and abort_event.is_set():
                _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Abort detected after screenshot.'})
                return

            # --- Handle CAPTCHA first ---
            if has_captcha and not has_user_answer:
                question = 'CAPTCHA detected. Please solve the CAPTCHA and provide the verification code.'
                _persist({'type': 'agent_question', 'question': question})
                _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': question})
                if user_reply_event:
                    user_reply_event.clear()
                deadline = time.time() + 300
                while time.time() < deadline:
                    if abort_event and abort_event.is_set():
                        return
                    if feedback_queue:
                        user_feedback = feedback_queue.pop(0)
                        _persist({'type': 'user_feedback', 'message': user_feedback})
                        _send_event_sync(loop, send_event, {'type': 'log', 'message': f'User solved CAPTCHA: {user_feedback}'})
                        break
                    time.sleep(0.3)
                continue  # next iteration will handle after CAPTCHA

            # --- Direct form detection: skip LLM, ask user immediately with real field names ---
            already_asked = any(
                item.get('type') == 'agent_question' for item in conversation_history
            )
            has_user_answer = any(
                item.get('type') == 'user_feedback' for item in conversation_history
            )
            if form_fields and not has_user_answer:
                labels = [f['label'] for f in form_fields if f.get('label')]
                question = 'Please provide values for: ' + ', '.join(labels)
                _persist({'type': 'agent_question', 'question': question})
                _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': question})
                if user_reply_event:
                    user_reply_event.clear()
                deadline = time.time() + 300
                while time.time() < deadline:
                    if abort_event and abort_event.is_set():
                        return
                    if feedback_queue:
                        user_feedback = feedback_queue.pop(0)
                        _persist({'type': 'user_feedback', 'message': user_feedback})
                        _send_event_sync(loop, send_event, {'type': 'log', 'message': f'User answered: {user_feedback}'})
                        break
                    time.sleep(0.3)
                continue  # next iteration will fill the form with the answer

            # --- If user already answered, fill the form directly ---
            if form_fields and has_user_answer:
                last_answer = next(
                    (item['message'] for item in reversed(conversation_history) if item.get('type') == 'user_feedback'),
                    ''
                )
                fill_fields = []
                missing_labels = []
                for field in form_fields:
                    label = field['label']
                    selector = field['selector']
                    pattern = re.search(rf'{re.escape(label)}[:\s]+([^,\n]+)', last_answer, re.IGNORECASE)
                    value = pattern.group(1).strip() if pattern else ''
                    # If only one field and user gave a plain answer, use it directly
                    if not value and len(form_fields) == 1:
                        value = last_answer.strip()
                    if value:
                        fill_fields.append({'selector': selector, 'value': value})
                    else:
                        missing_labels.append(label)

                # NEVER hallucinate — if any field is missing, ask the user
                if missing_labels:
                    question = f'I could not find values for: {", ".join(missing_labels)}. Please provide them (format — FieldName: value).'
                    _persist({'type': 'agent_question', 'question': question})
                    _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': question})
                    if user_reply_event:
                        user_reply_event.clear()
                    deadline = time.time() + 300
                    while time.time() < deadline:
                        if abort_event and abort_event.is_set():
                            return
                        if feedback_queue:
                            user_feedback = feedback_queue.pop(0)
                            _persist({'type': 'user_feedback', 'message': user_feedback})
                            break
                        time.sleep(0.3)
                    continue  # restart loop with new answer

                if fill_fields:
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': 'FILL_FORM', 'args': str(fill_fields), 'status': 'running'})
                    for f in fill_fields:
                        try:
                            _fill_form_field(page, f['selector'], f['value'])
                            time.sleep(0.3)
                        except Exception as fe:
                            _send_event_sync(loop, send_event, {'type': 'log', 'message': f'Fill error {f["selector"]}: {fe}'})
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': 'FILL_FORM', 'args': 'done', 'status': 'done'})
                    # Submit
                    try:
                        page.click('input[type=submit], button[type=submit]', timeout=5000)
                        page.wait_for_load_state('domcontentloaded', timeout=8000)
                    except Exception:
                        pass
                    # Show result with submitted values
                    result_data = {f['selector'].strip("[name=']").replace("']", ''): f['value'] for f in fill_fields}
                    result_data['status'] = 'Form submitted successfully'
                    _send_event_sync(loop, send_event, {'type': 'result', 'data': result_data})
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SUBMIT', 'args': 'Form submitted', 'status': 'done'})
                    save_session(task, 'FORM_FILL', str(result_data), status='done')
                    next_q = 'Form submitted. What do you want to do next?'
                    _persist({'type': 'agent_question', 'question': next_q})
                    _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': next_q})
                    if user_reply_event:
                        user_reply_event.clear()
                    deadline = time.time() + 300
                    while time.time() < deadline:
                        if abort_event and abort_event.is_set():
                            return
                        if feedback_queue:
                            user_feedback = feedback_queue.pop(0)
                            _persist({'type': 'user_feedback', 'message': user_feedback})
                            has_user_answer = False
                            break
                        time.sleep(0.3)
                    else:
                        return
                    continue
            # --- No form: use LLM ---
            _send_event_sync(loop, send_event, {'type': 'thinking', 'message': f'Analyzing page — iteration {iteration + 1}/20.'})
            memory_ctx = get_memory_context(task)
            system_prompt = _format_action_prompt(
                task=task,
                page_url=page.url,
                page_text=page_text,
                iteration=iteration + 1,
                conversation_history=conversation_history,
                form_fields=form_fields,
                memory_context=memory_ctx,
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
                        _human_type(page, selector, text)
                        page.keyboard.press('Enter')
                        page.wait_for_load_state('domcontentloaded', timeout=8000)
                    except Exception as type_err:
                        _send_event_sync(loop, send_event, {
                            'type': 'log',
                            'message': f'Type failed on "{selector}": {type_err}'
                        })
                elif name == 'fill_form':
                    fields = action.get('fields', [])
                    submit_selector = action.get('submit_selector', '')
                    try:
                        for field in fields:
                            selector = field.get('selector', '')
                            value = field.get('value', '')
                            _fill_form_field(page, selector, value)
                            time.sleep(random.uniform(0.3, 0.7))  # Pause between fields
                        if submit_selector:
                            page.click(submit_selector, timeout=5000)
                            page.wait_for_load_state('domcontentloaded', timeout=8000)
                    except Exception as form_err:
                        _send_event_sync(loop, send_event, {
                            'type': 'log',
                            'message': f'Fill form failed: {form_err}'
                        })
                elif name == 'ask_user':
                    question = action.get('question', '')
                    _persist({"type": "agent_question", "question": question})
                    _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': question})
                    if user_reply_event:
                        user_reply_event.clear()
                    deadline = time.time() + 300
                    while time.time() < deadline:
                        if abort_event and abort_event.is_set():
                            return
                        if feedback_queue:
                            user_feedback = feedback_queue.pop(0)
                            _persist({"type": "user_feedback", "message": user_feedback})
                            _send_event_sync(loop, send_event, {'type': 'log', 'message': f'User answered: {user_feedback}'})
                            break
                        if user_reply_event and user_reply_event.is_set() and feedback_queue:
                            break
                        time.sleep(0.3)
                    else:
                        _send_event_sync(loop, send_event, {'type': 'log', 'message': 'ask_user timeout.'})
                    continue
                elif name == 'scroll':
                    page.evaluate('window.scrollBy(0, 600)')
                elif name == 'extract':
                    _send_event_sync(loop, send_event, {'type': 'result', 'data': action.get('data', {})})
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'done'})
                    save_session(task, 'DATA_EXTRACT', str(action.get('data', {})), status='done')
                    next_q = 'Données extraites. Que souhaitez-vous faire ensuite ?'
                    _persist({'type': 'agent_question', 'question': next_q})
                    _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': next_q})
                    if user_reply_event:
                        user_reply_event.clear()
                    deadline = time.time() + 300
                    while time.time() < deadline:
                        if abort_event and abort_event.is_set():
                            return
                        if feedback_queue:
                            user_feedback = feedback_queue.pop(0)
                            _persist({'type': 'user_feedback', 'message': user_feedback})
                            _send_event_sync(loop, send_event, {'type': 'log', 'message': f'New instruction: {user_feedback}'})
                            break
                        time.sleep(0.3)
                    else:
                        return
                    if any(w in user_feedback.lower() for w in ('stop', 'quit', 'arrête', 'non', 'quitter', 'fin', 'exit')):
                        return
                    task = user_feedback.strip()
                    append_conversation('user', task, task=task)
                    continue
                elif name == 'done':
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'done'})
                    save_session(task, 'DONE', args, status='done')
                    next_q = 'Tâche terminée. Que souhaitez-vous que je fasse ensuite ?'
                    _persist({'type': 'agent_question', 'question': next_q})
                    _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': next_q})
                    if user_reply_event:
                        user_reply_event.clear()
                    deadline = time.time() + 300
                    while time.time() < deadline:
                        if abort_event and abort_event.is_set():
                            return
                        if feedback_queue:
                            user_feedback = feedback_queue.pop(0)
                            _persist({'type': 'user_feedback', 'message': user_feedback})
                            _send_event_sync(loop, send_event, {'type': 'log', 'message': f'New instruction: {user_feedback}'})
                            break
                        time.sleep(0.3)
                    else:
                        return
                    if any(w in user_feedback.lower() for w in ('stop', 'quit', 'arrête', 'non', 'quitter', 'fin', 'exit')):
                        return
                    task = user_feedback.strip()
                    append_conversation('user', task, task=task)
                    continue
                else:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': f'Unknown action: {name}'})
            except Exception as exc:
                _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'error'})
                _send_event_sync(loop, send_event, {'type': 'error', 'message': str(exc)})
                return

            _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'done'})

            # --- Ask user after every meaningful action ---
            if name in ('navigate', 'click', 'type', 'fill_form'):
                next_q = f'Action {name_upper} effectuée. Que souhaitez-vous faire ensuite ? (dites "continue" pour que je continue automatiquement, ou donnez une nouvelle instruction)'
                _persist({'type': 'agent_question', 'question': next_q})
                _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': next_q})
                if user_reply_event:
                    user_reply_event.clear()
                deadline = time.time() + 300
                while time.time() < deadline:
                    if abort_event and abort_event.is_set():
                        return
                    if feedback_queue:
                        user_feedback = feedback_queue.pop(0)
                        _persist({'type': 'user_feedback', 'message': user_feedback})
                        _send_event_sync(loop, send_event, {'type': 'log', 'message': f'User: {user_feedback}'})
                        break
                    time.sleep(0.3)
                else:
                    return
                # If user says stop/quit/non, end session
                if any(w in user_feedback.lower() for w in ('stop', 'quit', 'arrête', 'non', 'quitter', 'fin', 'exit')):
                    save_session(task, 'USER_STOP', user_feedback, status='done')
                    return
                # New instruction becomes the new task, with full history as context
                if user_feedback.strip().lower() not in ('continue', 'continuer', 'ok', 'oui', 'yes', 'go'):
                    task = user_feedback.strip()
                    append_conversation('user', task, task=task)
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': f'New task: {task}'})
                    # Reset stuck-loop counter so new instruction gets fresh iterations
                    _last_state.update({'url': None, 'text_hash': None, 'count': 0})

            time.sleep(random.uniform(1.5, 3.0))

        title = page.title()
        _send_event_sync(loop, send_event, {'type': 'result', 'data': {'page_title': title}})

    with sync_playwright() as playwright:
        browser, page = _create_playwright_browser_page(
            playwright,
            stale=stale_browser or PLAYWRIGHT_STALE,
            skip_anti_bot=skip_anti_bot or PLAYWRIGHT_SKIP_ANTI_BOT,
            headless=not show_browser,
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

                conversation_history: List[Dict] = list(get_conversation_history(100))

                def _persist_lc(entry: Dict) -> None:
                    conversation_history.append(entry)
                    role = entry.get('type', 'agent')
                    msg = entry.get('message') or entry.get('question') or ''
                    append_conversation(role, msg, task=task)

                # --- Stuck-loop detection state (LangChain loop) ---
                _lc_last_state: Dict[str, Any] = {'url': None, 'text_hash': None, 'count': 0}

                for iteration in range(10):
                    mcp_context.update_state(current_url=page.url, iteration=iteration + 1, status='running')
                    _send_event_sync(loop, send_event, {
                        'type': 'iteration',
                        'current': iteration + 1,
                        'total': 10
                    })

                    # Check for user feedback
                    if feedback_queue:
                        while feedback_queue:
                            user_feedback = feedback_queue.pop(0)
                            _persist_lc({"type": "user_feedback", "message": user_feedback})
                            _send_event_sync(loop, send_event, {'type': 'log', 'message': f'User feedback: {user_feedback}'})

                    if abort_event and abort_event.is_set():
                        _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Abort requested during execution.'})
                        return

                    _send_event_sync(loop, send_event, {
                        'type': 'thinking',
                        'message': f'Analyzing browser state — iteration {iteration + 1}/10.'
                    })

                    page_text = rpa.get_page_text(max_chars=2000)
                    screenshot_base64 = rpa.take_screenshot()
                    form_fields = _get_form_fields(page)
                    has_captcha = _detect_captcha(page)

                    # --- Stuck-loop detection ---
                    _lc_text_hash = hash(page_text[:500])
                    if page.url == _lc_last_state['url'] and _lc_text_hash == _lc_last_state['text_hash']:
                        _lc_last_state['count'] += 1
                    else:
                        _lc_last_state.update({'url': page.url, 'text_hash': _lc_text_hash, 'count': 1})
                    if _lc_last_state['count'] > 3:
                        _lc_last_state['count'] = 0
                        stuck_q = (
                            f"I seem to be stuck on {page.url} after several iterations without progress. "
                            f"What should I do next? You can tell me to navigate somewhere else, "
                            f"try a different approach, or stop."
                        )
                        _persist_lc({'type': 'agent_question', 'question': stuck_q})
                        _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': stuck_q})
                        if user_reply_event:
                            user_reply_event.clear()
                        deadline = time.time() + 300
                        while time.time() < deadline:
                            if abort_event and abort_event.is_set():
                                return
                            if feedback_queue:
                                user_feedback = feedback_queue.pop(0)
                                _persist_lc({'type': 'user_feedback', 'message': user_feedback})
                                _send_event_sync(loop, send_event, {'type': 'log', 'message': f'User guidance: {user_feedback}'})
                                break
                            time.sleep(0.3)
                        continue

                    # --- Handle CAPTCHA first ---
                    if has_captcha and not any(item.get('type') == 'user_feedback' for item in conversation_history):
                        question = 'CAPTCHA detected. Please solve the CAPTCHA and provide the verification code.'
                        _persist_lc({'type': 'agent_question', 'question': question})
                        _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': question})
                        if user_reply_event:
                            user_reply_event.clear()
                        deadline = time.time() + 300
                        while time.time() < deadline:
                            if abort_event and abort_event.is_set():
                                return
                            if feedback_queue:
                                user_feedback = feedback_queue.pop(0)
                                _persist_lc({'type': 'user_feedback', 'message': user_feedback})
                                _send_event_sync(loop, send_event, {'type': 'log', 'message': f'User solved CAPTCHA: {user_feedback}'})
                                break
                            time.sleep(0.3)
                        continue  # next iteration

                    # --- Handle forms ---
                    has_user_answer = any(item.get('type') == 'user_feedback' for item in conversation_history)
                    if form_fields and not has_user_answer:
                        labels = [f['label'] for f in form_fields if f.get('label')]
                        question = 'Please provide values for: ' + ', '.join(labels)
                        _persist_lc({'type': 'agent_question', 'question': question})
                        _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': question})
                        if user_reply_event:
                            user_reply_event.clear()
                        deadline = time.time() + 300
                        while time.time() < deadline:
                            if abort_event and abort_event.is_set():
                                return
                            if feedback_queue:
                                user_feedback = feedback_queue.pop(0)
                                _persist_lc({'type': 'user_feedback', 'message': user_feedback})
                                _send_event_sync(loop, send_event, {'type': 'log', 'message': f'User answered: {user_feedback}'})
                                break
                            time.sleep(0.3)
                        continue  # next iteration will fill

                    if form_fields and has_user_answer:
                        last_answer = next(
                            (item['message'] for item in reversed(conversation_history) if item.get('type') == 'user_feedback'),
                            ''
                        )
                        fill_fields = []
                        for field in form_fields:
                            label = field['label']
                            selector = field['selector']
                            pattern = re.search(rf'{re.escape(label)}[:\s]+([^,]+)', last_answer, re.IGNORECASE)
                            value = pattern.group(1).strip() if pattern else ''
                            if value:
                                fill_fields.append({'selector': selector, 'value': value})
                        if fill_fields:
                            _send_event_sync(loop, send_event, {'type': 'step', 'name': 'FILL_FORM', 'args': str(fill_fields), 'status': 'running'})
                            for f in fill_fields:
                                try:
                                    _fill_form_field(page, f['selector'], f['value'])
                                    time.sleep(0.3)
                                except Exception as fe:
                                    _send_event_sync(loop, send_event, {'type': 'log', 'message': f'Fill error {f["selector"]}: {fe}'})
                            _send_event_sync(loop, send_event, {'type': 'step', 'name': 'FILL_FORM', 'args': 'done', 'status': 'done'})
                            try:
                                page.click('input[type=submit], button[type=submit]', timeout=5000)
                                page.wait_for_load_state('domcontentloaded', timeout=8000)
                            except Exception:
                                pass
                            result_data = {f['selector'].strip("[name=']").replace("']", ''): f['value'] for f in fill_fields}
                            result_data['status'] = 'Form submitted successfully'
                            _send_event_sync(loop, send_event, {'type': 'result', 'data': result_data})
                            _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SUBMIT', 'args': 'Form submitted', 'status': 'done'})
                            save_session(task, 'FORM_FILL', str(result_data), status='done')
                            next_q = 'Form submitted. What do you want to do next?'
                            _persist_lc({'type': 'agent_question', 'question': next_q})
                            _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': next_q})
                            if user_reply_event:
                                user_reply_event.clear()
                            deadline = time.time() + 300
                            while time.time() < deadline:
                                if abort_event and abort_event.is_set():
                                    return
                                if feedback_queue:
                                    user_feedback = feedback_queue.pop(0)
                                    _persist_lc({'type': 'user_feedback', 'message': user_feedback})
                                    has_user_answer = False
                                    break
                                time.sleep(0.3)
                            else:
                                return
                            continue

                    prompt = _format_react_prompt(
                        task=task,
                        page_url=page.url,
                        page_text=page_text,
                        iteration=iteration + 1,
                        context=mcp_context.get_context_summary(),
                        memory_context=get_memory_context(task),
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
                        save_session(task, 'DONE', action.get('summary', ''), status='done')
                        next_q = 'Tâche terminée. Que souhaitez-vous que je fasse ensuite ?'
                        _persist_lc({'type': 'agent_question', 'question': next_q})
                        _send_event_sync(loop, send_event, {'type': 'step', 'name': 'DONE', 'args': action.get('summary', ''), 'status': 'done'})
                        _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': next_q})
                        if user_reply_event:
                            user_reply_event.clear()
                        deadline = time.time() + 300
                        while time.time() < deadline:
                            if abort_event and abort_event.is_set():
                                return
                            if feedback_queue:
                                user_feedback = feedback_queue.pop(0)
                                _persist_lc({'type': 'user_feedback', 'message': user_feedback})
                                _send_event_sync(loop, send_event, {'type': 'log', 'message': f'New instruction: {user_feedback}'})
                                break
                            time.sleep(0.3)
                        else:
                            return
                        continue

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

                    time.sleep(random.uniform(1.5, 3.0))

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
    feedback_queue: Optional[List[str]] = None,
    user_reply_event: Optional[Any] = None,
    show_browser: bool = False,
) -> MCPContext:
    # Save user task to persistent memory
    append_conversation('user', task, task=task)

    # Inject full memory context as a priority system message before anything else
    memory_ctx = get_memory_context(task, n_sessions=10, n_conv=30)
    if memory_ctx:
        await send_event({'type': 'system_context', 'message': memory_ctx})

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
            feedback_queue,
            user_reply_event,
            show_browser,
        )
    except Exception as exc:
        await send_event({'type': 'error', 'message': str(exc)})
        await send_event({'type': 'done'})
    # NOTE: 'done' is NOT sent here automatically — the agent sends ask_user first,
    # then waits for user reply. The stream stays open until abort or timeout.
    return mcp_context
