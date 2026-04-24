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
- Treat select elements and dropdowns as normal form fields. Never skip a form section because it contains a selection control. If a value is needed and missing, use ask_user or fill_form appropriately.
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
    """Fill a form field — handles input, textarea, and select."""
    locator = page.locator(selector).first
    try:
        meta = locator.evaluate("""el => ({
            tag: el.tagName.toLowerCase(),
            role: (el.getAttribute('role') || '').toLowerCase(),
            type: (el.getAttribute('type') || '').toLowerCase(),
            popup: (el.getAttribute('aria-haspopup') || '').toLowerCase()
        })""")
    except Exception:
        meta = {'tag': 'input', 'role': '', 'type': '', 'popup': ''}

    tag = meta.get('tag', 'input')
    role = meta.get('role', '')
    input_type = meta.get('type', '')
    popup = meta.get('popup', '')

    if role in ('combobox', 'listbox') or popup == 'listbox':
        try:
            locator.scroll_into_view_if_needed()
            locator.click()
            time.sleep(0.2)
        except Exception:
            pass

        custom_option_locators = [
            lambda: page.get_by_role('option', name=value, exact=True).first,
            lambda: page.get_by_role('option', name=value, exact=False).first,
            lambda: page.get_by_role('listbox').get_by_text(value, exact=True).first,
            lambda: page.get_by_role('listbox').get_by_text(value, exact=False).first,
            lambda: page.get_by_text(value, exact=True).first,
            lambda: page.get_by_text(value, exact=False).first,
        ]
        for option_locator_factory in custom_option_locators:
            try:
                option_locator = option_locator_factory()
                option_locator.scroll_into_view_if_needed()
                option_locator.click(timeout=3000)
                return
            except Exception:
                continue
        raise ValueError(f'No option matching "{value}" in custom select {selector}')

    # --- Handle <select> ---
    if tag == 'select':
        el = locator
        # 1. exact label match
        try:
            el.select_option(label=value)
            return
        except Exception:
            pass
        # 2. exact value match
        try:
            el.select_option(value=value)
            return
        except Exception:
            pass
        # 3. fuzzy: option text/value contains the input (case-insensitive)
        try:
            options = el.evaluate(
                'el => Array.from(el.options).map(o => ({v: o.value, t: o.text.trim()}))'
            )
            match = next(
                (o for o in options
                 if value.lower() in o['t'].lower() or value.lower() in o['v'].lower()),
                None
            )
            if match:
                el.select_option(value=match['v'])
                return
        except Exception:
            pass
        raise ValueError(f'No option matching "{value}" in select {selector}')

    # --- Handle checkbox / radio ---
    if tag == 'input':
        try:
            if input_type in ('checkbox', 'radio'):
                if value.lower() in ('1', 'true', 'yes', 'oui', 'on'):
                    locator.check()
                else:
                    locator.uncheck()
                return
        except Exception:
            pass

    # --- Handle text / email / password / textarea ---
    try:
        el = locator
        el.scroll_into_view_if_needed()
        el.click()
        el.fill('')
        _human_type(page, selector, value)
        return
    except Exception:
        pass

    # Fallback strategies
    for loc in [
        lambda: page.get_by_placeholder(selector),
        lambda: page.get_by_label(selector),
        lambda: page.locator('input[type="text"]:visible').first,
        lambda: page.locator('input[type="email"]:visible').first,
        lambda: page.locator('input[type="password"]:visible').first,
        lambda: page.locator('textarea:visible').first,
    ]:
        try:
            element = loc()
            element.scroll_into_view_if_needed()
            element.click()
            element.fill('')
            _human_type(page, selector, value)
            return
        except (OSError, RuntimeError):
            continue
    raise ValueError(f'Could not fill form field: {selector}')


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
            const inferSelectKind = (options) => {
                const normalized = options.map(o => o.trim().toLowerCase()).filter(Boolean);
                const monthNames = [
                    'january', 'february', 'march', 'april', 'may', 'june',
                    'july', 'august', 'september', 'october', 'november', 'december'
                ];
                if (normalized.some(o => monthNames.includes(o))) return 'Month';
                const numeric = normalized.map(o => Number(o)).filter(n => !Number.isNaN(n));
                if (numeric.length >= 28 && Math.min(...numeric) <= 1 && Math.max(...numeric) >= 28) return 'Day';
                if (numeric.length >= 20 && Math.max(...numeric) > 1900) return 'Year';
                return '';
            };
            const unique = (items) => items.filter((item, index, arr) => item && arr.indexOf(item) === index);
            const getRefText = (attr) => {
                const ids = (attr || '').split(/\\s+/).map(v => v.trim()).filter(Boolean);
                return ids
                    .map(id => document.getElementById(id)?.textContent?.trim() || '')
                    .filter(Boolean)
                    .join(' ');
            };
            const getPopupOptions = (el) => {
                const controlledIds = [
                    el.getAttribute('aria-controls'),
                    el.getAttribute('aria-owns')
                ].filter(Boolean);
                const fromControlled = controlledIds.flatMap(id => {
                    const popup = document.getElementById(id);
                    if (!popup) return [];
                    return Array.from(popup.querySelectorAll('[role="option"], option, li')).map(node => node.textContent.trim());
                });
                const visibleLists = Array.from(document.querySelectorAll('[role="listbox"], [role="menu"], ul[role], div[role="dialog"]'))
                    .filter(node => {
                        const style = window.getComputedStyle(node);
                        const rect = node.getBoundingClientRect();
                        return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
                    })
                    .flatMap(node => Array.from(node.querySelectorAll('[role="option"], option, li')).map(item => item.textContent.trim()));
                return unique([...fromControlled, ...visibleLists].map(v => v.trim()).filter(Boolean));
            };
            const isTechnicalValue = (text) => /^_?[a-z]+(?:_[a-z0-9]+)+_?$/i.test((text || '').trim());
            const isNoisyLabel = (text) => {
                const value = (text || '').trim();
                if (!value) return true;
                if (value.length > 80) return true;
                if (/[0-9]{8,}/.test(value)) return true;
                if (isTechnicalValue(value)) return true;
                return false;
            };
            const inputs = document.querySelectorAll(`
                input:not([type=hidden]):not([type=submit]):not([type=reset]):not([type=button]),
                textarea,
                select,
                [role="combobox"],
                [role="listbox"],
                [aria-haspopup="listbox"]
            `);
            inputs.forEach((el, i) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                if (style.display === 'none' || style.visibility === 'hidden' || rect.width === 0 || rect.height === 0) return;

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
                if (!label) label = getRefText(el.getAttribute('aria-labelledby'));
                if (!label) label = el.getAttribute('aria-label') || '';
                if (!label) label = getRefText(el.getAttribute('aria-describedby'));
                if (!label) label = el.getAttribute('title') || '';
                if (!label) {
                    const group = el.closest('fieldset,[role="group"],[role="radiogroup"]');
                    const legend = group?.querySelector('legend,[aria-label],[data-label]');
                    if (legend) label = (legend.textContent || legend.getAttribute('aria-label') || legend.getAttribute('data-label') || '').trim();
                }
                const placeholder = el.getAttribute('placeholder') || '';
                const tag = el.tagName.toLowerCase();
                const role = (el.getAttribute('role') || '').toLowerCase();
                const inputType = (el.getAttribute('type') || '').toLowerCase();
                const popup = (el.getAttribute('aria-haspopup') || '').toLowerCase();
                const type = tag === 'select' || role === 'combobox' || role === 'listbox' || popup === 'listbox'
                    ? 'select'
                    : (inputType || tag);
                const options = tag === 'select'
                    ? Array.from(el.options).map(o => o.textContent.trim()).filter(Boolean)
                    : Array.from(el.querySelectorAll('[role="option"], option')).map(o => o.textContent.trim()).filter(Boolean)
                        .filter((option, index, arr) => arr.indexOf(option) === index);
                const resolvedOptions = type === 'select' && options.length === 0 ? getPopupOptions(el) : options;
                const selectKind = type === 'select' ? inferSelectKind(resolvedOptions) : '';
                const firstOption = resolvedOptions[0] || '';
                const selectHint = type === 'select'
                    ? (selectKind || placeholder || firstOption || (!isTechnicalValue(el.getAttribute('name') || '') ? el.getAttribute('name') : '') || (!isTechnicalValue(el.id || '') ? el.id : '') || '')
                    : '';
                const selector = el.id
                    ? `#${el.id}`
                    : el.name
                        ? `[name='${el.name}']`
                        : el.getAttribute('data-testid')
                            ? `[data-testid='${el.getAttribute('data-testid')}']`
                            : el.getAttribute('aria-label')
                                ? `[aria-label='${el.getAttribute('aria-label')}']`
                                : `${tag}:nth-of-type(${i+1})`;
                const cleanLabel = type === 'select'
                    ? (
                        isNoisyLabel(label)
                            ? (selectHint || type)
                            : (label && selectHint && !label.toLowerCase().includes(selectHint.toLowerCase())
                                ? `${label} - ${selectHint}`
                                : (label || selectHint || type))
                    )
                    : (label || placeholder || (!isTechnicalValue(el.getAttribute('name') || '') ? el.getAttribute('name') : '') || (!isTechnicalValue(el.id || '') ? el.id : '') || type);
                fields.push({ label: cleanLabel, placeholder, type, selector, options: resolvedOptions });
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


def _get_select_options(page, selector: str) -> List[str]:
    """Best-effort extraction of options from native or custom select widgets."""
    try:
        options = page.locator(selector).first.evaluate("""el => {
            const unique = (items) => items.filter((item, index, arr) => item && arr.indexOf(item) === index);
            const direct = el.tagName.toLowerCase() === 'select'
                ? Array.from(el.options).map(o => o.textContent.trim())
                : Array.from(el.querySelectorAll('[role="option"], option')).map(o => o.textContent.trim());
            if (direct.filter(Boolean).length) return unique(direct.map(v => v.trim()).filter(Boolean));

            const controlledIds = [el.getAttribute('aria-controls'), el.getAttribute('aria-owns')].filter(Boolean);
            const controlled = controlledIds.flatMap(id => {
                const popup = document.getElementById(id);
                if (!popup) return [];
                return Array.from(popup.querySelectorAll('[role="option"], option, li')).map(node => node.textContent.trim());
            });
            const visibleLists = Array.from(document.querySelectorAll('[role="listbox"], [role="menu"]'))
                .filter(node => {
                    const style = window.getComputedStyle(node);
                    const rect = node.getBoundingClientRect();
                    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
                })
                .flatMap(node => Array.from(node.querySelectorAll('[role="option"], option, li')).map(item => item.textContent.trim()));
            return unique([...controlled, ...visibleLists].map(v => v.trim()).filter(Boolean));
        }""")
        return options or []
    except Exception:
        return []


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
        lines = []
        for field in form_fields:
            options = field.get('options') or []
            option_suffix = f" options={options}" if options else ""
            lines.append(
                f"  - label='{field['label']}' placeholder='{field['placeholder']}' "
                f"type='{field['type']}' selector='{field['selector']}'{option_suffix}"
            )
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
- Treat select elements and dropdowns as normal form fields. Never skip a form section because it contains a selection control. If the user provided a value, use fill_form; otherwise use ask_user with the available options.
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

            has_user_answer = any(
                item.get('type') == 'user_feedback' for item in conversation_history
            )

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

            # --- Direct form detection: ask and fill field by field ---
            already_asked = any(
                item.get('type') == 'agent_question' for item in conversation_history
            )
            if form_fields and not has_user_answer:
                all_filled = True
                for field in form_fields:
                    label = field['label']
                    selector = field['selector']
                    field_type = field.get('type', '')
                    value = None
                    # Build question — for select, list available options
                    if field_type == 'select':
                        try:
                            options = field.get('options') or _get_select_options(page, selector)
                            if options:
                                opts_str = ', '.join(options)
                                base_question = f'{label} (options: {opts_str}) :'
                            else:
                                fallback_label = label if label and label.lower() != 'select' else 'Choose an option'
                                base_question = f'{fallback_label} :'
                        except Exception:
                            fallback_label = label if label and label.lower() != 'select' else 'Choose an option'
                            base_question = f'{fallback_label} :'
                    else:
                        base_question = f'{label} :'
                    # Ask for this field, retry on fill error
                    while True:
                        question = base_question if value is None else f'Erreur sur "{label}". Entrez une nouvelle valeur :'
                        _persist({'type': 'agent_question', 'question': question})
                        _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': question})
                        if user_reply_event:
                            user_reply_event.clear()
                        deadline = time.time() + 300
                        answered = False
                        while time.time() < deadline:
                            if abort_event and abort_event.is_set():
                                return
                            if feedback_queue:
                                val = feedback_queue.pop(0)
                                _persist({'type': 'user_feedback', 'message': val})
                                value = val.strip()
                                answered = True
                                break
                            time.sleep(0.3)
                        if not answered:
                            return
                        # Try to fill immediately
                        try:
                            _fill_form_field(page, selector, value)
                            _send_event_sync(loop, send_event, {'type': 'step', 'name': 'FILL', 'args': f'{label} = {value}', 'status': 'done'})
                            break  # field filled successfully, move to next
                        except Exception as fe:
                            err_msg = f'Impossible de remplir "{label}" : {fe}'
                            _send_event_sync(loop, send_event, {'type': 'log', 'message': err_msg})
                            # Loop back to re-ask with error message
                            question = f'Erreur sur "{label}" ({fe}). Entrez une nouvelle valeur :'
                            value = None  # force re-ask
                # All fields filled — submit
                _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SUBMIT', 'args': 'submitting form', 'status': 'running'})
                try:
                    page.click('input[type=submit], button[type=submit]', timeout=5000)
                    page.wait_for_load_state('domcontentloaded', timeout=8000)
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SUBMIT', 'args': 'Form submitted', 'status': 'done'})
                except Exception:
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SUBMIT', 'args': 'no submit button found', 'status': 'done'})
                save_session(task, 'FORM_FILL', 'form filled field by field', status='done')
                # Clear form Q&A from history so next iteration doesn't re-fill
                conversation_history[:] = [e for e in conversation_history if e.get('type') not in ('agent_question', 'user_feedback')]
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
                    # Wait silently for next instruction typed in the input bar
                    deadline = time.time() + 600
                    while time.time() < deadline:
                        if abort_event and abort_event.is_set():
                            return
                        if feedback_queue:
                            nxt = feedback_queue.pop(0)
                            _persist({'type': 'user_feedback', 'message': nxt})
                            task = nxt.strip()
                            append_conversation('user', task, task=task)
                            _last_state.update({'url': None, 'text_hash': None, 'count': 0})
                            break
                        time.sleep(0.5)
                    else:
                        return
                    continue
                elif name == 'done':
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'done'})
                    save_session(task, 'DONE', args, status='done')
                    # Wait silently for next instruction typed in the input bar
                    deadline = time.time() + 600
                    while time.time() < deadline:
                        if abort_event and abort_event.is_set():
                            return
                        if feedback_queue:
                            nxt = feedback_queue.pop(0)
                            _persist({'type': 'user_feedback', 'message': nxt})
                            task = nxt.strip()
                            append_conversation('user', task, task=task)
                            _last_state.update({'url': None, 'text_hash': None, 'count': 0})
                            break
                        time.sleep(0.5)
                    else:
                        return
                    continue
                else:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': f'Unknown action: {name}'})
            except Exception as exc:
                _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'error'})
                _send_event_sync(loop, send_event, {'type': 'error', 'message': str(exc)})
                return

            _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'done'})

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
            except (OSError, RuntimeError) as nav_err:
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

            _run_legacy_loop(page)

        except (OSError, RuntimeError) as exc:
            _send_event_sync(loop, send_event, {'type': 'error', 'message': str(exc)})
        finally:
            browser.close()


def _extract_start_url(task: str) -> str:
    explicit_url = re.search(r'(https?://[^\s,;"]+)', task, re.IGNORECASE)
    if explicit_url:
        return explicit_url.group(1).strip().rstrip('.?,;')

    domain_match = re.search(
        r'\b([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com|net|org|io|ai|gov|edu|fr|co|us)(?:/[^\s]*)?\b',
        task, re.IGNORECASE
    )
    if domain_match:
        domain = domain_match.group(0).strip().rstrip('.?,;')
        search_intent = re.search(
            r'\b(search|find|look for|lookup|look up|what is|who is|how to|where is|details|information|recherche|cherche|trouver|scan|search for|find the|list the|find all|discover|crawl|audit)\b',
            task, re.IGNORECASE
        )
        direct_navigation = re.search(
            r'\b(go to|visit|open|navigate to|browse to|access|launch|open website|visit site|check the site)\b',
            task, re.IGNORECASE
        )
        if direct_navigation and not search_intent:
            return f'https://{domain}'
        return ''

    return ''


async def _get_start_url(task: str) -> str:
    extracted_url = _extract_start_url(task)
    if extracted_url:
        return extracted_url
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
    append_conversation('user', task, task=task)

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
    return mcp_context
