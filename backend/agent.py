import asyncio
import base64
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import io
from urllib.parse import quote_plus, urlparse

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
from backend.memory import append_conversation, ensure_topic_session, get_active_session, get_conversation_history, get_memory_context, save_session
from backend.conversational_agent import is_conversational_question, answer_conversational_question
from backend.workflow_manager import (
    find_matching_action_memory,
    get_or_create_action_memory_workflow,
    get_workflow_actions,
    record_workflow_action,
)

PROVIDER = os.getenv('PROVIDER', 'gemini').lower()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PLAYWRIGHT_STALE = os.getenv('PLAYWRIGHT_STALE', 'true').lower() in ('1', 'true', 'yes')
PLAYWRIGHT_SKIP_ANTI_BOT = os.getenv('PLAYWRIGHT_SKIP_ANTI_BOT', 'true').lower() in ('1', 'true', 'yes')
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
PAGE_ACTION_PREFIX = '__PAGE_ACTION__:'

def _normalize_gemini_model_name(model_name: str) -> str:
    if model_name.startswith('publishers/google/models/'):
        parts = model_name.split('/')
        return 'models/' + '/'.join(parts[3:])
    return model_name


def _extract_page_action(message: str) -> str:
    text = (message or '').strip()
    if text.startswith(PAGE_ACTION_PREFIX):
        return text[len(PAGE_ACTION_PREFIX):].strip()
    return ''


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
            user_correction = f"âš ï¸ {corrections[-1].get('message', '')}"

    memory_section = f"{memory_context}\n\n" if memory_context else ""
    return f"""# GSAM â€” Web Navigation Agent
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
- RELEVANCE CHECK BEFORE done: Before using the done action, verify that the extracted or found data actually answers the user's task. If the page content is empty, off-topic, or does not match the task intent, do NOT use done â€” use ask_user to explain what was found and ask for clarification.
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
    """Fill a form field â€” handles input, textarea, and select."""
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
        r'^donne\s+moi\s+(?:des\s+)?(?:information[s]?\s+(?:sur|Ã \s+propos\s+de)\s+)?',
        r'^infos?\s+sur\s+',
        r'^quel\s+(?:est|sont)\s+',
        r'^qui\s+est\s+',
        r'^oÃ¹\s+(?:est|se\s+trouve)\s+',
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


def _source_search_url(product_query: str, source_label: str, source_domain: str) -> str:
    query = _extract_search_query(product_query)
    domain = source_domain.lower().strip()
    if domain.startswith('www.'):
        domain = domain[4:]
    if 'amazon' in domain:
        return f'https://www.amazon.com/s?k={quote_plus(query)}'
    elif 'aliexpress' in domain:
        return f'https://www.aliexpress.com/wholesale?SearchText={quote_plus(query)}'
    elif 'ebay' in domain:
        return f'https://www.ebay.com/sch/i.html?_nkw={quote_plus(query)}'
    elif 'alibaba' in domain:
        return f'https://www.alibaba.com/trade/search?SearchText={quote_plus(query)}'
    elif 'etsy' in domain:
        return f'https://www.etsy.com/search?q={quote_plus(query)}'
    elif 'walmart' in domain:
        return f'https://www.walmart.com/search?q={quote_plus(query)}'
    elif 'github.com' in domain:
        return f'https://github.com/search?q={quote_plus(query)}&type=users'
    elif 'linkedin.com' in domain:
        return f'https://www.linkedin.com/search/results/people/?keywords={quote_plus(query)}'
    elif 'youtube.com' in domain:
        return f'https://www.youtube.com/results?search_query={quote_plus(query)}'
    elif 'reddit.com' in domain:
        return f'https://www.reddit.com/search/?q={quote_plus(query)}'
    elif 'x.com' in domain or 'twitter.com' in domain:
        return f'https://x.com/search?q={quote_plus(query)}&src=typed_query'
    else:
        return f'https://{domain}/search?q={quote_plus(query)}'


def _build_playwright_args(skip_anti_bot: bool) -> List[str]:
    args = ['--no-sandbox']
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
            user_agent=PLAYWRIGHT_USER_AGENT,
            locale='en-US',
            ignore_https_errors=True,
        )
        page = browser.pages[0] if browser.pages else browser.new_page()
        return browser, page

    browser = playwright.chromium.launch(headless=headless, args=launch_args)
    context = browser.new_context(
        viewport={'width': 1280, 'height': 800},
        user_agent=PLAYWRIGHT_USER_AGENT,
        locale='en-US',
        extra_http_headers={'accept-language': 'en-US,en;q=0.9'},
        ignore_https_errors=True,
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


def _format_action_prompt(
    task: str,
    page_url: str,
    page_text: str,
    iteration: int,
    conversation_history: List[Dict] = None,
    form_fields: List[Dict] = None,
    memory_context: str = '',
    agent_context: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_description: Optional[str] = None,
) -> str:
    recent_history = ""
    if conversation_history:
        recent = conversation_history[-20:]
        recent_history = "\n".join([f"- {item.get('type', 'unknown')}: {item.get('message', item.get('question', 'N/A'))}" for item in recent])

    user_correction = ""
    if conversation_history:
        corrections = [item for item in conversation_history if item.get('type') == 'user_feedback']
        if corrections:
            user_correction = f"âš ï¸ {corrections[-1].get('message', '')}"

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
    agent_section = ""
    if agent_context:
        agent_section = f"""=== SELECTED CONTEXTUAL AGENT ===
Agent name: {agent_name or 'Unnamed agent'}
Agent description: {agent_description or ''}

Agent system context:
{agent_context}
=== END SELECTED CONTEXTUAL AGENT ===

Contextual agent rules:
- Treat the user's task as the input for this selected agent. If the user only provided a product, topic, name, or short phrase, infer the workflow from the agent system context.
- Use the agent system context to choose search queries, sources, report format, and completion criteria.
- If the agent system context names specific websites or domains, use those websites first. Do not switch to generic Google search unless no website/source is provided.
- Extract the exact fields requested by the agent system context. Examples: nom/name, username, profil/link, domaine/travail/bio, email, prix, avis, localisation.
- If the context asks for research, competitors, comparison, products, sources, or a report: collect information from several relevant sources before done.
- CRITICAL FOR E-COMMERCE SITES (Amazon, eBay, AliExpress, Alibaba, Etsy, Walmart):
  1. If you see a SEARCH RESULTS page with multiple products listed, you MUST click on the FIRST product link to go to the actual product page
  2. Once on a PRODUCT PAGE (URL contains /dp/, /item/, /itm/, or 'product'), the system will automatically extract data
  3. Do NOT use extract action on search results pages - only click to navigate to product pages
  4. Do NOT use ask_user when doing product research - always click on products to get to product pages
- For non-commerce sources such as GitHub, LinkedIn, YouTube, Reddit, or a custom domain: use that site's own search/results pages and extract the relevant visible results.
- For each requested source, finish collecting useful data before moving to the next source.
- Build the final answer as a report that directly follows the selected agent's context.
- Do not finish with done until the result satisfies the selected agent's requested output.
- Do not ask the user what to do next after the first task is given. Search, extract, and report from the configured sources. Only ask_user for CAPTCHA, login credentials, or a truly missing required value.

"""

    return f"""# GSAM â€” Web Navigation Agent
{memory_section}{agent_section}## TASK: {task}
## CURRENT STATE: iteration {iteration}, url: {page_url}
## PAGE CONTENT: {page_text[:800]}...
## CONVERSATION HISTORY: {recent_history or 'None'}
## USER CORRECTION (if any): {user_correction}
{fields_section}## RULES:
*** CRITICAL: NEVER invent, guess, or hallucinate values for any form field. If the user has not explicitly provided a value, use ask_user. ***
- IMPORTANT: For contextual agents, NEVER use ask_user unless CAPTCHA, login credentials, or a required missing value blocks the task. Use extract to collect data from the current page.
- If FORM FIELDS are detected AND user has NOT provided values in CONVERSATION HISTORY: use ask_user listing all field labels.
- If FORM FIELDS are detected AND user already provided values: use fill_form with exactly those user-provided values. Never substitute your own.
- If you cannot match a field label to the user's answer: use ask_user for that specific missing field. Do NOT fill it with fake data.
- If you are stuck or uncertain: use ask_user to ask the user for guidance instead of retrying the same action.
- Treat select elements and dropdowns as normal form fields. Never skip a form section because it contains a selection control. If the user provided a value, use fill_form; otherwise use ask_user with the available options.
- For login forms, ask for credentials â€” never invent them.
- If CAPTCHA is detected: use ask_user.
- After user provides values: use fill_form with those exact values.
- If the page has the information requested: use extract.
- RELEVANCE CHECK BEFORE done: Before using the done action, verify that the extracted or found data actually answers the user's task. If the page content is empty, off-topic, or does not match the task intent, do NOT use done â€” use ask_user to explain what was found and ask for clarification.
- EMPTY OR IRRELEVANT RESULTS: If after a search or data extraction the results are empty, unrelated to the task, or clearly wrong, do NOT invent a continuation or retry the same action. Immediately use ask_user to explain what happened and ask the user how to proceed.
- If task is complete AND results are relevant: use done.
- Avoid repetitive actions; think step by step.
- NEVER return empty/null â€” always return valid JSON.
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


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _action_selector(action: Dict[str, Any]) -> Optional[str]:
    name = action.get('action')
    if name in ('click', 'type'):
        return action.get('selector') or None
    if name == 'fill_form':
        fields = action.get('fields') or []
        if fields:
            return fields[0].get('selector') or None
        return action.get('submit_selector') or None
    if name == 'submit':
        return action.get('selector') or action.get('submit_selector') or None
    return None


def _action_input_value(action: Dict[str, Any]) -> Optional[str]:
    if action.get('action') == 'type':
        return action.get('text') or None
    if action.get('action') == 'fill_form':
        fields = action.get('fields') or []
        values = [str(field.get('value', '')) for field in fields if field.get('value') is not None]
        return json.dumps(values, ensure_ascii=False) if values else None
    return None


def _is_search_like_url(url: str) -> bool:
    host = (urlparse(url).netloc or '').lower()
    return any(domain in host for domain in ('google.', 'bing.', 'duckduckgo.'))


def _prepare_replayed_action(
    action_data: Dict[str, Any],
    task: str,
    page_url: str,
    contextual_search_query: Optional[str],
) -> Dict[str, Any]:
    action = dict(action_data)
    if action.get('action') == 'type' and _is_search_like_url(page_url):
        action['text'] = contextual_search_query or task
    if action.get('action') == 'fill_form' and _is_search_like_url(page_url):
        fields = [dict(field) for field in action.get('fields', [])]
        if len(fields) == 1:
            fields[0]['value'] = contextual_search_query or task
            action['fields'] = fields
    return action


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
    agent_context: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_description: Optional[str] = None,
    contextual_search_query: Optional[str] = None,
    memory_namespace: Optional[str] = None,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    def _run_legacy_loop(page):
        nonlocal task  # allow updating task with new user instructions
        url = page.url
        screenshot_bytes = page.screenshot(full_page=False)
        screenshot_base64 = _to_base64_png(screenshot_bytes)
        _send_event_sync(loop, send_event, {'type': 'screenshot', 'data': screenshot_base64})
        _send_event_sync(loop, send_event, {'type': 'url', 'value': url})

        # Load full persistent history so agent remembers all previous conversations
        conversation_history: List[Dict] = list(get_conversation_history(100, namespace=memory_namespace))  # Last 100 messages
        memory_ctx = get_memory_context(task, namespace=memory_namespace)
        contextual_sources = _get_contextual_sources(agent_context)
        contextual_source_index = 1 if (agent_context and contextual_sources) else 0
        contextual_research_results: List[Dict[str, Any]] = []
        contextual_target_count = len(contextual_sources) if (agent_context and contextual_sources) else 0
        if contextual_sources:
            _send_event_sync(loop, send_event, {
                'type': 'log',
                'message': 'Contextual sources detected: ' + ', '.join(
                    f"{source['name']} ({source['domain']})" for source in contextual_sources
                )
            })

        # --- Stuck-loop detection state ---
        _last_state: Dict[str, Any] = {'url': None, 'text_hash': None, 'count': 0}
        _skip_form_once = False  # set True after AUTRE ACTION to bypass form detection

        def _persist(entry: Dict) -> None:
            """Append to RAM + disk simultaneously."""
            conversation_history.append(entry)
            role = entry.get('type', 'agent')
            msg = entry.get('message') or entry.get('question') or ''
            append_conversation(role, msg, task=task, namespace=memory_namespace)

        def _switch_to_page_action(raw_feedback: str) -> bool:
            nonlocal task
            page_action = _extract_page_action(raw_feedback)
            if not page_action:
                return False
            task = page_action
            ensure_topic_session(task, namespace=memory_namespace, topic=agent_name or agent_description or None, allow_topic_switch=not bool(agent_id))
            _persist({'type': 'user_feedback', 'message': task})
            append_conversation('user', task, task=task, namespace=memory_namespace)
            conversation_history[:] = list(get_conversation_history(100, namespace=memory_namespace))
            active = get_active_session(memory_namespace)
            if active:
                _send_event_sync(loop, send_event, {'type': 'session', 'data': active})
            _send_event_sync(loop, send_event, {
                'type': 'log',
                'message': f'User redirected the agent to a different action on the current page: {task}'
            })
            _last_state.update({'url': None, 'text_hash': None, 'count': 0})
            return True

        _DIRECT_ACTION_RE = re.compile(
            r'^(click|clique|cliquer|appuie|appuyer|press|tap|scroll|scrolle|'
            r'type|tape|remplis|fill|submit|soumet|navigate|navigue|'
            r'ouvre|open|ferme|close|telecharge|download|copie|copy|'
            r'selectionne|select|coche|zoom)',
            re.IGNORECASE
        )

        def _is_direct_action(instruction: str) -> bool:
            return bool(_DIRECT_ACTION_RE.match(instruction.strip()))

        def _is_search_results_page(url: str) -> bool:
            return (
                'google.com/search' in url
                or 'google.com/sorry' in url
                or 'bing.com/search' in url
                or 'duckduckgo.com/' in url
            )

        def _click_first_search_result(page) -> Optional[str]:
            # Google + Bing result selectors
            for selector in [
                'a h3', 'div#search a h3',
                '#b_results h2 a', '.b_algo h2 a', 'li.b_algo a h2',
            ]:
                try:
                    if page.locator(selector).count() > 0:
                        page.locator(selector).first.click(timeout=8000)
                        page.wait_for_load_state('domcontentloaded', timeout=10000)
                        return selector
                except Exception:
                    pass
            return None

        def _go_to_next_contextual_source(page, reason: str) -> bool:
            nonlocal contextual_source_index
            if not (agent_context and contextual_sources):
                return False
            if contextual_source_index >= len(contextual_sources):
                return False
            next_source = contextual_sources[contextual_source_index]
            contextual_source_index += 1
            next_url = _source_search_url(task, next_source['name'], next_source['domain'])
            _send_event_sync(loop, send_event, {
                'type': 'log',
                'message': f'{reason}; moving to next contextual source: {next_source["name"]}'
            })
            try:
                page.goto(next_url, timeout=20000)
                page.wait_for_load_state('domcontentloaded', timeout=10000)
            except Exception as nav_err:
                _send_event_sync(loop, send_event, {
                    'type': 'log',
                    'message': f'Navigation to {next_source["name"]} failed: {nav_err}'
                })
            _send_event_sync(loop, send_event, {'type': 'url', 'value': page.url})
            _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SEARCH', 'args': next_source['name'], 'status': 'done'})
            _last_state.update({'url': None, 'text_hash': None, 'count': 0})
            return True

        def _force_google_query(task: str, page) -> bool:
            if _is_direct_action(task):
                return False
            try:
                if 'google.com' in page.url and 'search' not in page.url:
                    page.goto(_create_google_search_url(contextual_search_query or task), timeout=20000)
                    page.wait_for_load_state('domcontentloaded', timeout=10000)
                    return True
            except Exception:
                pass
            return False

        action_memory_workflow: Optional[Dict[str, Any]] = None
        action_memory_step = 0

        def _ensure_action_memory_workflow() -> Optional[str]:
            nonlocal action_memory_workflow, action_memory_step
            if not user_id:
                return None
            if action_memory_workflow is None:
                try:
                    action_memory_workflow = get_or_create_action_memory_workflow(user_id, task, start_url)
                    action_memory_step = len(get_workflow_actions(action_memory_workflow['id']))
                except Exception as memory_err:
                    _send_event_sync(loop, send_event, {
                        'type': 'log',
                        'message': f'Action memory unavailable: {memory_err}'
                    })
                    action_memory_workflow = {}
            return action_memory_workflow.get('id') if action_memory_workflow else None

        def _record_playwright_action(action: Dict[str, Any], status: str, error_message: str = None) -> None:
            nonlocal action_memory_step
            name = action.get('action')
            if name not in {'click', 'type', 'fill_form', 'submit'}:
                return
            selector = _action_selector(action)
            page_url = getattr(page, 'url', '') or ''
            timestamp = _utc_timestamp()
            action_data = dict(action)
            action_data.update({
                'selector': selector,
                'url': page_url,
                'status': status,
                'timestamp': timestamp,
            })
            if error_message:
                action_data['error'] = error_message
            mcp_context.add_action(name, action_data)

            workflow_id = _ensure_action_memory_workflow()
            if not workflow_id:
                return
            try:
                action_memory_step += 1
                record_workflow_action(
                    workflow_id=workflow_id,
                    step_number=action_memory_step,
                    action_type=name,
                    action_data=action_data,
                    page_url=page_url,
                    selector=selector,
                    input_value=_action_input_value(action),
                    success=status == 'success',
                    error_message=error_message,
                )
            except Exception as memory_err:
                _send_event_sync(loop, send_event, {
                    'type': 'log',
                    'message': f'Action memory write failed: {memory_err}'
                })

        def _execute_playwright_action(action: Dict[str, Any], replay: bool = False) -> tuple[bool, Optional[str]]:
            name = action.get('action')
            try:
                if name == 'click':
                    selector = action.get('selector', '')
                    try:
                        page.click(selector, timeout=5000)
                    except Exception:
                        try:
                            page.get_by_text(selector, exact=False).first.click(timeout=5000)
                        except Exception:
                            page.get_by_role('button', name=selector).click(timeout=5000)
                elif name == 'type':
                    selector = action.get('selector', '')
                    text = action.get('text') or ''
                    _human_type(page, selector, text)
                    page.keyboard.press('Enter')
                    page.wait_for_load_state('domcontentloaded', timeout=8000)
                elif name == 'fill_form':
                    fields = action.get('fields', [])
                    submit_selector = action.get('submit_selector', '')
                    for field in fields:
                        selector = field.get('selector', '')
                        value = field.get('value', '')
                        _fill_form_field(page, selector, value)
                        time.sleep(random.uniform(0.3, 0.7))
                    if submit_selector:
                        page.click(submit_selector, timeout=5000)
                        page.wait_for_load_state('domcontentloaded', timeout=8000)
                        if not replay:
                            _record_playwright_action(
                                {'action': 'submit', 'selector': submit_selector},
                                'success',
                            )
                elif name == 'submit':
                    selector = action.get('selector') or action.get('submit_selector') or ''
                    page.click(selector, timeout=5000)
                    page.wait_for_load_state('domcontentloaded', timeout=8000)
                else:
                    return False, f'Unsupported replay action: {name}'
                return True, None
            except Exception as exc:
                return False, str(exc)

        replayed_action_memory = False

        def _replay_saved_actions_once() -> bool:
            nonlocal replayed_action_memory
            if replayed_action_memory or not user_id:
                return False
            replayed_action_memory = True
            try:
                workflow = find_matching_action_memory(user_id, task, page.url)
                if not workflow:
                    return False
                saved_actions = [
                    item for item in get_workflow_actions(workflow['id'])
                    if item.get('success', True)
                    and item.get('actionType') in {'click', 'type', 'fill_form', 'submit'}
                ][:8]
            except Exception as memory_err:
                _send_event_sync(loop, send_event, {
                    'type': 'log',
                    'message': f'Action memory replay lookup failed: {memory_err}'
                })
                return False
            if not saved_actions:
                return False

            _send_event_sync(loop, send_event, {
                'type': 'log',
                'message': f'Replaying {len(saved_actions)} saved action(s) from action memory.'
            })
            replayed_any = False
            for saved_action in saved_actions:
                action_data = saved_action.get('actionData') or {}
                action_data.setdefault('action', saved_action.get('actionType'))
                action_data = _prepare_replayed_action(action_data, task, page.url, contextual_search_query)
                name = action_data.get('action')
                selector = _action_selector(action_data) or ''
                args = selector
                if name == 'type':
                    args = f"{selector} | {action_data.get('text', '')}"
                _send_event_sync(loop, send_event, {
                    'type': 'step',
                    'name': f'REPLAY_{str(name or "").upper()}',
                    'args': args,
                    'status': 'running',
                })
                success, error_message = _execute_playwright_action(action_data, replay=True)
                _send_event_sync(loop, send_event, {
                    'type': 'step',
                    'name': f'REPLAY_{str(name or "").upper()}',
                    'args': args,
                    'status': 'done' if success else 'error',
                })
                if not success:
                    _send_event_sync(loop, send_event, {
                        'type': 'log',
                        'message': f'Replay stopped on {name}: {error_message}'
                    })
                    break
                replayed_any = True
                _send_event_sync(loop, send_event, {'type': 'url', 'value': page.url})
                time.sleep(random.uniform(0.5, 1.0))
            if replayed_any:
                _last_state.update({'url': None, 'text_hash': None, 'count': 0})
            return replayed_any

        for iteration in range(20):
            _send_event_sync(loop, send_event, {
                'type': 'iteration',
                'current': iteration + 1,
                'total': 20
            })

            if abort_event and abort_event.is_set():
                _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Abort requested during execution.'})
                return

            # Check if this is a conversational question
            if is_conversational_question(task):
                answer = answer_conversational_question(task, conversation_history)
                _persist({'type': 'agent_question', 'question': answer})
                _send_event_sync(loop, send_event, {'type': 'ask_user', 'question': answer})
                if user_reply_event:
                    user_reply_event.clear()
                deadline = time.time() + 300
                while time.time() < deadline:
                    if abort_event and abort_event.is_set():
                        return
                    if feedback_queue:
                        nxt = feedback_queue.pop(0)
                        task = nxt.strip()
                        ensure_topic_session(task, namespace=memory_namespace, topic=agent_name or agent_description or None, allow_topic_switch=not bool(agent_id))
                        _persist({'type': 'user_feedback', 'message': task})
                        append_conversation('user', task, task=task, namespace=memory_namespace)
                        conversation_history[:] = list(get_conversation_history(100, namespace=memory_namespace))
                        active = get_active_session(memory_namespace)
                        if active:
                            _send_event_sync(loop, send_event, {'type': 'session', 'data': active})
                        _last_state.update({'url': None, 'text_hash': None, 'count': 0})
                        break
                    time.sleep(0.3)
                else:
                    return
                continue

            if _replay_saved_actions_once():
                continue

            if _is_search_results_page(page.url) and not agent_context:
                clicked_selector = _click_first_search_result(page)
                if clicked_selector:
                    _record_playwright_action({'action': 'click', 'selector': clicked_selector}, 'success')
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
            if agent_context:
                form_fields = []
            if _is_search_results_page(page.url):
                form_fields = []
            has_captcha = _detect_captcha(page)
            _send_event_sync(loop, send_event, {'type': 'screenshot', 'data': screenshot_base64})

            # --- Stuck-loop detection ---
            _cur_text_hash = hash(page_text[:500])
            if page.url == _last_state['url'] and _cur_text_hash == _last_state['text_hash']:
                _last_state['count'] += 1
            else:
                _last_state.update({'url': page.url, 'text_hash': _cur_text_hash, 'count': 1})
            # Force extract after 2 iterations on e-commerce
            if _last_state['count'] >= 2 and agent_context and contextual_target_count:
                is_ecom = any(d in page.url.lower() for d in ['amazon.com', 'aliexpress.com', 'ebay.com'])
                if is_ecom:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Forcing extract after stuck...'})
                    try:
                        pd = page.evaluate("""() => {
                            const clean = (text) => (text || '').replace(/\\s+/g, ' ').trim();
                            let title = '';
                            const titleEl = document.querySelector('#productTitle, .product-title, h1[class*="title"], h1');
                            if (titleEl) title = clean(titleEl.textContent);
                            if (!title || title.length > 200) title = 'Produit trouve';
                            let price = '';
                            for (const selector of ['.a-price .a-offscreen', 'span.a-price > span.a-offscreen', '#priceblock_ourprice']) {
                                const el = document.querySelector(selector);
                                if (el) {
                                    const text = clean(el.textContent);
                                    if (text.match(/[$€£¥]|\\d+[.,]\\d{2}/)) {
                                        price = text;
                                        break;
                                    }
                                }
                            }
                            if (!price) price = 'Prix non affiche';
                            return {product_title: title, price: price, url: window.location.href, availability: 'Disponibilite inconnue'};
                        }""")
                    except:
                        pd = {'product_title': 'Extraction echouee', 'price': 'N/A', 'url': page.url, 'availability': 'Inconnu'}
                    sn = contextual_sources[contextual_source_index - 1]['name'] if 0 < contextual_source_index <= len(contextual_sources) else 'Source'
                    contextual_research_results.append({'source': sn, 'url': page.url, 'data': pd})
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': 'EXTRACT', 'args': f'{sn}: forced', 'status': 'done'})
                    save_session(task, 'DATA_EXTRACT', str(pd), status='done', namespace=memory_namespace)
                    _last_state['count'] = 0
                    if len(contextual_research_results) >= contextual_target_count:
                        report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                        _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                        save_session(task, 'DONE', report, status='done', namespace=memory_namespace)
                        return
                    if _go_to_next_contextual_source(page, 'Finished'):
                        continue
                    report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                    _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                    save_session(task, 'DONE', report, status='done', namespace=memory_namespace)
                    return
            
            if _last_state['count'] > 3:
                _last_state['count'] = 0
                if agent_context and contextual_target_count:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': f'Stuck on {page.url}, skipping to next source...'})
                    # Skip current source and move to next
                    if _go_to_next_contextual_source(page, 'Current source appears stuck'):
                        continue
                    # No more sources, generate report with what we have
                    if contextual_research_results:
                        report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                        _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                        save_session(task, 'DONE', report, status='done', namespace=memory_namespace)
                        return
                    # No results at all
                    _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': 'No data collected', 'sources': []}})
                    return
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
                        if _switch_to_page_action(user_feedback):
                            break
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
                if show_browser:
                    question = 'CAPTCHA detected. Anti-bot mode is already enabled. Please solve the CAPTCHA in the visible browser window, then tell me "done".'
                else:
                    question = 'CAPTCHA detected. Anti-bot mode is already enabled, but this site still requires human verification. Please solve the CAPTCHA and provide the verification code.'
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
                        if _switch_to_page_action(user_feedback):
                            break
                        _persist({'type': 'user_feedback', 'message': user_feedback})
                        _send_event_sync(loop, send_event, {'type': 'log', 'message': f'User solved CAPTCHA: {user_feedback}'})
                        break
                    time.sleep(0.3)
                continue  # next iteration will handle after CAPTCHA

            # --- Direct form detection: ask and fill field by field ---
            already_asked = any(
                item.get('type') == 'agent_question' for item in conversation_history
            )
            if _skip_form_once:
                _skip_form_once = False
                form_fields = []
            if form_fields and not has_user_answer:
                redirect_requested = False
                for field in form_fields:
                    label = field['label']
                    selector = field['selector']
                    field_type = field.get('type', '')
                    value = None
                    # Build question â€” for select, list available options
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
                                if _switch_to_page_action(val):
                                    redirect_requested = True
                                    answered = True
                                    break
                                _persist({'type': 'user_feedback', 'message': val})
                                value = val.strip()
                                answered = True
                                break
                            time.sleep(0.3)
                        if not answered:
                            return
                        if redirect_requested:
                            break
                        # Try to fill immediately
                        fill_action = {'action': 'fill_form', 'fields': [{'selector': selector, 'value': value}]}
                        try:
                            _fill_form_field(page, selector, value)
                            _record_playwright_action(fill_action, 'success')
                            _send_event_sync(loop, send_event, {'type': 'step', 'name': 'FILL', 'args': f'{label} = {value}', 'status': 'done'})
                            break  # field filled successfully, move to next
                        except Exception as fe:
                            err_msg = f'Impossible de remplir "{label}" : {fe}'
                            _record_playwright_action(fill_action, 'error', str(fe))
                            _send_event_sync(loop, send_event, {'type': 'log', 'message': err_msg})
                            # Loop back to re-ask with error message
                            question = f'Erreur sur "{label}" ({fe}). Entrez une nouvelle valeur :'
                            value = None  # force re-ask
                    if redirect_requested:
                        break
                if redirect_requested:
                    conversation_history[:] = [e for e in conversation_history if e.get('type') not in ('agent_question', 'user_feedback')]
                    _skip_form_once = True
                    continue
                # All fields filled â€” submit
                _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SUBMIT', 'args': 'submitting form', 'status': 'running'})
                try:
                    submit_selector = 'input[type=submit], button[type=submit]'
                    page.click(submit_selector, timeout=5000)
                    page.wait_for_load_state('domcontentloaded', timeout=8000)
                    _record_playwright_action({'action': 'submit', 'selector': submit_selector}, 'success')
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SUBMIT', 'args': 'Form submitted', 'status': 'done'})
                except Exception as submit_err:
                    _record_playwright_action(
                        {'action': 'submit', 'selector': 'input[type=submit], button[type=submit]'},
                        'error',
                        str(submit_err),
                    )
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SUBMIT', 'args': 'no submit button found', 'status': 'done'})
                save_session(task, 'FORM_FILL', 'form filled field by field', status='done', namespace=memory_namespace)
                # Clear form Q&A from history so next iteration doesn't re-fill
                conversation_history[:] = [e for e in conversation_history if e.get('type') not in ('agent_question', 'user_feedback')]
                continue
            # --- No form: use LLM ---
            _send_event_sync(loop, send_event, {'type': 'thinking', 'message': f'Analyzing page â€” iteration {iteration + 1}/20.'})
            memory_ctx = get_memory_context(task, namespace=memory_namespace)
            if contextual_research_results:
                memory_ctx += "\n\n## CONTEXTUAL RESEARCH COLLECTED SO FAR:\n"
                for item in contextual_research_results[-8:]:
                    memory_ctx += f"- {item.get('source', 'Source')}: {item.get('url', '')} => {json.dumps(item.get('data', {}), ensure_ascii=False)[:500]}\n"
            
            # Force extract on e-commerce PRODUCT pages for contextual agents
            if agent_context and contextual_target_count:
                is_ecommerce = any(domain in page.url.lower() for domain in ['amazon.com', 'aliexpress.com', 'ebay.com', 'alibaba.com', 'etsy.com', 'walmart.com'])
                is_product_page = (
                    '/dp/' in page.url or 
                    '/item/' in page.url or 
                    '/itm/' in page.url or 
                    'product' in page.url.lower() or
                    '/gp/product/' in page.url or
                    '/gp/aw/d/' in page.url or
                    'amazon.com/Apple' in page.url or
                    'amazon.com/apple' in page.url or
                    'ebay.com/p/' in page.url or
                    'aliexpress.com/item/' in page.url
                )
                if not is_product_page:
                    try:
                        has_product_title = page.locator('#productTitle, .product-title, h1[id*="title"]').count() > 0
                        if has_product_title:
                            is_product_page = True
                    except Exception:
                        pass
                is_github_search = 'github.com/search' in page.url.lower()
                # Only force extract if we're on an actual product page (not search results)
                if is_ecommerce and is_product_page:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': 'E-commerce product page detected, extracting product data...'})
                    
                    # Extract real product data using JavaScript
                    try:
                        product_data = page.evaluate("""() => {
                            // Try to find product title
                            const title = document.querySelector('#productTitle, .product-title, h1[class*="title"], [data-testid="product-title"]')?.textContent?.trim() || 
                                         document.querySelector('h1')?.textContent?.trim() || 'Product title not found';
                            
                            // Try to find price - improved selectors for Amazon
                            const priceSelectors = [
                                '.a-price .a-offscreen',
                                '.a-price-whole',
                                'span.a-price > span.a-offscreen',
                                'span[class*="priceToPay"] .a-offscreen',
                                '#priceblock_ourprice',
                                '#priceblock_dealprice',
                                '.a-color-price',
                                '[data-a-color="price"]',
                                '[class*="price"]',
                                '[data-testid*="price"]',
                                '.price',
                                'span[class*="Price"]'
                            ];
                            let price = 'Price not displayed';
                            for (const selector of priceSelectors) {
                                const el = document.querySelector(selector);
                                if (el) {
                                    const text = el.textContent.trim();
                                    // Check if contains currency symbol or number
                                    if (text.match(/[$]|\\d+[.,]\\d+/)) {
                                        price = text;
                                        break;
                                    }
                                }
                            }
                            
                            // Fallback: search for price in page text
                            if (price === 'Price not displayed') {
                                const priceMatch = document.body.textContent.match(/[$]\\s*\\d+[.,]\\d{2}/);
                                if (priceMatch) {
                                    price = priceMatch[0];
                                }
                            }
                            
                            // Try to find availability
                            const availText = document.body.textContent.toLowerCase();
                            let availability = 'Unknown';
                            if (availText.includes('in stock') || availText.includes('available')) {
                                availability = 'In stock';
                            } else if (availText.includes('out of stock') || availText.includes('unavailable')) {
                                availability = 'Out of stock';
                            }
                            
                            return {
                                product_title: title,
                                price: price,
                                url: window.location.href,
                                availability: availability
                            };
                        }""")
                    except Exception:
                        product_data = {
                            'product_title': 'Extraction failed',
                            'price': 'N/A',
                            'url': page.url,
                            'availability': 'Unknown'
                        }
                    
                    action = {'action': 'extract', 'data': product_data}
                    name = 'extract'
                    name_upper = 'EXTRACT'
                    args = json.dumps(product_data, ensure_ascii=False)
                elif is_github_search:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': 'GitHub search page detected, opening visible user profiles for targeted extraction...'})
                    try:
                        profile_urls = page.evaluate("""() => {
                            const seen = new Set();
                            const urls = [];
                            const anchors = Array.from(document.querySelectorAll('a[href^="/"][href]:not([href*="?"]):not([href*="/search"])'));
                            for (const anchor of anchors) {
                                const href = anchor.getAttribute('href') || '';
                                const parts = href.split('/').filter(Boolean);
                                if (parts.length !== 1) continue;
                                const username = parts[0];
                                if (!username || seen.has(username) || ['features', 'topics', 'collections', 'trending', 'marketplace'].includes(username.toLowerCase())) continue;
                                seen.add(username);
                                urls.push(`https://github.com/${username}`);
                                if (urls.length >= 6) break;
                            }
                            return urls;
                        }""")
                        profiles = []
                        for profile_url in profile_urls[:6]:
                            try:
                                page.goto(profile_url, timeout=15000)
                                page.wait_for_load_state('domcontentloaded', timeout=8000)
                                profile = page.evaluate("""() => {
                                    const textOf = (selector) => document.querySelector(selector)?.textContent?.replace(/\\s+/g, ' ').trim() || '';
                                    const attrOf = (selector, attr) => document.querySelector(selector)?.getAttribute(attr) || '';
                                    const mailHref = attrOf('a[href^="mailto:"]', 'href');
                                    const email = mailHref ? mailHref.replace(/^mailto:/, '').trim() : '';
                                    const name = textOf('.p-name') || textOf('[itemprop="name"]') || textOf('h1 span') || '';
                                    const username = textOf('.p-nickname') || location.pathname.split('/').filter(Boolean)[0] || '';
                                    const bio = textOf('.p-note') || textOf('[data-bio-text]') || '';
                                    const company = textOf('[itemprop="worksFor"]') || '';
                                    const locationText = textOf('[itemprop="homeLocation"]') || '';
                                    const website = attrOf('[itemprop="url"]', 'href') || '';
                                    const repos = Array.from(document.querySelectorAll('a[itemprop="name codeRepository"], .pinned-item-list-item-content a'))
                                        .map(a => a.textContent.replace(/\\s+/g, ' ').trim())
                                        .filter(Boolean)
                                        .slice(0, 5);
                                    const domainParts = [bio, company, repos.length ? `Repos: ${repos.join(', ')}` : ''].filter(Boolean);
                                    return {
                                        username,
                                        name: name || username,
                                        profile: location.href,
                                        email,
                                        domain_work: domainParts.join(' | ') || 'Non visible sur le profil GitHub',
                                        location: locationText,
                                        website,
                                    };
                                }""")
                                profiles.append(profile)
                            except Exception as profile_err:
                                profiles.append({
                                    'name': profile_url.rstrip('/').split('/')[-1],
                                    'username': profile_url.rstrip('/').split('/')[-1],
                                    'profile': profile_url,
                                    'email': '',
                                    'domain_work': f'Profil trouve, extraction detaillee echouee: {profile_err}',
                                })
                        if not profiles:
                            text = page.evaluate("""() => document.body.innerText.replace(/\\s+/g, ' ').trim().slice(0, 1800)""")
                            github_data = {'search_results': [], 'page_summary': text, 'url': page.url}
                        else:
                            github_data = {'search_results': profiles, 'url': page.url}
                    except Exception as extract_err:
                        try:
                            page.goto(_source_search_url(task, 'GitHub', 'github.com'), timeout=15000)
                            page.wait_for_load_state('domcontentloaded', timeout=8000)
                            text = page.evaluate("""() => document.body.innerText.replace(/\\s+/g, ' ').trim().slice(0, 1800)""")
                            github_data = {'search_results': [], 'page_summary': text, 'url': page.url, 'error': f'GitHub extraction failed: {extract_err}'}
                        except Exception:
                            github_data = {
                                'search_results': [],
                                'url': page.url,
                                'error': f'GitHub extraction failed: {extract_err}',
                            }
                    action = {'action': 'extract', 'data': github_data}
                    name = 'extract'
                    name_upper = 'EXTRACT'
                    args = json.dumps(github_data, ensure_ascii=False)
                elif any(domain in page.url.lower() for domain in ['linkedin.com/search', 'reddit.com/search', 'youtube.com/results', 'x.com/search']):
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Contextual source results page detected, extracting visible links...'})
                    try:
                        source_data = page.evaluate("""() => {
                            const emailRegex = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}/ig;
                            const seen = new Set();
                            const results = [];
                            for (const anchor of Array.from(document.querySelectorAll('a[href]'))) {
                                const href = anchor.href;
                                const title = (anchor.innerText || anchor.getAttribute('aria-label') || '').replace(/\\s+/g, ' ').trim();
                                if (!href || !title || seen.has(href)) continue;
                                if (title.length < 3) continue;
                                seen.add(href);
                                const card = anchor.closest('article, li, div') || anchor.parentElement;
                                const description = (card?.innerText || title).replace(/\\s+/g, ' ').trim().slice(0, 500);
                                const emailMatch = description.match(emailRegex);
                                results.push({ title, url: href, description, email: emailMatch ? emailMatch[0] : '' });
                                if (results.length >= 10) break;
                            }
                            return { search_results: results, url: window.location.href };
                        }""")
                    except Exception as extract_err:
                        source_data = {
                            'search_results': [],
                            'url': page.url,
                            'error': f'Source extraction failed: {extract_err}',
                        }
                    action = {'action': 'extract', 'data': source_data}
                    name = 'extract'
                    name_upper = 'EXTRACT'
                    args = json.dumps(source_data, ensure_ascii=False)
                else:
                    system_prompt = _format_action_prompt(
                        task=task,
                        page_url=page.url,
                        page_text=page_text,
                        iteration=iteration + 1,
                        conversation_history=conversation_history,
                        form_fields=form_fields,
                        memory_context=memory_ctx,
                        agent_context=agent_context,
                        agent_name=agent_name,
                        agent_description=agent_description,
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
            else:
                system_prompt = _format_action_prompt(
                    task=task,
                    page_url=page.url,
                    page_text=page_text,
                    iteration=iteration + 1,
                    conversation_history=conversation_history,
                    form_fields=form_fields,
                    memory_context=memory_ctx,
                    agent_context=agent_context,
                    agent_name=agent_name,
                    agent_description=agent_description,
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

            if agent_context and name == 'ask_user':
                question_text = (action.get('question') or '').lower()
                blocking_question = any(word in question_text for word in [
                    'captcha', 'login', 'password', 'mot de passe', 'credential',
                    'identifiant', 'connexion', 'verification code', 'code de verification',
                ])
                if not blocking_question:
                    _send_event_sync(loop, send_event, {
                        'type': 'log',
                        'message': 'Contextual agent asked for guidance; extracting visible page data instead.'
                    })
                    try:
                        visible_data = page.evaluate("""() => {
                            const emailRegex = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}/ig;
                            const seen = new Set();
                            const results = [];
                            for (const anchor of Array.from(document.querySelectorAll('a[href]'))) {
                                const href = anchor.href;
                                const title = (anchor.innerText || anchor.getAttribute('aria-label') || '').replace(/\\s+/g, ' ').trim();
                                if (!href || !title || seen.has(href)) continue;
                                if (title.length < 3) continue;
                                seen.add(href);
                                const card = anchor.closest('article, li, div') || anchor.parentElement;
                                const description = (card?.innerText || title).replace(/\\s+/g, ' ').trim().slice(0, 500);
                                const emailMatch = description.match(emailRegex);
                                results.push({ title, url: href, description, email: emailMatch ? emailMatch[0] : '' });
                                if (results.length >= 10) break;
                            }
                            const pageText = document.body.innerText.replace(/\\s+/g, ' ').trim().slice(0, 1800);
                            return { search_results: results, page_summary: pageText, url: window.location.href };
                        }""")
                    except Exception as extract_err:
                        visible_data = {
                            'search_results': [],
                            'url': page.url,
                            'error': f'Visible extraction failed: {extract_err}',
                        }
                    action = {'action': 'extract', 'data': visible_data}
                    name = 'extract'
                    name_upper = 'EXTRACT'
                    args = json.dumps(visible_data, ensure_ascii=False)

            action_text = (name + ' ' + args).lower() if name else args.lower()
            needs_confirm = any(word in action_text for word in {'submit', 'purchase', 'delete', 'send', 'confirm', 'pay'})
            if needs_confirm:
                _send_event_sync(loop, send_event, {
                    'type': 'safety',
                    'explanation': f'The agent is about to: {name_upper} â€” {args}. '
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
                            'message': f'Safety timeout â€” skipping action: {name_upper}'
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
                    success, error_message = _execute_playwright_action(action)
                    _record_playwright_action(action, 'success' if success else 'error', error_message)
                    if not success:
                        _send_event_sync(loop, send_event, {
                            'type': 'log',
                            'message': f'Click failed on "{action.get("selector", "")}": {error_message}'
                        })
                elif name == 'type':
                    success, error_message = _execute_playwright_action(action)
                    _record_playwright_action(action, 'success' if success else 'error', error_message)
                    if not success:
                        _send_event_sync(loop, send_event, {
                            'type': 'log',
                            'message': f'Type failed on "{action.get("selector", "")}": {error_message}'
                        })
                elif name == 'fill_form':
                    success, error_message = _execute_playwright_action(action)
                    _record_playwright_action(action, 'success' if success else 'error', error_message)
                    if not success:
                        _send_event_sync(loop, send_event, {
                            'type': 'log',
                            'message': f'Fill form failed: {error_message}'
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
                            if _switch_to_page_action(user_feedback):
                                break
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
                    extracted_data = action.get('data', {})
                    if agent_context and contextual_target_count:
                        source_name = contextual_sources[contextual_source_index - 1]['name'] if 0 < contextual_source_index <= len(contextual_sources) else 'Recherche web'
                        contextual_research_results.append({
                            'source': source_name,
                            'url': page.url,
                            'data': extracted_data,
                        })
                        _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': f'{source_name}: {args}', 'status': 'done'})
                        save_session(task, 'DATA_EXTRACT', str(extracted_data), status='done', namespace=memory_namespace)

                        if len(contextual_research_results) >= contextual_target_count:
                            report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                            _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                            save_session(task, 'DONE', report, status='done', namespace=memory_namespace)
                            return

                        if _go_to_next_contextual_source(page, 'Finished extracting current source'):
                            continue

                        report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                        _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                        save_session(task, 'DONE', report, status='done', namespace=memory_namespace)
                        return

                    _send_event_sync(loop, send_event, {'type': 'result', 'data': extracted_data})
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'done'})
                    save_session(task, 'DATA_EXTRACT', str(extracted_data), status='done', namespace=memory_namespace)
                    # Wait silently for next instruction typed in the input bar
                    deadline = time.time() + 600
                    while time.time() < deadline:
                        if abort_event and abort_event.is_set():
                            return
                        if feedback_queue:
                            nxt = feedback_queue.pop(0)
                            task = nxt.strip()
                            ensure_topic_session(task, namespace=memory_namespace, topic=agent_name or agent_description or None, allow_topic_switch=not bool(agent_id))
                            _persist({'type': 'user_feedback', 'message': task})
                            append_conversation('user', task, task=task, namespace=memory_namespace)
                            conversation_history[:] = list(get_conversation_history(100, namespace=memory_namespace))
                            active = get_active_session(memory_namespace)
                            if active:
                                _send_event_sync(loop, send_event, {'type': 'session', 'data': active})
                            _last_state.update({'url': None, 'text_hash': None, 'count': 0})
                            break
                        time.sleep(0.5)
                    else:
                        return
                    continue
                elif name == 'done':
                    if agent_context and contextual_source_index < len(contextual_sources):
                        if contextual_research_results:
                            _go_to_next_contextual_source(page, 'Model tried to finish before all contextual sources were checked')
                            continue
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': name_upper, 'args': args, 'status': 'done'})
                    if agent_context and contextual_research_results:
                        report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                        _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                    save_session(task, 'DONE', args, status='done', namespace=memory_namespace)
                    # Wait silently for next instruction typed in the input bar
                    deadline = time.time() + 600
                    while time.time() < deadline:
                        if abort_event and abort_event.is_set():
                            return
                        if feedback_queue:
                            nxt = feedback_queue.pop(0)
                            task = nxt.strip()
                            ensure_topic_session(task, namespace=memory_namespace, topic=agent_name or agent_description or None, allow_topic_switch=not bool(agent_id))
                            _persist({'type': 'user_feedback', 'message': task})
                            append_conversation('user', task, task=task, namespace=memory_namespace)
                            conversation_history[:] = list(get_conversation_history(100, namespace=memory_namespace))
                            active = get_active_session(memory_namespace)
                            if active:
                                _send_event_sync(loop, send_event, {'type': 'session', 'data': active})
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
        _apply_anti_bot_page_settings(page)  # always apply JS shim

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


def _build_contextual_search_query(
    task: str,
    agent_context: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_description: Optional[str] = None,
) -> str:
    if not agent_context:
        return task
    base_query = _extract_search_query(task)
    context_words = " ".join(
        part.strip()
        for part in [agent_name or "", agent_description or "", agent_context or ""]
        if part and part.strip()
    ).lower()

    terms: List[str] = []
    if re.search(r'\b(concurrent|concurrents|concurrence|competitor|competitors|comparatif|compare|marche|marché|market)\b', context_words):
        terms.extend(['concurrents', 'comparatif', 'prix'])
    if re.search(r'\b(avis|review|reviews)\b', context_words):
        terms.append('avis')

    known_sources = [
        ('amazon', 'Amazon'),
        ('alibaba', 'Alibaba'),
        ('aliexpress', 'AliExpress'),
        ('ebay', 'eBay'),
        ('etsy', 'Etsy'),
        ('google shopping', 'Google Shopping'),
        ('walmart', 'Walmart'),
    ]
    for needle, label in known_sources:
        if needle in context_words:
            terms.append(label)

    if not terms:
        profile_words = " ".join([agent_name or "", agent_description or ""]).lower()
        blocked = {
            'avec', 'pour', 'dans', 'plus', 'cette', 'agent', 'assistant',
            'rapport', 'final', 'faire', 'donne', 'donner', 'cherche',
        }
        for word in re.findall(r'[\w-]{4,}', profile_words):
            if word not in blocked and word not in terms:
                terms.append(word)
            if len(terms) >= 4:
                break

    deduped = []
    for term in terms:
        if term.lower() not in [item.lower() for item in deduped]:
            deduped.append(term)
        if len(deduped) >= 7:
            break

    suffix = " ".join(deduped)
    return f"{base_query} {suffix}".strip()


def _get_contextual_sources(agent_context: Optional[str]) -> List[Dict[str, str]]:
    if not agent_context:
        return []
    context = agent_context.lower()
    sources: List[Dict[str, str]] = []

    def add_source(label: str, domain: str) -> None:
        clean_domain = domain.lower().strip().rstrip('/.,;:)')
        if clean_domain.startswith('www.'):
            clean_domain = clean_domain[4:]
        if not clean_domain or clean_domain in {'http', 'https'}:
            return
        if any(item['domain'].lower() == clean_domain for item in sources):
            return
        sources.append({'name': label, 'domain': clean_domain})

    for raw_url in re.findall(r'https?://[^\s)\]}>,"\']+', agent_context, flags=re.IGNORECASE):
        parsed = urlparse(raw_url)
        domain = parsed.netloc.lower()
        if domain:
            label = domain[4:] if domain.startswith('www.') else domain
            add_source(label.split('.')[0].title(), domain)

    for raw_domain in re.findall(r'\b(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+\b', agent_context, flags=re.IGNORECASE):
        if raw_domain.lower() in {'127.0.0.1'}:
            continue
        label_domain = raw_domain[4:] if raw_domain.lower().startswith('www.') else raw_domain
        add_source(label_domain.split('.')[0].title(), label_domain)

    candidates = [
        ('amazon', 'Amazon', 'amazon.com'),
        ('alibaba', 'Alibaba', 'alibaba.com'),
        ('aliexpress', 'AliExpress', 'aliexpress.com'),
        ('ebay', 'eBay', 'ebay.com'),
        ('etsy', 'Etsy', 'etsy.com'),
        ('github', 'GitHub', 'github.com'),
        ('git hub', 'GitHub', 'github.com'),
        ('gitgub', 'GitHub', 'github.com'),
        ('githib', 'GitHub', 'github.com'),
        ('linkedin', 'LinkedIn', 'linkedin.com'),
        ('linked in', 'LinkedIn', 'linkedin.com'),
        ('youtube', 'YouTube', 'youtube.com'),
        ('reddit', 'Reddit', 'reddit.com'),
        ('twitter', 'X', 'x.com'),
        ('x.com', 'X', 'x.com'),
        ('google shopping', 'Google Shopping', 'shopping.google.com'),
        ('walmart', 'Walmart', 'walmart.com'),
    ]
    for needle, label, domain in candidates:
        if needle in context:
            add_source(label, domain)

    if sources:
        return sources[:6]
    if re.search(r'\b(concurrent|concurrents|concurrence|competitor|competitors|comparatif|marche|marché|market|produit|product)\b', context):
        return [
            {'name': 'Amazon', 'domain': 'amazon.com'},
            {'name': 'Alibaba', 'domain': 'alibaba.com'},
            {'name': 'AliExpress', 'domain': 'aliexpress.com'},
            {'name': 'eBay', 'domain': 'ebay.com'},
            {'name': 'Google Shopping', 'domain': 'shopping.google.com'},
        ]
    return []


def _format_contextual_research_report(
    task: str,
    agent_name: Optional[str],
    results: List[Dict[str, Any]],
    agent_description: Optional[str] = None,
    agent_context: Optional[str] = None,
) -> str:
    subject = _extract_search_query(task)
    instructions = "\n".join([agent_description or "", agent_context or ""]).lower()
    is_people_search = any(word in instructions for word in [
        'utilisateur', 'etulisateur', 'user', 'profil', 'profile', 'candidat',
        'chercheur', 'developer', 'developpeur', 'développeur',
    ])

    records = []
    sources = []
    for item in results:
        source = item.get('source', 'Source')
        url = item.get('url', '')
        if url:
            sources.append((source, url))
        data = item.get('data', {})
        if isinstance(data, dict) and isinstance(data.get('search_results'), list):
            for result in data.get('search_results', [])[:5]:
                records.append({
                    'source': source,
                    'title': result.get('title') or result.get('name') or result.get('username') or 'Resultat sans titre',
                    'url': result.get('url') or result.get('link') or result.get('profile') or url,
                    'price': result.get('price') or result.get('prix') or '',
                    'description': result.get('description') or result.get('snippet') or result.get('domain_work') or '',
                    'email': result.get('email') or '',
                    'username': result.get('username') or '',
                    'location': result.get('location') or '',
                    'website': result.get('website') or '',
                })
        elif isinstance(data, dict):
            title = data.get('product_title') or data.get('title') or data.get('name')
            if title or data.get('price') or data.get('url'):
                records.append({
                    'source': source,
                    'title': title or 'Produit trouve',
                    'url': data.get('url') or url,
                    'price': data.get('price') or data.get('prix') or '',
                    'description': data.get('description') or data.get('availability') or '',
                    'email': data.get('email') or '',
                    'username': data.get('username') or '',
                    'location': data.get('location') or '',
                    'website': data.get('website') or '',
                })
    products = records

    def wants(*words: str) -> bool:
        return any(word in instructions for word in words)

    sections = ['sources', 'produits']
    if wants('prix', 'price', 'tarif') and not is_people_search:
        sections.append('prix')
    if wants('comparaison', 'compare', 'comparatif', 'concurrent', 'competitor') and not is_people_search:
        sections.append('comparaison')
    if wants('avis', 'review') and not is_people_search:
        sections.append('avis')
    if wants('recommandation', 'recommendation', 'conseil', 'opportunit') and not is_people_search:
        sections.append('recommandations')
    if wants('conclusion', 'resume', 'résumé', 'rapport'):
        sections.append('conclusion')
    if len(sections) == 2 and is_people_search:
        sections.append('conclusion')
    elif len(sections) == 2:
        sections.extend(['prix', 'recommandations'])

    ordered = []
    for section in sections:
        if section not in ordered:
            ordered.append(section)
    ordered = ordered[:6]

    source_count = len(set(source for source, _ in sources)) or len(results)
    lines = [
        f"# {agent_name or 'Rapport'}",
        "",
        f"Sujet: **{subject}**",
        f"J'ai trouve {len(products)} resultat(s) sur {source_count} source(s).",
        "",
    ]

    for section in ordered:
        if section == 'sources':
            lines.append('## Sources consultees')
            if sources:
                for source, url in sources[:6]:
                    lines.append(f"- **{source}**: {url}")
            else:
                lines.append('- Aucune source exploitable trouvee.')
            lines.append('')

        elif section == 'produits':
            lines.append('## Profils trouves' if is_people_search else '## Resultats principaux')
            if products:
                for product_item in products[:8]:
                    label = product_item.get('title') or product_item.get('username') or 'Resultat'
                    lines.append(f"- **{product_item['source']}** - {label}")
                    if product_item.get('username'):
                        lines.append(f"  Username: {product_item['username']}")
                    if product_item['price']:
                        lines.append(f"  Prix: {product_item['price']}")
                    if product_item.get('email'):
                        lines.append(f"  Email: {product_item['email']}")
                    elif is_people_search:
                        lines.append("  Email: Non visible publiquement")
                    if product_item.get('description'):
                        lines.append(f"  Domaine/infos: {product_item['description'][:220]}")
                    if product_item.get('location'):
                        lines.append(f"  Localisation: {product_item['location']}")
                    if product_item.get('website'):
                        lines.append(f"  Site: {product_item['website']}")
                    if product_item['url']:
                        lines.append(f"  Profil: {product_item['url']}" if is_people_search else f"  Lien: {product_item['url']}")
            else:
                lines.append("- Aucun resultat cible clair n'a ete extrait.")
            lines.append('')

        elif section == 'prix':
            lines.append('## Prix')
            priced = [item for item in products if item.get('price')]
            if priced:
                for item in priced[:8]:
                    lines.append(f"- **{item['source']}**: {item['price']} - {item['title']}")
            else:
                lines.append('- Prix non affiches dans les donnees collectees. Ouvre les liens pour verifier les prix actuels.')
            lines.append('')

        elif section == 'comparaison':
            lines.append('## Comparaison rapide')
            if products:
                for item in products[:5]:
                    detail = item.get('description') or 'Details a verifier sur la page source.'
                    lines.append(f"- **{item['source']}**: {item['title']} - {detail[:160]}")
            else:
                lines.append('- Pas assez de donnees pour comparer correctement.')
            lines.append('')

        elif section == 'avis':
            lines.append('## Avis / signaux')
            lines.append('- Les avis clients ne sont pas toujours visibles dans les resultats extraits.')
            lines.append('- Verifie les notes et commentaires directement sur les liens sources avant decision.')
            lines.append('')

        elif section == 'recommandations':
            lines.append('## Recommandations simples')
            if products:
                lines.append("- Compare d'abord les prix visibles et les frais de livraison.")
                lines.append('- Priorise les pages avec titre clair, vendeur identifiable et avis clients.')
                lines.append('- Si un prix manque, ouvre le lien avant de conclure.')
            else:
                lines.append('- Essaie un nom de produit plus precis ou une orthographe alternative.')
            lines.append('')

        elif section == 'conclusion':
            lines.append('## Conclusion')
            if products and is_people_search:
                lines.append("Les profils ci-dessus correspondent aux resultats visibles sur les sources configurees. Les emails ne sont inclus que lorsqu'ils sont publics sur le profil.")
            elif products:
                lines.append('La recherche a trouve plusieurs pistes utiles. Le meilleur choix depend surtout du prix final, des avis et de la livraison.')
            else:
                lines.append("La recherche n'a pas trouve assez de donnees fiables pour conclure. Il faut relancer avec un terme plus precis.")
            lines.append('')

    return "\n".join(lines).strip()

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
    agent_context: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_description: Optional[str] = None,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> MCPContext:
    skip_anti_bot = False
    memory_namespace = f'agent:{agent_id}' if agent_id else 'default'
    agent_topic = agent_name or agent_description or None
    active_session = ensure_topic_session(
        task,
        namespace=memory_namespace,
        topic=agent_topic,
        allow_topic_switch=not bool(agent_id),
    )
    append_conversation('user', task, task=task, namespace=memory_namespace)

    memory_ctx = get_memory_context(task, n_sessions=10, n_conv=30, namespace=memory_namespace)
    if memory_ctx:
        await send_event({'type': 'system_context', 'message': memory_ctx})
    if agent_context:
        await send_event({
            'type': 'agent_context',
            'agent': {
                'name': agent_name or '',
                'description': agent_description or '',
            },
        })
    await send_event({
        'type': 'session',
        'data': {
            'id': active_session.get('id', ''),
            'topic': active_session.get('topic', task),
            'summary': active_session.get('summary', ''),
        }
    })

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

    contextual_search_query = _build_contextual_search_query(
        task,
        agent_context=agent_context,
        agent_name=agent_name,
        agent_description=agent_description,
    )
    contextual_sources = _get_contextual_sources(agent_context)
    if agent_context and contextual_sources:
        first_source = contextual_sources[0]
        start_url = _source_search_url(task, first_source['name'], first_source['domain'])
    else:
        start_url = await _get_start_url(contextual_search_query)
    auto_skip = _task_requires_skip_anti_bot(task)
    if auto_skip:
        await send_event({'type': 'log', 'message': 'Anti-bot-sensitive task detected.'})
    await send_event({'type': 'log', 'message': 'Anti-bot protections are always enabled for browser sessions.'})

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
            agent_context,
            agent_name,
            agent_description,
            contextual_search_query,
            memory_namespace,
            agent_id,
            user_id,
        )
    except Exception as exc:
        await send_event({'type': 'error', 'message': str(exc)})
        await send_event({'type': 'done'})
    return mcp_context
