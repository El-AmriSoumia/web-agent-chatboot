import base64
import io
import random
import time
from typing import Any, Dict, Optional
from PIL import Image

from backend.mcp import MCPContext


class RPAController:
    def __init__(self, page: Any, mcp_context: MCPContext, loop: Any, send_event: Any):
        self.page = page
        self.mcp_context = mcp_context
        self.loop = loop
        self.send_event = send_event

    def _send_event(self, data: Dict[str, Any]) -> None:
        try:
            future = self.loop.run_coroutine_threadsafe(self.send_event(data), self.loop)
            future.result(timeout=10)
        except Exception:
            pass

    def _log_action(self, action: str, details: Dict[str, Any]) -> None:
        self.mcp_context.add_action(action, details)
        self._send_event({'type': 'step', 'name': action.upper(), 'args': details, 'status': 'running'})

    def _human_type(self, selector: str, text: str) -> None:
        """Type text character by character with human-like delays."""
        try:
            element = self.page.locator(selector).first
            element.scroll_into_view_if_needed()
            element.click()
            element.fill('')  # Clear the field
            for char in text:
                self.page.keyboard.type(char)
                time.sleep(random.randint(40, 120) / 1000.0)
        except Exception as e:
            # Fallback to direct fill if human typing fails
            try:
                self.page.fill(selector, text)
            except Exception:
                raise e

    def navigate(self, url: str) -> str:
        self._log_action('navigate', {'url': url})
        try:
            self.page.goto(url, timeout=20000)
            self.page.wait_for_load_state('domcontentloaded', timeout=10000)
            self.mcp_context.update_state(current_url=self.page.url)
            self._send_event({'type': 'log', 'message': f'Navigated to {self.page.url}'})
            self._send_event({'type': 'step', 'name': 'NAVIGATE', 'args': url, 'status': 'done'})
            return f'Navigated to {self.page.url}'
        except Exception as exc:
            message = f'Navigation failed: {exc}'
            self.mcp_context.add_error(message, {'url': url})
            self._send_event({'type': 'error', 'message': message})
            return message

    def click(self, selector: str) -> str:
        self._log_action('click', {'selector': selector})
        try:
            element = self.page.locator(selector).first
            element.scroll_into_view_if_needed()
            element.click(timeout=5000)
            self._send_event({'type': 'log', 'message': f'Clicked selector {selector}'})
            self._send_event({'type': 'step', 'name': 'CLICK', 'args': selector, 'status': 'done'})
            return f'Clicked {selector}'
        except Exception:
            try:
                element = self.page.get_by_text(selector, exact=False).first
                element.scroll_into_view_if_needed()
                element.click(timeout=5000)
                self._send_event({'type': 'log', 'message': f'Clicked text {selector}'})
                self._send_event({'type': 'step', 'name': 'CLICK', 'args': selector, 'status': 'done'})
                return f'Clicked text {selector}'
            except Exception:
                try:
                    element = self.page.get_by_role('button', name=selector)
                    element.scroll_into_view_if_needed()
                    element.click(timeout=5000)
                    self._send_event({'type': 'log', 'message': f'Clicked role button {selector}'})
                    self._send_event({'type': 'step', 'name': 'CLICK', 'args': selector, 'status': 'done'})
                    return f'Clicked role button {selector}'
                except Exception as exc:
                    message = f'Click failed on "{selector}": {exc}'
                    self.mcp_context.add_error(message, {'selector': selector})
                    self._send_event({'type': 'error', 'message': message})
                    return message

    def type_text(self, selector: str, text: str) -> str:
        self._log_action('type', {'selector': selector, 'text': text})
        try:
            self._human_type(selector, text)
            self.page.keyboard.press('Enter')
            self.page.wait_for_load_state('domcontentloaded', timeout=8000)
            self._send_event({'type': 'log', 'message': f'Typed text into {selector}'})
            self._send_event({'type': 'step', 'name': 'TYPE', 'args': f'{selector} | {text}', 'status': 'done'})
            return f'Typed into {selector}'
        except Exception as exc:
            message = f'Type failed on "{selector}": {exc}'
            self.mcp_context.add_error(message, {'selector': selector, 'text': text})
            self._send_event({'type': 'error', 'message': message})
            return message

    def fill_form(self, fields: list) -> str:
        self._log_action('fill_form', {'fields': fields})
        try:
            for field in fields:
                selector = field.get('selector', '')
                value = field.get('value', '')
                self._human_type(selector, value)
                time.sleep(random.uniform(0.3, 0.7))  # Pause between fields
            self._send_event({'type': 'log', 'message': f'Filled form with {len(fields)} fields'})
            self._send_event({'type': 'step', 'name': 'FILL_FORM', 'args': f'{len(fields)} fields', 'status': 'done'})
            return f'Filled form with {len(fields)} fields'
        except Exception as exc:
            message = f'Fill form failed: {exc}'
            self.mcp_context.add_error(message, {'fields': fields})
            self._send_event({'type': 'error', 'message': message})
            return message

    def scroll(self, amount: int = 600) -> str:
        self._log_action('scroll', {'amount': amount})
        try:
            self.page.evaluate(f'window.scrollBy(0, {amount})')
            self._send_event({'type': 'log', 'message': f'Scrolled by {amount} pixels'})
            self._send_event({'type': 'step', 'name': 'SCROLL', 'args': amount, 'status': 'done'})
            return f'Scrolled by {amount}'
        except Exception as exc:
            message = f'Scroll failed: {exc}'
            self.mcp_context.add_error(message)
            self._send_event({'type': 'error', 'message': message})
            return message

    def get_page_text(self, max_chars: int = 3000) -> str:
        try:
            text = self.page.evaluate("""() => {
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

    def extract_text(self, max_chars: int = 3000) -> str:
        self._log_action('extract', {'max_chars': max_chars})
        try:
            text = self.page.evaluate("""() => {
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
            text = text[:max_chars]
            self.mcp_context.add_result('page_text', text)
            self._send_event({'type': 'result', 'data': {'page_text': text}})
            self._send_event({'type': 'step', 'name': 'EXTRACT', 'args': {'chars': len(text)}, 'status': 'done'})
            return text
        except Exception as exc:
            message = f'Extract failed: {exc}'
            self.mcp_context.add_error(message)
            self._send_event({'type': 'error', 'message': message})
            return ''

    def take_screenshot(self) -> str:
        self._log_action('screenshot', {})
        try:
            screenshot_bytes = self.page.screenshot(full_page=False)
            image = Image.open(io.BytesIO(screenshot_bytes))
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            self._send_event({'type': 'screenshot', 'data': encoded})
            self._send_event({'type': 'step', 'name': 'SCREENSHOT', 'status': 'done'})
            return encoded
        except Exception as exc:
            message = f'Screenshot failed: {exc}'
            self.mcp_context.add_error(message)
            self._send_event({'type': 'error', 'message': message})
            return ''

    def get_current_url(self) -> str:
        return getattr(self.page, 'url', '')
