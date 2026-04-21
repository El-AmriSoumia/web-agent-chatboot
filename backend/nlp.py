import asyncio
import json
import os
import re
from typing import Dict, List

from google import genai as genai_client
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

LANGCHAIN_GOOGLE_AVAILABLE = False
TASK_ANALYSIS_PARSER = None
TASK_ANALYSIS_PROMPT = None
try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    from langchain_core.output_parsers.pydantic import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_GOOGLE_AVAILABLE = True
except Exception:
    PydanticOutputParser = None
    PromptTemplate = None


class TaskAnalysis(BaseModel):
    intent: str
    entity: str
    subtasks: List[str]


if PydanticOutputParser and PromptTemplate:
    TASK_ANALYSIS_PARSER = PydanticOutputParser(pydantic_object=TaskAnalysis)
    TASK_ANALYSIS_PROMPT = PromptTemplate.from_template(
        """You are a web task analyzer.
Analyze this task and return only a raw JSON object that matches the schema below.

Task: "{task}"

{format_instructions}

The JSON must be valid and contain exactly the fields defined by the schema.
"""
    )


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
    if TASK_ANALYSIS_PROMPT and TASK_ANALYSIS_PARSER:
        prompt_text = TASK_ANALYSIS_PROMPT.format(
            task=text,
            format_instructions=TASK_ANALYSIS_PARSER.get_format_instructions()
        )
    else:
        prompt_text = f"""You are a web task analyzer.
Analyze this task and return only a raw JSON object that matches the schema below.

Task: "{text}"

{{
  "intent": "SCAN_SEARCH|FORM_FILL|DATA_EXTRACT|VAULT_EXPORT|DEEP_SWEEP",
  "entity": "main target name",
  "subtasks": ["step 1", "step 2", "step 3"]
}}

The JSON must be valid and contain exactly the fields defined by the schema.
"""
    try:
        _genai_client = genai_client.Client(api_key=os.getenv('GEMINI_API_KEY'))
        response = await asyncio.to_thread(
            _genai_client.models.generate_content,
            model='models/gemini-2.5-flash',
            contents=prompt_text
        )
        raw = response.text if hasattr(response, 'text') else str(response)
        result = _parse_json_safe(raw)
        if result and 'intent' in result and 'entity' in result and 'subtasks' in result:
            try:
                return TaskAnalysis.parse_obj(result).dict()
            except Exception:
                return result
        if TASK_ANALYSIS_PARSER:
            try:
                parsed = TASK_ANALYSIS_PARSER.parse(raw)
                return parsed.dict()
            except Exception:
                pass
        return result if result else {}
    except Exception as e:
        print(f"Gemini API call failed: {type(e).__name__}: {str(e)}")
        pass

    lower = text.lower()
    if any(word in lower for word in ['search', 'find', 'cherche', 'trouve', 'recherche', 'look for']):
        intent = 'SCAN_SEARCH'
    elif any(word in lower for word in ['fill', 'form', 'submit', 'remplis', 'soumets']):
        intent = 'FORM_FILL'
    elif any(word in lower for word in ['extract', 'scrape', 'get data', 'extrais', 'récupère']):
        intent = 'DATA_EXTRACT'
    elif any(word in lower for word in ['vault', 'archive', 'export', 'save', 'sauvegarde']):
        intent = 'VAULT_EXPORT'
    else:
        intent = 'DEEP_SWEEP'

    words = [w for w in re.findall(r'\w+', text) if len(w) > 3]
    entity = words[0].upper() if words else 'TARGET'

    subtasks = {
        'SCAN_SEARCH': ['Open browser', 'Navigate to search engine', 'Enter query', 'Extract results'],
        'FORM_FILL': ['Navigate to page', 'Identify form fields', 'Fill inputs', 'Submit form'],
        'DATA_EXTRACT': ['Navigate to target', 'Analyze page content', 'Extract structured data'],
        'VAULT_EXPORT': ['Navigate to target', 'Collect data', 'Format export', 'Archive findings'],
        'DEEP_SWEEP': ['Navigate to target', 'Analyze page structure', 'Execute sweep', 'Archive findings']
    }

    return {
        'intent': intent,
        'entity': entity,
        'subtasks': subtasks.get(intent, ['Analyze task', 'Execute', 'Report results'])
    }
