"""
Script Generator - Génère du code Playwright réutilisable à partir des actions enregistrées
"""
import json
import re
from datetime import datetime
from typing import Any, Dict, List


class PlaywrightScriptGenerator:
    """Génère du code Playwright réutilisable"""
    
    def __init__(self):
        self.actions = []
        self.parameters = {}
    
    def add_action(self, action_type: str, **kwargs):
        """Enregistre une action"""
        self.actions.append({
            'type': action_type,
            'data': kwargs
        })
    
    def detect_parameters(self):
        """Détecte les valeurs qui doivent être des paramètres"""
        params = {}
        
        for action in self.actions:
            if action['type'] == 'type':
                value = action['data'].get('text', '')
                selector = action['data'].get('selector', '')
                
                # Détecte les emails
                if '@' in value and re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value):
                    param_name = 'email'
                    params[param_name] = 'string'
                    action['data']['text'] = f'{{param_{param_name}}}'
                
                # Détecte les mots de passe
                elif 'password' in selector.lower() or 'pass' in selector.lower():
                    param_name = 'password'
                    params[param_name] = 'string'
                    action['data']['text'] = f'{{param_{param_name}}}'
                
                # Détecte les numéros de téléphone
                elif re.match(r'^\+?\d{10,}$', value.replace(' ', '')):
                    param_name = 'phone'
                    params[param_name] = 'string'
                    action['data']['text'] = f'{{param_{param_name}}}'
        
        self.parameters = params
    
    def generate_python_code(self, function_name: str, description: str) -> str:
        """Génère le code Python complet"""
        
        self.detect_parameters()
        
        # Paramètres de la fonction
        param_list = ', '.join(self.parameters.keys()) if self.parameters else ''
        
        code = f'''"""
Auto-generated Playwright workflow
Generated: {datetime.now().isoformat()}
Description: {description}
"""

from playwright.sync_api import sync_playwright
import json
import time

def {function_name}({param_list}):
    """
    {description}
    
'''
        
        if self.parameters:
            code += '    Parameters:\n'
            for param, ptype in self.parameters.items():
                code += f'        {param} ({ptype}): Input parameter\n'
        
        code += '''    
    Returns:
        dict: Execution result with success status and data
    """
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        result_data = {{}}
        
        try:
'''
        
        # Génère le code pour chaque action
        for i, action in enumerate(self.actions, 1):
            code += f'            # Step {i}: {action["type"]}\n'
            
            if action['type'] == 'navigate':
                url = action['data'].get('url', '')
                code += f'            page.goto("{url}", timeout=20000)\n'
                code += f'            page.wait_for_load_state("domcontentloaded", timeout=10000)\n'
            
            elif action['type'] == 'click':
                selector = action['data'].get('selector', '')
                code += f'            try:\n'
                code += f'                page.click("{selector}", timeout=5000)\n'
                code += f'            except Exception:\n'
                code += f'                page.get_by_text("{selector}", exact=False).first.click(timeout=5000)\n'
            
            elif action['type'] == 'type':
                selector = action['data'].get('selector', '')
                text = action['data'].get('text', '')
                
                # Remplace les placeholders par les paramètres
                for param in self.parameters.keys():
                    text = text.replace(f'{{param_{param}}}', f'{{{param}}}')
                
                code += f'            page.fill("{selector}", f"{text}")\n'
                code += f'            time.sleep(0.5)\n'
            
            elif action['type'] == 'fill_form':
                fields = action['data'].get('fields', [])
                code += f'            # Fill form fields\n'
                for field in fields:
                    sel = field.get('selector', '')
                    val = field.get('value', '')
                    code += f'            page.fill("{sel}", "{val}")\n'
                    code += f'            time.sleep(0.3)\n'
                
                submit = action['data'].get('submit_selector')
                if submit:
                    code += f'            page.click("{submit}", timeout=5000)\n'
                    code += f'            page.wait_for_load_state("networkidle", timeout=10000)\n'
            
            elif action['type'] == 'scroll':
                direction = action['data'].get('direction', 'down')
                amount = 600 if direction == 'down' else -600
                code += f'            page.evaluate("window.scrollBy(0, {amount})")\n'
                code += f'            time.sleep(0.5)\n'
            
            elif action['type'] == 'extract':
                data = action['data'].get('data', {})
                code += f'            result_data = {json.dumps(data, indent=16)}\n'
            
            elif action['type'] == 'done':
                summary = action['data'].get('summary', '')
                code += f'            # Task completed: {summary}\n'
            
            code += '\n'
        
        code += '''            
            # Get final page state
            title = page.title()
            url = page.url()
            
            browser.close()
            
            return {{
                "success": True,
                "title": title,
                "url": url,
                "data": result_data
            }}
        
        except Exception as e:
            browser.close()
            return {{
                "success": False,
                "error": str(e)
            }}


if __name__ == "__main__":
    # Example execution
'''
        
        # Exemple d'exécution
        if self.parameters:
            example_params = ', '.join([f'"{param}_example"' for param in self.parameters.keys()])
            code += f'    result = {function_name}({example_params})\n'
        else:
            code += f'    result = {function_name}()\n'
        
        code += f'    print(json.dumps(result, indent=2))\n'
        
        return code
