import re

with open('agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the stuck-loop detection section and add forced extraction after 2 iterations
old_pattern = r"(            # --- Stuck-loop detection ---\r?\n            _cur_text_hash = hash\(page_text\[:500\]\)\r?\n            if page\.url == _last_state\['url'\] and _cur_text_hash == _last_state\['text_hash'\]:\r?\n                _last_state\['count'\] \+= 1\r?\n            else:\r?\n                _last_state\.update\(\{'url': page\.url, 'text_hash': _cur_text_hash, 'count': 1\}\)\r?\n            if _last_state\['count'\] > 3:)"

new_text = """            # --- Stuck-loop detection ---
            _cur_text_hash = hash(page_text[:500])
            if page.url == _last_state['url'] and _cur_text_hash == _last_state['text_hash']:
                _last_state['count'] += 1
            else:
                _last_state.update({'url': page.url, 'text_hash': _cur_text_hash, 'count': 1})
            
            # Force extract after 2 iterations on e-commerce for contextual agents
            if _last_state['count'] >= 2 and agent_context and contextual_target_count:
                is_ecom = any(d in page.url.lower() for d in ['amazon.com', 'aliexpress.com', 'ebay.com'])
                if is_ecom:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': f'Forcing extract after {_last_state["count"]} iterations...'})
                    try:
                        pd = page.evaluate(\"\"\"() => {
                            const t = document.querySelector('#productTitle, h1')?.textContent?.trim() || 'Product';
                            let p = 'Price not displayed';
                            for (const s of ['.a-price .a-offscreen', '[class*="price"]']) {
                                const e = document.querySelector(s);
                                if (e && e.textContent.match(/[$]|\\d+/)) { p = e.textContent.trim(); break; }
                            }
                            return {product_title: t, price: p, url: window.location.href, availability: 'Unknown'};
                        }\"\"\")
                    except:
                        pd = {'product_title': 'Failed', 'price': 'N/A', 'url': page.url, 'availability': 'Unknown'}
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
                    if _go_to_next_contextual_source(page, 'Finished current source'):
                        continue
                    report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                    _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                    save_session(task, 'DONE', report, status='done', namespace=memory_namespace)
                    return
            
            if _last_state['count'] > 3:"""

content = re.sub(old_pattern, new_text, content, flags=re.MULTILINE)

with open('agent.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed stuck-loop detection - now forces extract after 2 iterations on e-commerce pages")
