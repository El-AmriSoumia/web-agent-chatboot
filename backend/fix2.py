with open('agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace
old = """            if _last_state['count'] > 3:
                _last_state['count'] = 0
                if agent_context and contextual_target_count:"""

new = """            # Force extract after 2 iterations on e-commerce
            if _last_state['count'] >= 2 and agent_context and contextual_target_count:
                is_ecom = any(d in page.url.lower() for d in ['amazon.com', 'aliexpress.com', 'ebay.com'])
                if is_ecom:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': 'Forcing extract after stuck...'})
                    try:
                        pd = page.evaluate("() => {const t=document.querySelector('#productTitle,h1')?.textContent?.trim()||'Product';let p='N/A';for(const s of['.a-price .a-offscreen','[class*=price]']){const e=document.querySelector(s);if(e&&e.textContent.match(/[$]|\\d+/)){p=e.textContent.trim();break;}}return{product_title:t,price:p,url:window.location.href,availability:'Unknown'};}")
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
                    if _go_to_next_contextual_source(page, 'Finished'):
                        continue
                    report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                    _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                    save_session(task, 'DONE', report, status='done', namespace=memory_namespace)
                    return
            
            if _last_state['count'] > 3:
                _last_state['count'] = 0
                if agent_context and contextual_target_count:"""

if old in content:
    content = content.replace(old, new)
    with open('agent.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Added forced extraction after 2 iterations")
else:
    print("ERROR: Pattern not found")
