import re

with open('c:/Users/Elite/web-agent-chatboot/backend/agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_code = """            if _last_state['count'] > 3:

                _last_state['count'] = 0
                stuck_q = ("""

new_code = """            if _last_state['count'] > 3:

                _last_state['count'] = 0
                if agent_context and contextual_target_count:
                    _send_event_sync(loop, send_event, {'type': 'log', 'message': f'Stuck on {page.url}, forcing extract...'})
                    extracted_data = {'page_text': page_text[:1000], 'url': page.url}
                    source_name = contextual_sources[contextual_source_index - 1]['name'] if 0 < contextual_source_index <= len(contextual_sources) else 'Recherche web'
                    contextual_research_results.append({'source': source_name, 'url': page.url, 'data': extracted_data})
                    _send_event_sync(loop, send_event, {'type': 'step', 'name': 'EXTRACT', 'args': f'{source_name}: forced', 'status': 'done'})
                    save_session(task, 'DATA_EXTRACT', str(extracted_data), status='done', namespace=memory_namespace)
                    if len(contextual_research_results) >= contextual_target_count:
                        report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                        _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                        save_session(task, 'DONE', report, status='done', namespace=memory_namespace)
                        return
                    if contextual_source_index < len(contextual_sources):
                        next_source = contextual_sources[contextual_source_index]
                        contextual_source_index += 1
                        next_url = _source_search_url(task, next_source['name'], next_source['domain'])
                        page.goto(next_url, timeout=20000)
                        page.wait_for_load_state('domcontentloaded', timeout=10000)
                        _send_event_sync(loop, send_event, {'type': 'url', 'value': page.url})
                        _send_event_sync(loop, send_event, {'type': 'step', 'name': 'SEARCH', 'args': next_source['name'], 'status': 'done'})
                        continue
                    report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                    _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                    save_session(task, 'DONE', report, status='done', namespace=memory_namespace)
                    return
                stuck_q = ("""

content = content.replace(old_code, new_code)

with open('c:/Users/Elite/web-agent-chatboot/backend/agent.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied successfully")
