import re

with open('c:/Users/Elite/web-agent-chatboot/backend/agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the stuck detection logic
old_pattern = r"(if _last_state\['count'\] > 3:.*?_last_state\['count'\] = 0\s+)(stuck_q = \(.*?continue)"

new_code = r"""\1if agent_context and contextual_target_count:
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
                        _last_state.update({'url': None, 'text_hash': None, 'count': 0})
                        continue
                    report = _format_contextual_research_report(task, agent_name, contextual_research_results, agent_description, agent_context)
                    _send_event_sync(loop, send_event, {'type': 'result', 'data': {'report': report, 'sources': contextual_research_results}})
                    save_session(task, 'DONE', report, status='done', namespace=memory_namespace)
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
                continue"""

content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

with open('c:/Users/Elite/web-agent-chatboot/backend/agent.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied successfully")
