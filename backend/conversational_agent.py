import re
from typing import Dict, List


def is_conversational_question(task: str) -> bool:
    """Detect if the user is asking a conversational question rather than requesting a web action."""
    conversational_patterns = [
        r'\b(c\'?est quoi|what is|qu\'?est-ce que|quel est|quelle est)\b.*\b(mon|ma|mes|my|the|last|dernier|derni[eè]re)\b',
        r'\b(rappelle|remind|tell me|dis moi|montre moi|show me)\b.*\b(mon|ma|mes|my|what i|ce que j\'?ai)\b',
        r'\b(qui suis je|who am i|mon nom|my name)\b',
        r'\b(quel était|what was|quelle était)\b.*\b(mon|ma|mes|my)\b',
    ]
    return any(re.search(pattern, task.lower()) for pattern in conversational_patterns)


def answer_conversational_question(task: str, conversation_history: List[Dict]) -> str:
    """Generate an answer to a conversational question based on conversation history."""
    if not conversation_history:
        return "Je n'ai pas d'historique de conversation pour répondre à cette question."
    
    # Extract last user messages
    user_messages = [msg for msg in conversation_history if msg.get('type') == 'user_feedback']
    
    if 'dernier' in task.lower() or 'last' in task.lower():
        if user_messages:
            last_msg = user_messages[-1].get('message', 'Aucun message trouvé')
            return f"Votre dernier message était : \"{last_msg}\""
        return "Je n'ai pas trouvé de message précédent."
    
    # Default: show recent history
    recent = conversation_history[-5:]
    history_text = "\n".join([f"- {item.get('type', 'unknown')}: {item.get('message', item.get('question', 'N/A'))}" for item in recent])
    return f"Voici l'historique récent de notre conversation :\n{history_text}"
