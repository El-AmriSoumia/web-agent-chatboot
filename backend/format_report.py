"""
Script pour remplacer la fonction _format_contextual_research_report
avec un format adaptatif selon le type de recherche et le contexte
"""

new_function = '''def _format_contextual_research_report(
    task: str,
    agent_name: Optional[str],
    results: List[Dict[str, Any]],
    agent_description: Optional[str] = None,
    agent_context: Optional[str] = None,
) -> str:
    subject = _extract_search_query(task)
    instructions = "\\n".join([agent_description or "", agent_context or ""]).lower()
    
    # Detect search type
    is_people = any(w in instructions for w in ['utilisateur', 'user', 'profil', 'profile', 'candidat', 'developer', 'developpeur'])
    is_product = any(w in instructions for w in ['produit', 'product', 'prix', 'price', 'concurrent', 'competitor', 'comparatif'])
    
    # Extract data
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
                    'title': result.get('title') or result.get('name') or result.get('username') or 'Sans titre',
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
                    'title': title or 'Produit',
                    'url': data.get('url') or url,
                    'price': data.get('price') or data.get('prix') or '',
                    'description': data.get('description') or data.get('availability') or '',
                    'email': data.get('email') or '',
                    'username': data.get('username') or '',
                    'location': data.get('location') or '',
                    'website': data.get('website') or '',
                })
    
    if not records:
        return f"# {agent_name or 'Rapport'}\\n\\nSujet: **{subject}**\\n\\nAucune donnee exploitable trouvee sur les sources consultees."
    
    # Build report
    lines = [
        f"# {agent_name or 'Rapport'}",
        "",
        f"**{subject}**",
        "",
        f"{len(records)} resultat(s) trouve(s) sur {len(set(s for s, _ in sources))} source(s).",
        "",
    ]
    
    # Sources
    lines.append("## Sources")
    for source, url in sources[:6]:
        lines.append(f"- {source}: {url}")
    lines.append("")
    
    # Results
    if is_people:
        lines.append("## Profils")
        for rec in records[:8]:
            lines.append(f"\\n**{rec['title']}** ({rec['source']})")
            if rec.get('username'):
                lines.append(f"- Username: {rec['username']}")
            if rec.get('email'):
                lines.append(f"- Email: {rec['email']}")
            if rec.get('location'):
                lines.append(f"- Localisation: {rec['location']}")
            if rec.get('description'):
                lines.append(f"- Info: {rec['description'][:200]}")
            if rec.get('website'):
                lines.append(f"- Site: {rec['website']}")
            if rec['url']:
                lines.append(f"- Profil: {rec['url']}")
    elif is_product:
        lines.append("## Produits")
        for rec in records[:8]:
            lines.append(f"\\n**{rec['title']}** ({rec['source']})")
            if rec['price']:
                lines.append(f"- Prix: {rec['price']}")
            if rec.get('description'):
                lines.append(f"- Info: {rec['description'][:200]}")
            if rec['url']:
                lines.append(f"- Lien: {rec['url']}")
    else:
        lines.append("## Resultats")
        for rec in records[:8]:
            lines.append(f"\\n**{rec['title']}** ({rec['source']})")
            if rec.get('description'):
                lines.append(f"- {rec['description'][:200]}")
            if rec['url']:
                lines.append(f"- Lien: {rec['url']}")
    
    lines.append("")
    
    # Summary
    lines.append("## Synthese")
    if is_product:
        priced = [r for r in records if r.get('price')]
        if priced:
            lines.append(f"- {len(priced)} produit(s) avec prix affiche")
            lines.append("- Compare les prix et verifie la disponibilite avant achat")
        else:
            lines.append("- Prix non affiches, consulte les liens pour plus d'infos")
    elif is_people:
        with_email = [r for r in records if r.get('email')]
        lines.append(f"- {len(with_email)} profil(s) avec email public")
        lines.append("- Verifie les profils pour plus de details")
    else:
        lines.append("- Consulte les liens pour plus d'informations")
    
    return "\\n".join(lines).strip()
'''

print("Nouvelle fonction créée. Remplace la fonction _format_contextual_research_report dans agent.py")
print("Ligne approximative: recherche 'def _format_contextual_research_report'")
