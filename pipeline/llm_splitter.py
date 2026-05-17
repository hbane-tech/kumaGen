"""
pipeline/llm_splitter.py
Splitter hybride adaptatif pour Framework "Chunk-as-Clause".
Disjoncteur temporel (3s) avec Fallback algorithmique ultra-rapide anti-freeze [1].
"""
import json
import re
import requests

def llm_split_wagons(phrase_fr: str) -> list:
    """
    Découpe la phrase en wagons sémantiques.
    Tente une approche neurale (LLM) sous 3 secondes, sinon applique un fallback algorithmique [1].
    """
    url = "http://localhost:11434/api/generate"
    
    prompt = f"""
    Découpe la phrase française suivante en groupes circonstanciels (wagons).
    PHRASE : "{phrase_fr}"
    Réponds uniquement sous la forme d'un tableau JSON d'objets avec les clés "connecteur", "texte" et "type".
    """
    
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        # Sécurité absolue : timeout abaissé à 3.0s pour éviter tout gel à l'écran
        response = requests.post(url, json=payload, timeout=3.0)
        raw_response = response.json().get('response', '').strip()
        parsed_data = json.loads(raw_response)
        if isinstance(parsed_data, list) and len(parsed_data) > 0:
            return parsed_data
    except Exception:
        # Si Ollama est trop lent ou gelé, le disjoncteur s'active sans crash [1]
        pass

       # ─────────────────────────────────────────────────────────────────────────
    # 🛡️ FALLBACK ALGORITHMIQUE DÉTERMINISTE AGNOSTIQUE (ZÉRO HARDCODING)
    # ─────────────────────────────────────────────────────────────────────────
    # Extraction dynamique de tous les connecteurs présents dans la liste de jetons (T)
    # en se basant uniquement sur leur catégorie grammaticale ou leur dépendance universelle
    token_connectors = sorted([t for t in T if t.get('pos') in ('ADP', 'SCONJ', 'CCONJ') or t.get('dep') in ('case', 'cc')], key=lambda x: x['orig_index'])
    
    clean_phrase = phrase_fr.replace('«', '').replace('»', '').strip()
    validated_wagons = []
    
    if not token_connectors:
        return [{"connecteur": "", "texte": clean_phrase, "type": "simple"}]
        
    # Reconstruction linéaire des wagons guidée par la topologie des jetons du KG
    for idx, conn_tok in enumerate(token_connectors):
        start_idx = conn_tok['orig_index']
        
        # Détermination des frontières textuelles par intervalles de jetons
        connecteur_trouve = conn_tok.get('surface', '')
        
        # Reconstruction sémantique abstraite du type d'après les traits du Graphe de Connaissances
        type_sémantique = "simple"
        if conn_tok.get('role') == 'location' or conn_tok.get('is_loc', False):
            type_sémantique = "locative"
        elif conn_tok.get('role') == 'temporal':
            type_sémantique = "temporal"
            
        # Extraction du texte associé à ce wagon de connecteur
        # Le système récupère le texte du segment jusqu'au connecteur suivant
        words_after = [t.get('surface', '') for t in T if t['orig_index'] > start_idx]
        if idx + 1 < len(token_connectors):
            next_start = token_connectors[idx+1]['orig_index']
            words_after = [t.get('surface', '') for t in T if start_idx < t['orig_index'] < next_start]
            
        segment_texte = ' '.join(words_after).strip()
        
        validated_wagons.append({
            "connecteur": connecteur_trouve,
            "texte": segment_texte,
            "type": type_sémantique
        })
        
    return validated_wagons
