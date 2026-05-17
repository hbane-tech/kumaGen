"""
utils/normalize.py
Pure text normalization utilities.
No filtering, no lexicons — every word is meaningful and handled downstream.
"""

import re
import unicodedata


def clean_gloss(text: str) -> str:
    if not text:
        return text
    return text.strip().rstrip('.;,:').strip()


def normalize_token(token: str) -> str:
    # PROTECTION DATA-DRIVEN : Évite l'interruption du flux si un jeton arrive vide (None)
    if token is None:
        return ""
        
    # Cast en chaîne de caractères par sécurité (pour parer aux types numériques inattendus)
    token_str = str(token)
    
    token_str = unicodedata.normalize('NFC', token_str.lower().strip())
    token_str = re.sub(r"['\u2019\-]", ' ', token_str)
    token_str = token_str.strip('.,;:!?()')
    return token_str.strip()


def normalize_source(text: str) -> str:
    if not text:
        return ""
        
    text  = unicodedata.normalize('NFC', text.lower().strip())
    text  = re.sub(r"['\u2019\-]", ' ', text)
    words = [w.strip('.,;:!?()') for w in text.split()]
    return ' '.join(w for w in words if w)
