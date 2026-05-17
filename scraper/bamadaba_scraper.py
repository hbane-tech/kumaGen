import requests
from bs4 import BeautifulSoup
import unicodedata

BASE_URL = "http://cormand.huma-num.fr/Bamadaba/lexicon/{}.htm"

LETTERS = [
    "a","b","c","d","e","ɛ","f","g","h","i","j","k","l","m",
    "n","ɲ","ŋ","o","ɔ","p","r","s","t","u","w","x","y","z"
]


# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------

def normalize_text(text):
    """NFC-normalise and strip whitespace."""
    if not text:
        return text
    return unicodedata.normalize("NFC", text.strip())


def strip_diacritics(text):
    """
    Remove combining diacritical marks (tones, accents).
    "ń" → "n",  "bɛ́" → "bɛ",  "dɔ̀n" → "dɔn"
    Used to build lemma_plain for accent-insensitive exact matching.
    """
    if not text:
        return text
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


# ---------------------------------------------------------------------------
# POS parser
# ---------------------------------------------------------------------------

def parse_pos(ps_raw):
    if not ps_raw:
        return {"pos": "Other", "transitivity": None, "subtype": None}

    ps = ps_raw.lower().replace(" ", "").strip().rstrip(".")

    if ps.startswith("vt"):    return {"pos": "Verb",          "transitivity": "transitive",  "subtype": None}
    if ps.startswith("vi"):    return {"pos": "Verb",          "transitivity": "intransitive","subtype": None}
    if ps.startswith("vr"):    return {"pos": "Verb",          "transitivity": "reflexive",   "subtype": None}
    if ps.startswith("vq"):    return {"pos": "Verb",          "transitivity": None,          "subtype": "qualitative"}
    if ps.startswith("v"):     return {"pos": "Verb",          "transitivity": None,          "subtype": None}
    if ps.startswith("pers"):  return {"pos": "Pronoun",       "transitivity": None,          "subtype": "personal"}
    if ps.startswith("pron"):  return {"pos": "Pronoun",       "transitivity": None,          "subtype": None}
    if ps.startswith("adj"):   return {"pos": "Adjective",     "transitivity": None,          "subtype": None}
    if ps.startswith("adv"):   return {"pos": "Adverb",        "transitivity": None,          "subtype": None}
    if ps.startswith("intj"):  return {"pos": "Interjection",  "transitivity": None,          "subtype": None}
    if ps.startswith("prep"):  return {"pos": "Preposition",   "transitivity": None,          "subtype": None}
    if ps.startswith("conj"):  return {"pos": "Conjunction",   "transitivity": None,          "subtype": None}
    if ps.startswith("part"):  return {"pos": "Particle",      "transitivity": None,          "subtype": None}
    if ps.startswith("cop"):   return {"pos": "Copula",        "transitivity": None,          "subtype": None}
    if ps.startswith("aux"):   return {"pos": "Auxiliary",     "transitivity": None,          "subtype": None}
    if ps.startswith("neg"):   return {"pos": "Negation",      "transitivity": None,          "subtype": None}
    if ps.startswith("inter"): return {"pos": "Interrogative", "transitivity": None,          "subtype": None}
    if ps.startswith("det"):   return {"pos": "Determiner",    "transitivity": None,          "subtype": None}
    if ps.startswith("art"):   return {"pos": "Article",       "transitivity": None,          "subtype": None}
    if ps.startswith("n"):     return {"pos": "Noun",          "transitivity": None,          "subtype": None}

    return {"pos": "Other", "transitivity": None, "subtype": None}


# ---------------------------------------------------------------------------
# Lemma validation
# ---------------------------------------------------------------------------

def is_valid_lemma(lemma):
    if not lemma:
        return False
    if lemma.isdigit():
        return False
    # Allow single-char lemmas only if they are a real unicode letter
    # (Bambara pronouns: ń, i, a, u after NFC normalisation)
    if len(lemma) == 1 and not lemma.isalpha():
        return False
    return True


# ---------------------------------------------------------------------------
# Extract ONE entry from a <p class="lxP"> tag
# ---------------------------------------------------------------------------

def extract_entry(p_lemma):
    # 1. LEMMA
    lxe = p_lemma.find("span", class_="Lxe")
    if not lxe:
        return None

    lemma = normalize_text(lxe.get_text()).rstrip(".")
    if not is_valid_lemma(lemma):
        return None

    # lemma_plain: tone-stripped, used for accent-insensitive exact matching
    # so that querying "n" finds "ń", "i" finds "í", etc.
    lemma_plain = strip_diacritics(lemma)

    # 2. INFO BLOCK
    p_info = p_lemma.find_next_sibling("p", class_="lxP2")
    if not p_info:
        return None

    # 3. POS
    pos_tag     = p_info.find("span", class_="PS")
    pos_raw     = pos_tag.get_text(strip=True) if pos_tag else None
    parsed      = parse_pos(pos_raw)

    # 4. French fields
    def get_text(cls):
        tag = p_info.find("span", class_=cls)
        return normalize_text(tag.get_text()) if tag else None

    fr_short = get_text("GlFr1") or get_text("GlFr")
    fr_long  = get_text("EncFr")

    # fr     : short display gloss stored on the Word node
    # fr_emb : richest French text for LaBSE encoding.
    #          For pronouns (GlFr1 = "1SG, me, ma") the long encyclopedic
    #          gloss ("pronom de la première personne du singulier") aligns
    #          far better with queries like "I" / "je" / "moi".
    fr     = fr_short or fr_long
    fr_emb = fr_long  or fr_short

    return {
        "lemma":        lemma,
        "lemma_plain":  lemma_plain,
        "type":         parsed["pos"],
        "ps_raw":       pos_raw,
        "fr":           fr,
        "fr_emb":       fr_emb,
        "transitivity": parsed["transitivity"],
        "subtype":      parsed["subtype"],
    }


# ---------------------------------------------------------------------------
# MAIN SCRAPER
# ---------------------------------------------------------------------------

def scrape_all():
    all_entries = []

    for letter in LETTERS:
        print(f"Scraping {letter}...")
        url = BASE_URL.format(letter)
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        # FORCE correct decoding from raw bytes
        res.encoding = res.apparent_encoding or "utf-8"
        html = res.text

        soup = BeautifulSoup(html, "html.parser")
        # soup = BeautifulSoup(res.text, "html.parser")

        for p in soup.find_all("p", class_="lxP"):
            entry = extract_entry(p)
            if entry:
                all_entries.append(entry)

        print(f"  collected so far: {len(all_entries)}")

    return all_entries