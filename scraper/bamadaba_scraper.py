import requests
from bs4 import BeautifulSoup, Tag
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
# Extract ALL senses from a <p class="lxP"> tag
#
# Un mot polysémique (ex: "kálo") porte PLUSIEURS <p class="lxP2"> frères
# consécutifs, un par sens numéroté (1 • lune. / 2 • mois. / 3 • règles.).
# Seul le PREMIER porte le tag <span class="PS"> (POS partagé par tous les
# sens) — les suivants n'ont qu'un <span class="SnsN"> + <span class="GlFr">.
# Bug trouvé 2026-07-17 : find_next_sibling() ne retournait QUE ce premier
# bloc, perdant silencieusement tout sens 2+ de chaque entrée polysémique
# du dictionnaire entier (ex: "mois"/"règles" absents pour "kálo").
# ---------------------------------------------------------------------------

def extract_entries(p_lemma):
    # 1. LEMMA
    lxe = p_lemma.find("span", class_="Lxe")
    if not lxe:
        return []

    lemma = normalize_text(lxe.get_text()).rstrip(".")
    if not is_valid_lemma(lemma):
        return []

    # lemma_plain: tone-stripped, used for accent-insensitive exact matching
    # so that querying "n" finds "ń", "i" finds "í", etc.
    lemma_plain = strip_diacritics(lemma)

    # corpus_freq : nombre d'attestations dans le corpus de référence (bouton
    # "→ N" class="clnknt"), signal RÉEL de fréquence d'usage — pas une
    # heuristique. Partagé par tous les sens de cette entrée (un seul bouton
    # par headword). Découvert 2026-07-17 : deux homographes distincts
    # peuvent chacun avoir "jour" comme SENS N°1 de leur propre entrée
    # ('dá'="jour." et 'dón'="jour, date.", tous deux sense_index=1) —
    # sense_index seul ne peut pas les départager, mais leurs fréquences
    # corpus (dá=27 vs dón=3110) tranchent sans ambiguïté.
    corpus_freq = 0
    freq_tag = p_lemma.find("a", class_="clnknt") or p_lemma.find(
        lambda t: t.name == "a" and "clnknt" in (t.get("class") or []))
    if not freq_tag:
        _b = p_lemma.find("b", class_="clnknt")
        freq_tag = _b.find("a") if _b else None
    if freq_tag:
        _digits = "".join(c for c in freq_tag.get_text() if c.isdigit())
        if _digits:
            corpus_freq = int(_digits)

    # 2. Tous les blocs lxP2 frères consécutifs (jusqu'au prochain lxP)
    info_blocks = []
    sib = p_lemma.next_sibling
    while sib is not None:
        if isinstance(sib, Tag):
            classes = sib.get("class") or []
            if sib.name == "p" and "lxP2" in classes:
                info_blocks.append(sib)
            elif sib.name == "p" and "lxP" in classes:
                break
        sib = sib.next_sibling
    if not info_blocks:
        return []

    # 3. POS : seul le premier bloc le porte, partagé par tous les sens
    pos_tag = info_blocks[0].find("span", class_="PS")
    pos_raw = pos_tag.get_text(strip=True) if pos_tag else None
    parsed  = parse_pos(pos_raw)

    entries = []
    for block in info_blocks:
        # Un GlFr/GlFr1 n'est une VRAIE définition que s'il précède le premier
        # <Exe> du bloc — un GlFr qui ne vient qu'APRÈS un Exe est la
        # traduction de l'exemple, pas une définition indépendante (ex: sens
        # numéroté qui n'illustre que le sens parent par un exemple, sans
        # apporter de glose propre). Bug trouvé 2026-07-17 : le premier GlFr
        # trouvé dans le DOM était pris aveuglément, confondant traduction
        # d'exemple et définition pour ces sous-sens ("tìn" sens 'compassion'
        # récupérait la traduction de l'exemple comme si c'était son sens).
        _children = block.find_all("span", recursive=False)
        _first_exe_pos = next(
            (i for i, c in enumerate(_children) if "Exe" in (c.get("class") or [])),
            len(_children))

        def get_def_text(cls, _block=block, _limit=_first_exe_pos, _kids=_children):
            for i, tag in enumerate(_kids):
                if i >= _limit:
                    break
                if cls in (tag.get("class") or []):
                    return normalize_text(tag.get_text())
            return None

        fr_short = get_def_text("GlFr1") or get_def_text("GlFr")
        fr_long  = get_def_text("EncFr")

        # sense_index : numéro du sens DANS SA PROPRE entrée du dictionnaire
        # source ("1 •", "2 •"...) — départage entre sens d'UN MÊME headword.
        # Absent (mono-sens) → 1. Ne départage PAS deux headwords différents
        # (voir corpus_freq ci-dessus pour ce cas, ex: 'dá'/'dón' tous deux
        # sens n°1 de leur propre entrée mais fréquences très différentes).
        sns_tag = block.find("span", class_="SnsN")
        sense_index = 1
        if sns_tag:
            _digits = "".join(c for c in sns_tag.get_text() if c.isdigit())
            if _digits:
                sense_index = int(_digits)

        # fr     : short display gloss stored on the Word node
        # fr_emb : richest French text for LaBSE encoding.
        #          For pronouns (GlFr1 = "1SG, me, ma") the long encyclopedic
        #          gloss ("pronom de la première personne du singulier") aligns
        #          far better with queries like "I" / "je" / "moi".
        fr     = fr_short or fr_long
        fr_emb = fr_long  or fr_short
        if not fr:
            continue

        entries.append({
            "lemma":        lemma,
            "lemma_plain":  lemma_plain,
            "type":         parsed["pos"],
            "ps_raw":       pos_raw,
            "sense_index":  sense_index,
            "corpus_freq":  corpus_freq,
            "fr":           fr,
            "fr_emb":       fr_emb,
            "transitivity": parsed["transitivity"],
            "subtype":      parsed["subtype"],
        })
    return entries


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
            all_entries.extend(extract_entries(p))

        print(f"  collected so far: {len(all_entries)}")

    return all_entries