"""
pipeline/proposition_parser.py

Proposition-based Bambara sentence generator.

ARCHITECTURE
============
Instead of trying to fit a whole sentence into one slot dict,
we split by verb anchors and process each proposition independently.

Each verb defines one proposition with:
  - its own subject (found via head_index)
  - its own object/complement
  - its own TAM (from verb morphology)
  - its own pattern (standard / passive / reflexive / participial)

Propositions are then assembled with connectors.

FLOW
====
tokens
  → find_propositions()     # group tokens by verb anchor via head_index
  → Proposition.to_bambara() # apply pattern per proposition
  → assemble()              # join with connectors (ní, ko, sabu, hali ni...)

PATTERNS
========
Standard:     S TAM O XCOMP V [IOBJ ye] [LOC na] [ADV]
Passive:      S bɛ ka V [ADV] AGENT fɛ [LOC la]
Refl past:    S tun ye V [O] [LOC na]
Participial:  min/minniw V+ra/na COMPLEMENT fɛ
Qualitative:  S ka ADJ [INTENS]
Equative:     S ye ATTR ye
Imperative+:  V [IOBJ ye] O
Imperative-:  kana S O V
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# TAM + MORPHOLOGY HELPERS
# ══════════════════════════════════════════════════════════════════

_TAM = {
    ('pres', False): 'bɛ',
    ('pres', True):  'tɛ',
    ('fut',  False): 'bɛ na',
    ('fut',  True):  'tɛ na',
    ('past', False): 'yé',
    ('past', True):  'ma',
    ('cond', False): 'bɛ na',
    ('cond', True):  'tɛ na',
    ('imp',  False): '',
    ('imp',  True):  '',
}

def get_tam(tense: str, is_neg: bool) -> str:
    return _TAM.get((tense, is_neg), 'bɛ')

def perfect_past(bm: str) -> str:
    """
    Bambara perfect past suffix.
    ends in 'n' -> +na  (kalan -> kalanna)
    other       -> +ra  (jɛ -> jɛra, kɛ -> kɛra)
    """
    if not bm:
        return bm
    return (bm + 'na') if bm.endswith('n') else (bm + 'ra')

def min_form(is_plural: bool) -> str:
    """min (singular) / minniw (plural)"""
    return 'minniw' if is_plural else 'min'

def plural_bm(bm: str, tok: dict) -> str:
    # Never pluralize pronouns — they have fixed forms
    if tok and tok.get('is_plural')        and tok.get('pos') not in ('PRON','DET')        and tok.get('role') not in ('pronoun','possessive','demonstrative')        and bm and not bm.endswith('w'):
        return bm + 'w'
    return bm

def poss_np(poss_bm: str, noun_bm: str) -> str:
    """n NOUN (1st person) / PRON ka NOUN (others)"""
    if poss_bm in {'n'}:
        return f"{poss_bm} {noun_bm}"
    return f"{poss_bm} ka {noun_bm}"


# ══════════════════════════════════════════════════════════════════
# TOKEN HELPERS
# ══════════════════════════════════════════════════════════════════

def _bm(tok) -> str:
    if tok is None: return ''
    v = tok.get('bm') or tok.get('lemma') or tok.get('surface') or ''
    return str(v) if v else ''

def _pos(tok) -> str:  return (tok.get('pos') or '') if tok else ''
def _dep(tok) -> str:  return (tok.get('dep') or '') if tok else ''
def _role(tok) -> str: return (tok.get('role') or 'content') if tok else ''
def _orig(tok) -> int: return tok.get('orig_index', -1) if tok else -1

def _is_content(tok) -> bool:
    return _role(tok) not in ('article', 'preposition', 'auxiliary',
                               'punct', 'function_candidate')

def _j(*parts) -> str:
    return ' '.join(p for p in parts if p and str(p).strip())


# ══════════════════════════════════════════════════════════════════
# NP BUILDER
# ══════════════════════════════════════════════════════════════════

def build_np(head_tok: dict, all_tokens: list,
             already_used: set = None) -> str:
    """
    Build a full NP for head_tok using all_tokens.
    Finds: possessive, nmod (genitive), amod ADJs, acl participials.
    Returns Bambara string.
    """
    used    = already_used or set()
    head_orig = _orig(head_tok)
    if head_orig < 0:
        return _bm(head_tok)

    # Find all direct dependents of this noun
    deps = [t for t in all_tokens
            if t.get('head_index', -1) == head_orig
            and id(t) not in used]

    # Possessive determiner
    poss = next((t for t in deps
                 if _role(t) in ('pronoun', 'possessive')
                 and _dep(t) in ('det', 'poss')), None)

    # Genitive nmod (de/du/des)
    nmod = next((t for t in deps
                 if _dep(t) == 'nmod' and _is_content(t)
                 and t.get('is_genitive')), None)

    # Plain nmod (no de — juxtaposition like 'làbitani medesɛn')
    plain_nmod = next((t for t in deps
                       if _dep(t) == 'nmod' and _is_content(t)
                       and not t.get('is_genitive')
                       and t is not nmod), None) if not nmod else None

    # Adjectives (amod)
    adjs = [t for t in deps
            if _dep(t) == 'amod' and _pos(t) == 'ADJ']

    # Participial adjective: dep=acl ONLY (not acl:relcl)
    # acl:relcl = full relative clause -> handled by proposition engine
    # acl = bare participial adjective: jóginna, jɛra...
    acl = next((t for t in deps
                if _dep(t) == 'acl'
                and _pos(t) == 'VERB'), None)
    for t in [poss, nmod, plain_nmod, acl] + adjs:
        if t: used.add(id(t))

    # Build base noun
    noun_bm = plural_bm(_bm(head_tok), head_tok)

    if poss and nmod:
        base = f"{_bm(poss)} {_bm(nmod)} ka {noun_bm}"
    elif poss:
        base = poss_np(_bm(poss), noun_bm)
    elif nmod:
        # Genitive (de/du/des): always juxtaposition POSSESSOR NOUN
        # 'ka' is only for pronoun possession (n ka, i ka, a ka)
        # 'médecins de l'hôpital' -> làbitani medesɛnw (no ka)
        # 'chien de mon père' -> n fà ka wùlukɛ (ka comes from poss+nmod)
        base = f"{_bm(nmod)} {noun_bm}"
    elif plain_nmod:
        base = f"{_bm(plain_nmod)} {noun_bm}"
    else:
        base = noun_bm

    # Adjectives after noun
    adj_str = ' '.join(_bm(a) for a in adjs)
    if adj_str:
        base = f"{base} {adj_str}"

    # Participial adjective: min/minniw V+ra/na complement fɛ
    if acl:
        acl_orig = _orig(acl)
        is_plur  = head_tok.get('is_plural', False)
        min_f    = min_form(is_plur)
        v_past   = perfect_past(_bm(acl))
        # Complement of participial verb
        acl_deps = [t for t in all_tokens
                    if t.get('head_index', -1) == acl_orig
                    and _is_content(t)
                    and _dep(t) in ('obj','obl','nmod','iobj','obl:agent')]
        compl_parts = []
        for cd in acl_deps:
            used.add(id(cd))
            compl_parts.append(build_np(cd, all_tokens, used))
        compl_str = ' '.join(compl_parts)
        if compl_str:
            # Has agent/complement -> V+ra complement fɛ
            base = f"{base} {min_f} {v_past} {compl_str} fɛ"
        else:
            # No complement -> just participial, no fɛ
            # e.g. frère blessé -> bálimakɛ min jóginna
            base = f"{base} {min_f} {v_past}"

    # Full relative clause: dep=acl:relcl
    # Pattern: NOUN min/minniw TAM OBJ V
    # e.g. 'médecins qui ont soigné mon frère'
    #   -> medesɛnw minniw yé n bálimakɛ fúrakɛ
    relcl = next((t for t in deps
                  if _dep(t) == 'acl:relcl'
                  and _pos(t) == 'VERB'), None)
    if relcl:
        used.add(id(relcl))
        relcl_orig = _orig(relcl)
        is_plur    = head_tok.get('is_plural', False)
        min_f      = min_form(is_plur)
        rel_tense  = relcl.get('tense', 'pres')
        rel_neg    = relcl.get('is_neg', False)
        rel_tam    = get_tam(rel_tense, rel_neg)
        # Object of relcl verb
        relcl_obj  = next((t for t in all_tokens
                           if _dep(t) == 'obj'
                           and t.get('head_index',-1) == relcl_orig),
                          None)
        rparts     = [min_f]
        if rel_tam: rparts.append(rel_tam)
        if relcl_obj:
            used.add(id(relcl_obj))
            rparts.append(build_np(relcl_obj, all_tokens, used))
        rparts.append(_bm(relcl))
        base = f"{base} {' '.join(p for p in rparts if p)}"

    return base


# ══════════════════════════════════════════════════════════════════
# PROPOSITION
# ══════════════════════════════════════════════════════════════════

@dataclass
class Proposition:
    """
    One verb + all tokens that directly depend on it.
    The basic unit of Bambara sentence construction.
    """
    verb:       dict
    all_tokens: list
    connector:  str = ''    # how this prop connects to previous

    def __post_init__(self):
        self._used    = set()
        self._verb_orig = _orig(self.verb)
        # Direct dependents: head_index == verb's orig_index
        self._deps = [t for t in self.all_tokens
                      if t.get('head_index', -1) == self._verb_orig
                      and _role(t) != 'punct']

    # ── Slot finders ──────────────────────────────────────────────

    def _find(self, dep=None, pos=None, role=None,
              exclude=None) -> Optional[dict]:
        for t in self._deps:
            if exclude and t in exclude: continue
            if dep:
                allowed = dep if isinstance(dep, list) else [dep]
                if _dep(t) not in allowed: continue
            if pos:
                allowed = pos if isinstance(pos, list) else [pos]
                if _pos(t) not in allowed: continue
            if role:
                allowed = role if isinstance(role, list) else [role]
                if _role(t) not in allowed: continue
            return t
        return None

    def _find_all(self, dep=None, pos=None, role=None) -> list:
        result = []
        for t in self._deps:
            if dep:
                allowed = dep if isinstance(dep, list) else [dep]
                if _dep(t) not in allowed: continue
            if pos:
                allowed = pos if isinstance(pos, list) else [pos]
                if _pos(t) not in allowed: continue
            if role:
                allowed = role if isinstance(role, list) else [role]
                if _role(t) not in allowed: continue
            result.append(t)
        return result

    # ── Slot properties ───────────────────────────────────────────

    @property
    def subject(self) -> Optional[dict]:
        # Prefer nsubj, then demonstrative, then any pronoun
        s = self._find(dep='nsubj')
        if not s:
            s = self._find(role='demonstrative')
        if not s:
            s = self._find(role='pronoun')

        # If still no subject (promoted xcomp — parent verb was dropped)
        # find the nearest nsubj pronoun in all_tokens that is
        # positionally close to this verb and not yet claimed
        if not s and _dep(self.verb) == 'xcomp':
            verb_orig = _orig(self.verb)
            s = next(
                (t for t in sorted(self.all_tokens,
                                   key=lambda t: abs(_orig(t) - verb_orig))
                 if _role(t) == 'pronoun'
                 and _dep(t) == 'nsubj'
                 and abs(_orig(t) - verb_orig) <= 4),
                None)
        return s

    @property
    def obj(self) -> Optional[dict]:
        o = self._find(dep='obj')
        if not o:
            # noun-xcomp (à manger -> NOUN)
            o = self._find(dep='xcomp', pos='NOUN')
        if not o:
            # Search through xcomp chain for obj
            # 'engager à assurer le pouvoir' -> pouvoir is obj of assurer(xcomp)
            xc = self._find(dep='xcomp')
            if xc:
                xc_orig = _orig(xc)
                o = next((t for t in self.all_tokens
                          if _dep(t) == 'obj'
                          and t.get('head_index', -1) == xc_orig), None)
        return o

    @property
    def iobj(self) -> Optional[dict]:
        return self._find(dep='iobj')

    @property
    def agent(self) -> Optional[dict]:
        """'par les combattants' -> agent noun"""
        # Try obl:agent dep
        a = self._find(dep=['obl:agent'])
        if not a:
            # Try is_agent flag (set by pre-pass on 'par' governed nouns)
            a = next((t for t in self._deps
                      if t.get('is_agent') and _is_content(t)), None)
        if not a:
            # Try any noun/propn in deps that has is_agent in all_tokens
            a = next((t for t in self.all_tokens
                      if t.get('is_agent') and _is_content(t)
                      and t.get('head_index', -1) == self._verb_orig), None)
        return a

    @property
    def dative(self) -> Optional[dict]:
        """'assurer à la junte' -> dative beneficiary"""
        d = next((t for t in self._deps
                  if t.get('is_dative') and _is_content(t)), None)
        if not d:
            # Search through xcomp chain
            xc = self._find(dep='xcomp')
            if xc:
                xc_orig = _orig(xc)
                d = next((t for t in self.all_tokens
                          if t.get('is_dative') and _is_content(t)
                          and t.get('head_index', -1) == xc_orig), None)
        return d

    @property
    def xcomp(self) -> Optional[dict]:
        """infinitive complement verb"""
        return self._find(dep='xcomp', pos='VERB')

    @property
    def ccomp_verb(self) -> Optional[dict]:
        """reported speech complement verb"""
        return self._find(dep='ccomp')

    @property
    def loc(self) -> Optional[dict]:
        """locative noun"""
        return next((t for t in self._deps
                     if (t.get('is_loc') or
                         _dep(t) in ('obl','obl:arg','obl:mod'))
                     and _pos(t) in ('NOUN','PROPN')
                     and _is_content(t)
                     and not t.get('is_agent')
                     and not t.get('is_dative')), None)

    @property
    def advs(self) -> list:
        return [t for t in self._deps
                if _pos(t) == 'ADV' and _is_content(t)
                and _dep(t) == 'advmod']

    @property
    def intens(self) -> Optional[dict]:
        return next((t for t in self._deps
                     if _pos(t) == 'ADV' and _is_content(t)
                     and (t.get('bm') in ('kojugu','dɔ́rɔn','blen')
                          or t.get('lemma') in ('très','trop','très',
                                                 'very','too'))), None)

    @property
    def root_adj(self) -> Optional[dict]:
        """For qualitative: être + ADJ -> ADJ"""
        if _pos(self.verb) == 'ADJ':
            return self.verb
        if self.verb.get('lemma') in ('être','be','faire','avoir'):
            return self._find(dep=['attr','xcomp','acomp'],
                              pos='ADJ') or \
                   self._find(pos='ADJ')
        return None

    @property
    def attr(self) -> Optional[dict]:
        """For equative: S ye NOUN ye"""
        return self._find(dep='attr', pos='NOUN')

    @property
    def tense(self) -> str:
        return self.verb.get('tense', 'pres')

    @property
    def is_neg(self) -> bool:
        return self.verb.get('is_neg', False)

    @property
    def is_passive(self) -> bool:
        return self.verb.get('is_passive', False)

    @property
    def is_refl_past(self) -> bool:
        return self.verb.get('is_refl_past', False)

    @property
    def is_imperative(self) -> bool:
        return self.tense == 'imp'

    @property
    def is_participial(self) -> bool:
        """This verb is used as a past participle adjective"""
        return _dep(self.verb) in ('acl', 'acl:relcl') and \
               self.verb.get('head') in ('NOUN','PROPN')

    # ── NP building ───────────────────────────────────────────────

    def _np(self, tok: Optional[dict]) -> str:
        if not tok: return ''
        return build_np(tok, self.all_tokens, self._used)

    def _subj_str(self) -> str:
        s = self.subject
        if not s: return ''
        self._used.add(id(s))
        # Use demonstrative bm directly (already 'o in')
        if _role(s) == 'demonstrative':
            return _bm(s)
        # Check if subject has a demonstrative determiner (cette/ce/ces)
        # -> use 'o in' for the whole subject NP
        # Mark BOTH the noun AND all its dependents as used
        s_orig = _orig(s)
        demo_det = next((t for t in self.all_tokens
                         if _role(t) == 'demonstrative'
                         and t.get('head_index', -1) == s_orig), None)
        if demo_det:
            self._used.add(id(demo_det))
            self._used.add(id(s))  # mark subject noun itself as used
            # Mark ALL dependents of subject noun as used
            for t in self.all_tokens:
                if t.get('head_index', -1) == s_orig:
                    self._used.add(id(t))
            return 'o in'
        return self._np(s)

    def _obj_str(self) -> str:
        o = self.obj
        if not o: return ''
        self._used.add(id(o))
        return self._np(o)

    def _loc_str(self) -> str:
        l = self.loc
        if not l: return ''
        self._used.add(id(l))
        return self._np(l)

    def _adv_str(self) -> str:
        parts = []
        for a in self.advs:
            if a is not self.intens:
                parts.append(_bm(a))
                self._used.add(id(a))
        return ' '.join(parts)

    def _intens_str(self) -> str:
        i = self.intens
        if not i: return ''
        self._used.add(id(i))
        return _bm(i)

    def _unused_str(self) -> str:
        """Safety net: content tokens not yet placed."""
        # Build set of used orig_indices for cross-object comparison
        used_origs = {_orig(t) for t in self.all_tokens
                      if id(t) in self._used and _orig(t) >= 0}
        parts = []
        for t in self._deps:
            if id(t) in self._used: continue
            if _orig(t) in used_origs: continue  # same token, different object
            if not _is_content(t): continue
            bm = t.get('bm')
            if not bm: continue
            if _role(t) in ('pronoun','possessive','relative') \
               and _dep(t) in ('det','poss'): continue
            if _role(t) in _CONNECTOR_ROLES: continue
            if _role(t) == 'demonstrative': continue
            parts.append(bm)
            self._used.add(id(t))
        return ' '.join(parts)

    # ── Pattern renderers ─────────────────────────────────────────

    def _render_standard(self) -> str:
        """S TAM O XCOMP V [IOBJ ye] [LOC na] [CCOMP ko] [ADV] [INTENS]"""
        tam  = get_tam(self.tense, self.is_neg)
        S    = self._subj_str()
        O    = self._obj_str()
        V    = _bm(self.verb)
        self._used.add(id(self.verb))

        # xcomp: action before attitude verb
        xc = self.xcomp
        XCOMP = _bm(xc) if xc else ''
        if xc: self._used.add(id(xc))

        # Indirect object + ye
        io = self.iobj
        IOBJ = _j(_bm(io), 'ye') if io else ''
        if io: self._used.add(id(io))

        # Dative + ma (institutional beneficiary)
        dat = self.dative
        DAT = _j(self._np(dat), 'ma') if dat else ''
        if dat: self._used.add(id(dat))

        # Locative + na
        LOC = _j(self._loc_str(), 'na') if self.loc else ''

        # ccomp: reported speech -> ko S TAM V
        ccomp_v = self.ccomp_verb
        CCOMP = ''
        if ccomp_v:
            self._used.add(id(ccomp_v))
            cc_subj = next((t for t in self.all_tokens
                            if _role(t) == 'pronoun'
                            and _dep(t) == 'nsubj'
                            and t.get('head_index',-1) == _orig(ccomp_v)),
                           None)
            ctam = get_tam(ccomp_v.get('tense','pres'),
                           ccomp_v.get('is_neg', False))
            cparts = ['ko']
            if cc_subj:
                self._used.add(id(cc_subj))
                cparts.append(_bm(cc_subj))
            if ctam: cparts.append(ctam)
            cparts.append(_bm(ccomp_v))
            # Include adverbs depending on ccomp verb
            ccomp_orig = _orig(ccomp_v)
            for t in self.all_tokens:
                if t.get('head_index', -1) == ccomp_orig and                    _pos(t) == 'ADV' and _is_content(t):
                    self._used.add(id(t))
                    cparts.append(_bm(t))
            CCOMP = ' '.join(p for p in cparts if p)

        ADV    = self._adv_str()
        INTENS = self._intens_str()
        UNUSED = self._unused_str()

        return _j(S, tam, O, XCOMP, V, IOBJ, DAT, LOC,
                  CCOMP, ADV, INTENS, UNUSED)

    def _render_passive(self) -> str:
        """S bɛ ka V [ADV] AGENT fɛ [LOC la]"""
        S     = self._subj_str()
        V     = _bm(self.verb)
        self._used.add(id(self.verb))
        ag    = self.agent
        AGENT = _j(self._np(ag), 'fɛ') if ag else ''
        if ag: self._used.add(id(ag))
        LOC   = _j(self._loc_str(), 'la') if self.loc else ''
        ADV   = self._adv_str()
        UNUSED = self._unused_str()
        return _j(S, 'bɛ ka', V, ADV, AGENT, LOC, UNUSED)

    def _render_refl_past(self) -> str:
        """
        S tun ye V [ko S TAM XCOMP O DATIVE ma] [LOC na] [ADV]

        When verb has xcomp (s'engager à assurer):
        u tun ye sèndòn ko u bɛ na sémako gáranti fanga mara ma
        S tun ye V(engager) ko S TAM O(pouvoir) V(assurer) DATIVE(junte) ma
        """
        S   = self._subj_str()
        V   = _bm(self.verb)
        self._used.add(id(self.verb))

        # Check for xcomp — render as 'ko' complement clause
        xc  = self._find(dep='xcomp')
        KO_CLAUSE = ''
        if xc:
            self._used.add(id(xc))
            xc_orig  = _orig(xc)
            # Object of xcomp
            xc_obj   = next((t for t in self.all_tokens
                             if _dep(t) == 'obj'
                             and t.get('head_index', -1) == xc_orig), None)
            # Dative of xcomp
            xc_dat   = next((t for t in self.all_tokens
                             if t.get('is_dative') and _is_content(t)
                             and t.get('head_index', -1) == xc_orig), None)
            xc_obj_bm = ''
            if xc_obj:
                self._used.add(id(xc_obj))
                xc_obj_bm = build_np(xc_obj, self.all_tokens, self._used)
            xc_dat_bm = ''
            if xc_dat:
                self._used.add(id(xc_dat))
                xc_dat_bm = build_np(xc_dat, self.all_tokens, self._used)
            # TAM for xcomp (usually future/present)
            xc_tense = xc.get('tense', 'fut')
            xc_neg   = xc.get('is_neg', False)
            xc_tam   = get_tam(xc_tense, xc_neg)
            # Subject repeat (same as main subject)
            xc_s     = _bm(self.subject) if self.subject else ''
            dat_str  = _j(xc_dat_bm, 'ma') if xc_dat_bm else ''
            KO_CLAUSE = _j('ko', xc_s, xc_tam, xc_obj_bm, _bm(xc), dat_str)

        O      = self._obj_str() if not KO_CLAUSE else ''
        LOC    = _j(self._loc_str(), 'na') if self.loc else ''
        ADV    = self._adv_str()
        UNUSED = self._unused_str()
        return _j(S, 'tun ye', V, KO_CLAUSE or O, LOC, ADV, UNUSED)

    def _render_qualitative(self) -> str:
        """S ka ADJ [INTENS]"""
        S   = self._subj_str()
        ADJ = _bm(self.root_adj)
        self._used.add(id(self.root_adj))
        self._used.add(id(self.verb))
        INTENS = self._intens_str()
        UNUSED = self._unused_str()
        return _j(S, 'ka', ADJ, INTENS, UNUSED)

    def _render_equative(self) -> str:
        """S ye ATTR ye"""
        S    = self._subj_str()
        ATTR = _bm(self.attr)
        self._used.add(id(self.attr))
        self._used.add(id(self.verb))
        UNUSED = self._unused_str()
        return _j(S, 'ye', ATTR, 'ye', UNUSED)

    def _render_imperative(self) -> str:
        V = _bm(self.verb)
        self._used.add(id(self.verb))
        O = self._obj_str()
        io = self.iobj
        IOBJ = _j(_bm(io), 'ye') if io else ''
        if io: self._used.add(id(io))
        UNUSED = self._unused_str()
        if self.is_neg:
            return _j('kana', self._subj_str(), O, V, UNUSED)
        return _j(V, IOBJ, O, UNUSED)

    # ── Pattern selector ──────────────────────────────────────────

    def to_bambara(self) -> str:
        if self.is_imperative:
            result = self._render_imperative()
        elif self.is_passive:
            result = self._render_passive()
        elif self.is_refl_past:
            result = self._render_refl_past()
        elif self.root_adj:
            result = self._render_qualitative()
        elif self.attr:
            result = self._render_equative()
        else:
            result = self._render_standard()

        # Prepend connector if present
        if self.connector:
            return _j(self.connector, result)
        return result


# ══════════════════════════════════════════════════════════════════
# PROPOSITION FINDER
# ══════════════════════════════════════════════════════════════════

_SKIP_DEPS = frozenset((
    'aux', 'auxpass', 'cop',
    'mark', 'cc', 'punct',
))

_CONNECTOR_ROLES = frozenset((
    'conditional', 'causal', 'purpose',
    'temporal', 'concessive', 'conjunction',
    'contrast', 'disjunction',
))

# Connector BM loaded from KG at runtime via load_connector_bm(db)
# Fallback used only when KG unavailable
_CONNECTOR_BM_FALLBACK = {
    'conditional': 'ní',
    'concessive':  'hali ni',
    'causal':      'sabu',
    'purpose':     'janko',
    'temporal':    'tuma min',
    'conjunction': 'ani',
    'contrast':    'kɔ́nɔ',
    'disjunction': 'wala',
}

# Runtime cache — populated by load_connector_bm(db)
_CONNECTOR_BM: dict = {}


def load_connector_bm(db) -> dict:
    """
    Load connector role -> bm mappings from KG FunctionWord nodes.
    Called once at engine startup.
    Populates _CONNECTOR_BM cache.
    """
    global _CONNECTOR_BM
    if _CONNECTOR_BM:
        return _CONNECTOR_BM  # already loaded

    if db is None:
        _CONNECTOR_BM = dict(_CONNECTOR_BM_FALLBACK)
        return _CONNECTOR_BM

    try:
        results = db.query("""
            MATCH (f:FunctionWord)
            WHERE f.role IN ['conditional','concessive','causal',
                             'purpose','temporal','conjunction',
                             'contrast','disjunction']
            RETURN f.role AS role, f.bm AS bm
        """)
        mapping = {}
        for r in results:
            role = r.get('role')
            bm   = r.get('bm')
            if role and bm and role not in mapping:
                mapping[role] = bm
        # Merge with fallback for any missing roles
        for role, bm in _CONNECTOR_BM_FALLBACK.items():
            if role not in mapping:
                mapping[role] = bm
        _CONNECTOR_BM = mapping
        print(f"     📖 Loaded {len(_CONNECTOR_BM)} connector mappings from KG")
    except Exception as e:
        print(f"     ⚠️  Could not load connectors from KG: {e}, using fallback")
        _CONNECTOR_BM = dict(_CONNECTOR_BM_FALLBACK)

    return _CONNECTOR_BM


def _split_at_commas(tokens: list) -> list:
    """Split token list into clauses at comma boundaries."""
    clauses, current = [], []
    for t in tokens:
        if t.get('role') == 'punct' and t.get('surface') == ',':
            if current:
                clauses.append(current)
                current = []
        else:
            current.append(t)
    if current:
        clauses.append(current)
    return clauses if clauses else [tokens]


def find_propositions(tokens: list) -> list:
    """
    Group tokens into propositions by verb anchor.
    Comma = hard boundary — tokens never cross clause boundaries.
    Each connector token assigned to FIRST matching verb only.
    """
    active = [t for t in tokens
              if _role(t) not in ('article', 'function_candidate')]

    # Only top-level verbs — skip embedded modifiers
    # acl/acl:relcl -> relative clause modifying a noun (handled by build_np)
    # ccomp -> reported speech complement (handled by _render_standard ccomp)
    # acl with dep='acl' -> participial adjective (handled by build_np)
    _EMBEDDED_DEPS = frozenset((
        'aux', 'auxpass', 'cop', 'mark', 'cc', 'punct',
        'acl', 'acl:relcl',   # relative clause — handled inside build_np
        'ccomp',              # reported speech — handled inline in standard
    ))

    # xcomp verbs are excluded ONLY if their parent verb is also
    # a content verb in the proposition list.
    # If parent was dropped (aller/avoir as tense aux), promote xcomp.
    all_content_origs = {_orig(t) for t in active
                         if _pos(t) == 'VERB' and _role(t) == 'content'
                         and _dep(t) not in _EMBEDDED_DEPS}

    def _keep_verb(t):
        if _dep(t) in _EMBEDDED_DEPS: return False
        if _dep(t) == 'xcomp':
            # Keep xcomp if parent is NOT a top-level content verb
            # (parent was dropped as aux -> promote this verb)
            parent_orig = t.get('head_index', -1)
            parent_is_proposition = parent_orig in all_content_origs
            return not parent_is_proposition
        return True

    content_verbs = [t for t in active
                     if _pos(t) == 'VERB'
                     and _role(t) == 'content'
                     and _keep_verb(t)]

    if not content_verbs:
        return []

    content_verbs.sort(key=lambda t: _orig(t))

    # Split active tokens into clauses at comma boundaries
    # Each proposition only sees tokens from its own clause
    clauses = _split_at_commas(active)
    # Map each verb to its clause
    def _clause_for_verb(verb):
        verb_orig = _orig(verb)
        for clause in clauses:
            if any(_orig(t) == verb_orig for t in clause):
                return clause
        return active  # fallback

    # Track which connector tokens have been assigned
    used_connectors = set()

    propositions = []
    for verb in content_verbs:
        clause_tokens = _clause_for_verb(verb)
        prop = Proposition(verb=verb, all_tokens=clause_tokens)
        verb_orig = _orig(verb)

        # 1. Connector that directly depends on this verb
        for t in active:
            if id(t) in used_connectors: continue
            if _role(t) not in _CONNECTOR_ROLES: continue
            if t.get('head_index', -1) == verb_orig:
                bm_map = _CONNECTOR_BM or _CONNECTOR_BM_FALLBACK
                prop.connector = t.get('bm', '') or \
                                 bm_map.get(_role(t), '')
                used_connectors.add(id(t))
                break

        # 2. Global marker before this verb (si, quand, bien que...)
        if not prop.connector:
            for t in active:
                if id(t) in used_connectors: continue
                if _role(t) not in _CONNECTOR_ROLES: continue
                if _dep(t) in ('mark', 'cc', 'advmod') and \
                   _orig(t) < verb_orig:
                    bm_map = _CONNECTOR_BM or _CONNECTOR_BM_FALLBACK
                    prop.connector = t.get('bm', '') or \
                                     bm_map.get(_role(t), '')
                    used_connectors.add(id(t))
                    break

        propositions.append(prop)

    return propositions



# ══════════════════════════════════════════════════════════════════
# ASSEMBLER
# ══════════════════════════════════════════════════════════════════

def assemble(propositions: list) -> str:
    """
    Join propositions.

    Rules:
    - Propositions with connector: prepend connector
    - Adjacent propositions separated by comma
    - xcomp propositions (infinitive): preceded by 'ko'
    - Participial propositions: folded into parent NP (handled in build_np)
    """
    if not propositions:
        return ''

    parts = []
    for prop in propositions:
        bm = prop.to_bambara().strip()
        if not bm:
            continue

        # xcomp verbs are now excluded from propositions entirely
        # (handled as XCOMP slot in parent) — no ko prefix needed here

        parts.append(bm)

    # Join: use comma between independent clauses
    # Adjacent propositions without connector get comma
    result_parts = []
    for i, part in enumerate(parts):
        if i == 0:
            result_parts.append(part)
        else:
            # Check if this proposition already starts with a connector
            prop = propositions[i] if i < len(propositions) else None
            if prop and prop.connector:
                result_parts.append(part)
            else:
                result_parts.append(part)

    return ', '.join(result_parts)


# ══════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════

def translate_propositions(tokens: list) -> str:
    """
    Main entry point.
    Split tokens into propositions → translate each → assemble.
    """
    propositions = find_propositions(tokens)

    if not propositions:
        # No verbs found — try noun phrase
        return _translate_noun_phrase(tokens)

    return assemble(propositions)


def _translate_noun_phrase(tokens: list) -> str:
    """
    Verbless sentence parser (headlines, titles, noun phrases).

    Strategy: find LINK words as boundaries, split into NP groups,
    build each group, assemble with link semantics.

    Links:
      de/du/des  -> genitive: NP2 ka NP1  (possessor before possessed)
      au/à/dans  -> locative: NP la/na
      pour       -> purpose:  janko NP
      avec/ni    -> comitative: ni NP
      parenthetical () -> kept as-is

    e.g. 'Le bilan catastrophique des paramilitaires russes au Sahel'
      -> [bilan catastrophique] [des] [paramilitaires russes] [au] [Sahel]
      -> paramilitairew [bàlawuma] ka bilan bàlawuma Sahel la
    """
    _GENITIVE  = {'de', 'du', 'des', "d'"}
    _LOCATIVE  = {'au', 'à', 'en', 'dans', 'sur', 'chez',
                  'in', 'at', 'on'}
    _PURPOSE   = {'pour', 'for'}
    _COMITAT   = {'avec', 'with'}
    _ALL_LINKS = _GENITIVE | _LOCATIVE | _PURPOSE | _COMITAT

    # Work with original tokens (keep prepositions as boundary markers)
    toks = [t for t in tokens if _role(t) != 'punct']

    # Find link token positions
    link_positions = [(i, t) for i, t in enumerate(toks)
                      if t.get('surface','').lower() in _ALL_LINKS
                      or t.get('role') == 'preposition'
                      and t.get('surface','').lower() in _ALL_LINKS]

    # Split into NP groups at link boundaries
    groups   = []
    link_ops = []
    prev     = 0
    for idx, link_tok in link_positions:
        group = [t for t in toks[prev:idx]
                 if _role(t) not in ('article','preposition',
                                      'auxiliary','function_candidate')]
        if group:
            groups.append(group)
            link_ops.append((link_tok.get('surface','').lower(),
                             link_tok.get('role','')))
        prev = idx + 1
    # Last group
    last = [t for t in toks[prev:]
            if _role(t) not in ('article','preposition',
                                 'auxiliary','function_candidate')]
    if last:
        groups.append(last)

    if not groups:
        # No links found — single NP
        active = [t for t in toks
                  if _role(t) not in ('article','preposition',
                                       'auxiliary','function_candidate')]
        head = next((t for t in active
                     if _dep(t) == 'ROOT'
                     and _pos(t) in ('NOUN','PROPN')), None) or                next((t for t in active
                     if _pos(t) in ('NOUN','PROPN')
                     and _is_content(t)), None)
        if not head:
            return ' '.join(t.get('bm', t.get('lemma',''))
                            for t in active if _is_content(t))
        used = set()
        return build_np(head, active, used)

    # Build each NP group
    def _build_group(group: list) -> str:
        if not group: return ''
        # Find head noun (ROOT dep or first noun)
        head = next((t for t in group if _dep(t) == 'ROOT'
                     and _pos(t) in ('NOUN','PROPN','ADJ')), None) or                next((t for t in group
                     if _pos(t) in ('NOUN','PROPN')
                     and _is_content(t)), None)
        if not head:
            # All adjectives or no clear head
            return ' '.join(t.get('bm','') for t in group
                            if _is_content(t) and t.get('bm'))
        used = set()
        return build_np(head, group, used)

    np_parts = [_build_group(g) for g in groups]

    # Assemble with link semantics
    # link_ops[i] = link between np_parts[i] and np_parts[i+1]
    result = np_parts[0] if np_parts else ''

    for i, (link_surf, link_role) in enumerate(link_ops):
        if i + 1 >= len(np_parts): break
        next_np = np_parts[i + 1]
        if not next_np: continue

        if link_surf in _GENITIVE:
            # Genitive: POSSESSOR before POSSESSED
            # Swap: next_np ka result (possessor first)
            result = _j(next_np, result) if not result                      else _j(next_np, result)
        elif link_surf in _LOCATIVE:
            # Locative: append LOC la/na at end
            result = _j(result, next_np, 'la')
        elif link_surf in _PURPOSE:
            result = _j(result, 'janko', next_np)
        elif link_surf in _COMITAT:
            result = _j(result, 'ni', next_np)
        else:
            result = _j(result, next_np)

    return result