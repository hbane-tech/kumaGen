"""
eval/parser_eval.py — Évaluation du parsing contre UD French GSD (gold standard)
================================================================================
Évalue le pipeline de tagging de Kuma-MT (spaCy + corrections _fix_pos_errors)
contre le treebank Universal Dependencies UD_French-GSD, sans annotation manuelle.

Métriques standard (CoNLL / UD shared task) :
  UPOS  : Universal POS Accuracy
  UAS   : Unlabeled Attachment Score   (head correct)
  LAS   : Labeled Attachment Score     (head + deprel corrects)
  LAS-c : LAS sur label principal       (avant ':' — tolérant aux sous-types)

Alignement tokenisation système ↔ gold par spans de caractères (méthode
officielle UD : deux tokens sont comparables ssi ils couvrent exactement le
même intervalle de caractères non-espaces). Les tokens non-alignés (divergence
de tokenisation) sont comptés mais non scorés pour POS/dep.

Usage :
  python -m eval.parser_eval                      # GSD complet
  python -m eval.parser_eval --limit 100          # 100 premières phrases
  python -m eval.parser_eval --verbose            # erreurs détaillées
"""

import os
import sys
import argparse
from collections import defaultdict

UD_TEST = os.path.join(os.path.dirname(__file__), 'ud', 'fr_gsd-ud-test.conllu')


# ─────────────────────────────────────────────────────────────────────────────
# Parsing CoNLL-U (zéro dépendance)
# ─────────────────────────────────────────────────────────────────────────────
def read_conllu(path):
    """
    Yield (text, tokens, units) :
      tokens : tokens syntaxiques [{id, form, upos, head, deprel}]
      units  : unités de SURFACE [{surface, ids}] où ids = tokens syntaxiques
               couverts. Un multiword UD (du = de+le) → une unit 'du' couvrant
               [id_de, id_le]. Sert à reconstruire la surface réelle pour
               l'alignement de tokenisation.
    """
    sents = []
    text, toks, units = None, [], []
    mwt_until = -1   # fin de la plage multiword en cours

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('# text ='):
                text = line.split('=', 1)[1].strip()
            elif line.startswith('#'):
                continue
            elif line == '':
                if toks:
                    sents.append((text, toks, units))
                text, toks, units = None, [], []
                mwt_until = -1
            else:
                cols = line.split('\t')
                tid = cols[0]
                if '.' in tid:               # nœud vide → ignorer
                    continue
                if '-' in tid:               # multiword token (ex: 7-8 du)
                    a, b = tid.split('-')
                    units.append({'surface': cols[1],
                                  'ids': list(range(int(a), int(b) + 1))})
                    mwt_until = int(b)
                    continue
                _id = int(tid)
                toks.append({
                    'id':     _id,
                    'form':   cols[1],
                    'upos':   cols[3],
                    'head':   int(cols[6]) if cols[6] != '_' else 0,
                    'deprel': cols[7],
                })
                if _id > mwt_until:          # token simple (hors multiword)
                    units.append({'surface': cols[1], 'ids': [_id]})
        if toks:
            sents.append((text, toks, units))
    return sents


# ─────────────────────────────────────────────────────────────────────────────
# Alignement par spans de caractères non-espaces
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(s):
    return (s.lower()
             .replace('’', "'").replace('‘', "'")
             .replace('«', '"').replace('»', '"')
             .replace('œ', 'oe'))   # œ → oe


def _char_spans(surfaces):
    """Retourne [(start, end)] de chaque token dans la concat sans espaces."""
    spans, pos = [], 0
    for s in surfaces:
        cs = _normalize(s).replace(' ', '')
        spans.append((pos, pos + len(cs)))
        pos += len(cs)
    return spans


def align(sys_tokens, units):
    """
    Aligne les tokens système aux UNITÉS DE SURFACE gold par span de caractères.
    Retourne aligned = liste de (sys_idx, unit) pour les spans identiques.

    Un sys token aligné à une unit mono-token (ids=[x]) est scorable.
    Aligné à une unit multiword (du = [de,le]) → non scorable (skip).
    """
    sys_spans  = _char_spans([t['surface']  for t in sys_tokens])
    unit_spans = _char_spans([u['surface']  for u in units])

    unit_by_span = {sp: u for sp, u in zip(unit_spans, units)}
    aligned = []
    for si, sp in enumerate(sys_spans):
        u = unit_by_span.get(sp)
        if u is not None:
            aligned.append((si, u))
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation deprel (spaCy → UD)
# ─────────────────────────────────────────────────────────────────────────────
def norm_deprel(dep):
    d = (dep or '').lower()
    if d == 'root':
        return 'root'
    return d


def main_label(dep):
    return norm_deprel(dep).split(':')[0]


# ─────────────────────────────────────────────────────────────────────────────
# Groupes d'équivalence fonctionnelle (pour le rendu bambara)
# ─────────────────────────────────────────────────────────────────────────────
# Deux étiquettes du même groupe déclenchent le MÊME traitement dans le pipeline
# → une confusion intra-groupe n'affecte pas la sortie bambara ("bénigne").
# Une confusion inter-groupe peut changer la structure de clause ("impactante").
# Chaque groupe est justifié par une distinction que le bambara NE fait PAS.

# POS : le bambara ne distingue pas nom propre/commun (même slot, même rendu),
# et ignore les symboles/ponctuation/tokens inclassables.
POS_EQUIV = [
    {'NOUN', 'PROPN'},          # même slot nominal
    {'SYM', 'X', 'PUNCT'},      # non traduits
]

# Dépendances : groupes rendus identiquement en bambara.
DEP_EQUIV = [
    # Tous les obliques → wagon oblique identique (pas de distinction arg/mod)
    {'obl', 'obl:mod', 'obl:arg', 'obl:agent'},
    # Modifieurs nominaux → chaîne génitive (le bambara ne distingue pas
    # apposition / modifieur / nom plat)
    {'nmod', 'appos', 'flat', 'flat:name'},
    # Explétifs → tous supprimés en bambara
    {'expl', 'expl:subj', 'expl:comp', 'expl:pass', 'expl:pv', 'expl:impers'},
    # Constructions à verbe support / objets → même rendu objet
    {'obj', 'obj:lvc', 'obj:agent'},
    # Variantes de parataxe
    {'parataxis', 'parataxis:insert', 'parataxis:appos'},
]


def _same_group(a, b, groups):
    if a == b:
        return True
    for g in groups:
        if a in g and b in g:
            return True
    return False


def equiv_pos(gold, sys):
    return _same_group((gold or '').upper(), (sys or '').upper(), POS_EQUIV)


def equiv_dep(gold, sys):
    return _same_group(norm_deprel(gold), norm_deprel(sys), DEP_EQUIV)


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(limit=None, verbose=False, export=True):
    import warnings; warnings.filterwarnings('ignore')

    if not os.path.exists(UD_TEST):
        print(f"❌ Treebank introuvable : {UD_TEST}")
        print("   Télécharge : curl -sL -o eval/ud/fr_gsd-ud-test.conllu \\")
        print("     https://raw.githubusercontent.com/UniversalDependencies/"
              "UD_French-GSD/master/fr_gsd-ud-test.conllu")
        sys.exit(1)

    sents = read_conllu(UD_TEST)
    if limit:
        sents = sents[:limit]
    print(f"UD French GSD : {len(sents)} phrases\n")

    from pipeline.tokenizer import tokenize
    from kg.neo4j_client import Neo4jClient
    db = Neo4jClient()

    n_upos_ok = n_uas_ok = n_las_ok = n_lasc_ok = 0
    n_scored = n_gold_tokens = n_sys_tokens = 0
    # Compteurs "effectifs" : confusion bénigne (intra-groupe) comptée correcte
    n_upos_eff = n_las_eff = 0
    n_pos_benign = n_pos_impact = 0    # erreurs POS par nature
    n_dep_benign = n_dep_impact = 0    # erreurs dep par nature
    confusion_pos = defaultdict(int)
    confusion_dep = defaultdict(int)   # erreurs de LABEL pur (head correct)
    worst = []

    for i, (text, gold, units) in enumerate(sents, 1):
        if not text:
            continue
        print(f"\r  [{i:4d}/{len(sents)}] {text[:48]:<48}", end='', flush=True)

        try:
            sys_toks = tokenize(text, db=db)
        except Exception:
            sys_toks = []

        n_gold_tokens += len(gold)
        n_sys_tokens  += len(sys_toks)

        aligned = align(sys_toks, units)            # [(sys_idx, unit)]
        gold_by_id = {g['id']: g for g in gold}

        # map: sys orig_index → gold id (uniquement pour les units mono-token)
        sysorig2goldid = {}
        for si, u in aligned:
            if len(u['ids']) == 1:
                sysorig2goldid[sys_toks[si].get('orig_index')] = u['ids'][0]

        sent_ok = sent_tot = 0
        for si, u in aligned:
            if len(u['ids']) != 1:
                continue                            # multiword → non scorable
            st = sys_toks[si]
            gt = gold_by_id[u['ids'][0]]
            n_scored += 1

            # UPOS
            upos_ok = (st.get('pos', '').upper() == gt['upos'].upper())
            if upos_ok:
                n_upos_ok += 1
                n_upos_eff += 1
            else:
                confusion_pos[(gt['upos'], st.get('pos', '∅'))] += 1
                # Bénigne si gold/sys dans le même groupe d'équivalence
                if equiv_pos(gt['upos'], st.get('pos', '')):
                    n_pos_benign += 1
                    n_upos_eff += 1          # n'affecte pas la sortie → effectif OK
                else:
                    n_pos_impact += 1

            # Résoudre le head système → id gold
            sys_head_idx = st.get('head_index', -1)
            sys_orig     = st.get('orig_index', -1)
            is_sys_root  = (sys_head_idx == sys_orig or sys_head_idx < 0)
            if is_sys_root:
                sys_head_goldid = 0
            else:
                sys_head_goldid = sysorig2goldid.get(sys_head_idx, -999)

            uas_ok = (sys_head_goldid == gt['head'])
            if uas_ok:
                n_uas_ok += 1

            dep_ok  = (norm_deprel(st.get('dep')) == norm_deprel(gt['deprel']))
            depc_ok = (main_label(st.get('dep')) == main_label(gt['deprel']))
            if uas_ok and dep_ok:
                n_las_ok += 1
                n_las_eff += 1
            if uas_ok and depc_ok:
                n_lasc_ok += 1
            # Confusion sur le LABEL seul (head correct mais label faux)
            if uas_ok and not dep_ok:
                confusion_dep[(norm_deprel(gt['deprel']),
                               norm_deprel(st.get('dep')))] += 1
                # Bénigne si même groupe d'équivalence fonctionnelle
                if equiv_dep(gt['deprel'], st.get('dep')):
                    n_dep_benign += 1
                    n_las_eff += 1           # head OK + label équivalent → effectif OK
                else:
                    n_dep_impact += 1

            sent_tot += 1
            if upos_ok and uas_ok and dep_ok:
                sent_ok += 1

        if sent_tot:
            worst.append((sent_ok / sent_tot, text, sent_tot))

    print()
    n_aligned = n_scored
    eff = {
        'upos_eff': n_upos_eff, 'las_eff': n_las_eff,
        'pos_benign': n_pos_benign, 'pos_impact': n_pos_impact,
        'dep_benign': n_dep_benign, 'dep_impact': n_dep_impact,
    }
    _report(n_upos_ok, n_uas_ok, n_las_ok, n_lasc_ok, n_aligned,
            n_gold_tokens, n_sys_tokens, confusion_pos, confusion_dep, worst, eff)

    if export:
        import json, datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary = {
            'treebank': 'UD_French-GSD',
            'n_sentences': len(sents),
            'n_gold_tokens': n_gold_tokens,
            'n_scored': n_aligned,
            'scored_rate': round(n_aligned / n_gold_tokens * 100, 2) if n_gold_tokens else 0,
            'UPOS': round(n_upos_ok / n_aligned * 100, 2) if n_aligned else 0,
            'UAS':  round(n_uas_ok  / n_aligned * 100, 2) if n_aligned else 0,
            'LAS':  round(n_las_ok  / n_aligned * 100, 2) if n_aligned else 0,
            'LAS_main': round(n_lasc_ok / n_aligned * 100, 2) if n_aligned else 0,
            # Métriques effectives (confusions bénignes neutralisées)
            'UPOS_eff': round(n_upos_eff / n_aligned * 100, 2) if n_aligned else 0,
            'LAS_eff':  round(n_las_eff  / n_aligned * 100, 2) if n_aligned else 0,
            'pos_errors_benign':  n_pos_benign,
            'pos_errors_impact':  n_pos_impact,
            'dep_errors_benign':  n_dep_benign,
            'dep_errors_impact':  n_dep_impact,
            'impactful_error_rate': round(
                (n_pos_impact + n_dep_impact) /
                max(1, n_pos_benign + n_pos_impact + n_dep_benign + n_dep_impact) * 100, 2),
            # Confusions (top 30) pour les graphiques
            'confusion_pos': {f'{g}->{s}': c
                              for (g, s), c in sorted(confusion_pos.items(),
                                                      key=lambda x: -x[1])[:30]},
            'confusion_dep': {f'{g}->{s}': c
                              for (g, s), c in sorted(confusion_dep.items(),
                                                      key=lambda x: -x[1])[:30]},
        }
        path = f'parser_eval_{ts}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  JSON exporté : {path}")


def _report(upos, uas, las, lasc, n, n_gold, n_sys,
            conf_pos, conf_dep, worst, eff=None):
    print(f"\n{'═'*60}")
    print(f"  PARSING vs UD French GSD  (pipeline complet)")
    print(f"{'═'*60}")
    align_rate = n / n_gold * 100 if n_gold else 0
    print(f"  Tokens gold        : {n_gold}")
    print(f"  Tokens système     : {n_sys}")
    print(f"  Tokens alignés     : {n} ({align_rate:.1f}% du gold)")
    print(f"\n  ── Métriques standard ──")
    print(f"  UPOS  POS Accuracy            : {upos/n*100:6.2f}%")
    print(f"  UAS   Unlabeled Attach. Score : {uas/n*100:6.2f}%")
    print(f"  LAS   Labeled Attach. Score   : {las/n*100:6.2f}%")
    print(f"  LAS-c LAS (label principal)   : {lasc/n*100:6.2f}%")

    if eff:
        # Métriques "effectives" : confusions bénignes (intra-groupe) neutralisées
        upos_eff = eff['upos_eff'] / n * 100 if n else 0
        las_eff  = eff['las_eff']  / n * 100 if n else 0
        pos_err  = eff['pos_benign'] + eff['pos_impact']
        dep_err  = eff['dep_benign'] + eff['dep_impact']
        pos_imp_pct = eff['pos_impact'] / pos_err * 100 if pos_err else 0
        dep_imp_pct = eff['dep_impact'] / dep_err * 100 if dep_err else 0
        print(f"\n  ── Métriques effectives (impact sur le rendu bambara) ──")
        print(f"  UPOS* POS effectif            : {upos_eff:6.2f}%  "
              f"(+{upos_eff - upos/n*100:.2f})")
        print(f"  LAS*  LAS effectif            : {las_eff:6.2f}%  "
              f"(+{las_eff - las/n*100:.2f})")
        print(f"\n  Erreurs POS  : {pos_err:>4}  "
              f"→ bénignes {eff['pos_benign']} | impactantes {eff['pos_impact']} "
              f"({pos_imp_pct:.0f}%)")
        print(f"  Erreurs dep  : {dep_err:>4}  "
              f"→ bénignes {eff['dep_benign']} | impactantes {eff['dep_impact']} "
              f"({dep_imp_pct:.0f}%)")
        _tot_err = pos_err + dep_err
        _tot_imp = eff['pos_impact'] + eff['dep_impact']
        print(f"\n  ➜ {_tot_imp}/{_tot_err} erreurs de parsing affectent réellement "
              f"la sortie ({_tot_imp/_tot_err*100:.0f}%)" if _tot_err else "")

    if conf_pos:
        print(f"\n  ── Top confusions POS (gold → sys) ──")
        for (g, s), c in sorted(conf_pos.items(), key=lambda x: -x[1])[:8]:
            tag = 'bénigne' if equiv_pos(g, s) else 'IMPACT'
            print(f"     {g:>8} → {s:<8} : {c:>3}  [{tag}]")
    if conf_dep:
        print(f"\n  ── Top confusions dep — label seul, head correct (gold → sys) ──")
        for (g, s), c in sorted(conf_dep.items(), key=lambda x: -x[1])[:8]:
            tag = 'bénigne' if equiv_dep(g, s) else 'IMPACT'
            print(f"     {g:>14} → {s:<14} : {c:>3}  [{tag}]")
    print(f"{'═'*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing eval vs UD French GSD')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-export', action='store_true')
    args = parser.parse_args()
    evaluate(limit=args.limit, verbose=args.verbose, export=not args.no_export)
