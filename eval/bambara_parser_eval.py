"""
eval/bambara_parser_eval.py — Évaluation du parseur UDPipe bambara contre
UD_Bambara-CRB (gold standard)
================================================================================
Évalue le modèle UDPipe entraîné sur le treebank UD_Bambara (Aplonova & Tyers,
distribué dans eval/bambara_udpipe.py, model source :
https://github.com/KatyaAplonova/UD_Bambara) contre le gold standard officiel
UniversalDependencies/UD_Bambara-CRB — pour savoir combien on peut faire
confiance à bambara_pos_tree/bambara_dep_tree (voir test_phrases.py) avant de
s'en servir comme métrique dans eval_report.py.

Contrairement à eval/parser_eval.py (qui évalue NOTRE PROPRE pipeline français
et son impact sur le rendu bambara via des groupes d'équivalence), ce script
évalue un outil TIERS déjà publié sur SES PROPRES données gold — pas de notion
de "confusion bénigne pour le rendu", juste la qualité brute du parseur.

Métriques standard (CoNLL / UD shared task) :
  UPOS  : Universal POS Accuracy
  UAS   : Unlabeled Attachment Score   (head correct)
  LAS   : Labeled Attachment Score     (head + deprel corrects)
  LAS-c : LAS sur label principal       (avant ':' — tolérant aux sous-types)

Alignement système ↔ gold par spans de caractères non-espaces (même méthode
que eval/parser_eval.py) — nécessaire même si le modèle tokenize lui-même le
texte brut : sa propre segmentation peut diverger de l'annotation gold.

Usage :
  python -m eval.bambara_parser_eval                # treebank complet
  python -m eval.bambara_parser_eval --limit 200    # 200 premières phrases
"""

import os
import sys
import argparse
from collections import defaultdict

UD_TEST = os.path.join(os.path.dirname(__file__), 'ud', 'bm_crb-ud-test.conllu')
UD_TEST_URL = ("https://raw.githubusercontent.com/UniversalDependencies/"
               "UD_Bambara-CRB/master/bm_crb-ud-test.conllu")


# ─────────────────────────────────────────────────────────────────────────────
# Parsing CoNLL-U (même logique que eval/parser_eval.py)
# ─────────────────────────────────────────────────────────────────────────────
def read_conllu(path):
    """Yield (text, tokens, units) — voir eval/parser_eval.py:read_conllu."""
    sents = []
    text, toks, units = None, [], []
    mwt_until = -1

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
                if '.' in tid:
                    continue
                if '-' in tid:
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
                if _id > mwt_until:
                    units.append({'surface': cols[1], 'ids': [_id]})
        if toks:
            sents.append((text, toks, units))
    return sents


def download_treebank(dest=UD_TEST):
    if os.path.exists(dest):
        return True
    try:
        import urllib.request
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(UD_TEST_URL, dest)
        return os.path.exists(dest)
    except Exception as e:
        print(f"  Échec du téléchargement du treebank UD_Bambara-CRB : {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Alignement par spans de caractères non-espaces (identique à parser_eval.py)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(s):
    return (s or '').lower().replace('’', "'").replace('‘', "'")


def _char_spans(surfaces):
    spans, pos = [], 0
    for s in surfaces:
        cs = _normalize(s).replace(' ', '')
        spans.append((pos, pos + len(cs)))
        pos += len(cs)
    return spans


def align(sys_tokens, units):
    """sys_tokens: [{'id','form',...}] (1-based id, UDPipe output).
    units: gold surface units. Retourne [(sys_idx, unit)] pour spans identiques."""
    sys_spans  = _char_spans([t['form'] for t in sys_tokens])
    unit_spans = _char_spans([u['surface'] for u in units])
    unit_by_span = {sp: u for sp, u in zip(unit_spans, units)}
    aligned = []
    for si, sp in enumerate(sys_spans):
        u = unit_by_span.get(sp)
        if u is not None:
            aligned.append((si, u))
    return aligned


def norm_deprel(dep):
    d = (dep or '').lower()
    return 'root' if d == 'root' else d


def main_label(dep):
    return norm_deprel(dep).split(':')[0]


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(limit=None, export=True):
    import warnings; warnings.filterwarnings('ignore')

    if not os.path.exists(UD_TEST) and not download_treebank():
        print(f" Treebank introuvable : {UD_TEST}")
        print(f"   Télécharge : curl -sL -o {UD_TEST} {UD_TEST_URL}")
        sys.exit(1)

    from eval.bambara_udpipe import parse_bambara

    sents = read_conllu(UD_TEST)
    if limit:
        sents = sents[:limit]
    print(f"UD_Bambara-CRB : {len(sents)} phrases\n")

    n_upos_ok = n_uas_ok = n_las_ok = n_lasc_ok = 0
    n_scored = n_gold_tokens = n_sys_tokens = 0
    confusion_pos = defaultdict(int)
    confusion_dep = defaultdict(int)
    worst = []

    for i, (text, gold, units) in enumerate(sents, 1):
        if not text:
            continue
        print(f"\r  [{i:4d}/{len(sents)}] {text[:48]:<48}", end='', flush=True)

        try:
            sys_toks = parse_bambara(text)
        except Exception:
            sys_toks = []

        n_gold_tokens += len(gold)
        n_sys_tokens  += len(sys_toks)

        aligned = align(sys_toks, units)
        gold_by_id = {g['id']: g for g in gold}

        # map: sys token id (1-based, UDPipe) → gold id, pour résoudre les heads
        sysid2goldid = {}
        for si, u in aligned:
            if len(u['ids']) == 1:
                sysid2goldid[sys_toks[si]['id']] = u['ids'][0]

        sent_ok = sent_tot = 0
        for si, u in aligned:
            if len(u['ids']) != 1:
                continue
            st = sys_toks[si]
            gt = gold_by_id[u['ids'][0]]
            n_scored += 1

            upos_ok = (st['upos'].upper() == gt['upos'].upper())
            if upos_ok:
                n_upos_ok += 1
            else:
                confusion_pos[(gt['upos'], st['upos'] or '∅')] += 1

            sys_head_goldid = 0 if st['head'] == 0 else sysid2goldid.get(st['head'], -999)
            uas_ok = (sys_head_goldid == gt['head'])
            if uas_ok:
                n_uas_ok += 1

            dep_ok  = (norm_deprel(st['deprel']) == norm_deprel(gt['deprel']))
            depc_ok = (main_label(st['deprel']) == main_label(gt['deprel']))
            if uas_ok and dep_ok:
                n_las_ok += 1
            if uas_ok and depc_ok:
                n_lasc_ok += 1
            if uas_ok and not dep_ok:
                confusion_dep[(norm_deprel(gt['deprel']), norm_deprel(st['deprel']))] += 1

            sent_tot += 1
            if upos_ok and uas_ok and dep_ok:
                sent_ok += 1

        if sent_tot:
            worst.append((sent_ok / sent_tot, text, sent_tot))

    print()
    _report(n_upos_ok, n_uas_ok, n_las_ok, n_lasc_ok, n_scored,
            n_gold_tokens, n_sys_tokens, confusion_pos, confusion_dep, worst)

    if export:
        import json, datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary = {
            'treebank': 'UD_Bambara-CRB',
            'model': 'eval/ud/bambara.model (UDPipe, Aplonova & Tyers)',
            'n_sentences': len(sents),
            'n_gold_tokens': n_gold_tokens,
            'n_scored': n_scored,
            'scored_rate': round(n_scored / n_gold_tokens * 100, 2) if n_gold_tokens else 0,
            'UPOS': round(n_upos_ok / n_scored * 100, 2) if n_scored else 0,
            'UAS':  round(n_uas_ok  / n_scored * 100, 2) if n_scored else 0,
            'LAS':  round(n_las_ok  / n_scored * 100, 2) if n_scored else 0,
            'LAS_main': round(n_lasc_ok / n_scored * 100, 2) if n_scored else 0,
            'confusion_pos': {f'{g}->{s}': c
                              for (g, s), c in sorted(confusion_pos.items(),
                                                      key=lambda x: -x[1])[:30]},
            'confusion_dep': {f'{g}->{s}': c
                              for (g, s), c in sorted(confusion_dep.items(),
                                                      key=lambda x: -x[1])[:30]},
        }
        path = f'eval/bambara_parser_eval_{ts}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  JSON exporté : {path}")

    return {
        'upos': n_upos_ok / n_scored if n_scored else 0,
        'uas':  n_uas_ok  / n_scored if n_scored else 0,
        'las':  n_las_ok  / n_scored if n_scored else 0,
    }


def _report(upos, uas, las, lasc, n, n_gold, n_sys, conf_pos, conf_dep, worst):
    print(f"\n{'═'*60}")
    print(f"  PARSING vs UD_Bambara-CRB  (modèle UDPipe / UD_Bambara)")
    print(f"{'═'*60}")
    align_rate = n / n_gold * 100 if n_gold else 0
    print(f"  Tokens gold        : {n_gold}")
    print(f"  Tokens système     : {n_sys}")
    print(f"  Tokens alignés     : {n} ({align_rate:.1f}% du gold)")
    if n:
        print(f"\n  ── Métriques standard ──")
        print(f"  UPOS  POS Accuracy            : {upos/n*100:6.2f}%")
        print(f"  UAS   Unlabeled Attach. Score : {uas/n*100:6.2f}%")
        print(f"  LAS   Labeled Attach. Score   : {las/n*100:6.2f}%")
        print(f"  LAS-c LAS (label principal)   : {lasc/n*100:6.2f}%")
    if conf_pos:
        print(f"\n  ── Top confusions POS (gold → sys) ──")
        for (g, s), c in sorted(conf_pos.items(), key=lambda x: -x[1])[:8]:
            print(f"     {g:>8} → {s:<8} : {c:>3}")
    if conf_dep:
        print(f"\n  ── Top confusions dep — label seul, head correct (gold → sys) ──")
        for (g, s), c in sorted(conf_dep.items(), key=lambda x: -x[1])[:8]:
            print(f"     {g:>14} → {s:<14} : {c:>3}")
    if worst:
        print(f"\n  ── 5 phrases les moins bien parsées ──")
        for score, text, tot in sorted(worst)[:5]:
            print(f"     {score*100:5.1f}%  ({tot:>2} tok)  {text[:60]}")
    print(f"{'═'*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing eval vs UD_Bambara-CRB')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no-export', action='store_true')
    args = parser.parse_args()
    evaluate(limit=args.limit, export=not args.no_export)
