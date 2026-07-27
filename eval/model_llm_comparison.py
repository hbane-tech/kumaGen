"""
eval/model_llm_comparison.py
=============================
Compare Qwen2.5:3b (current production LLM, via Ollama) against GPT-4 and
GPT-3.5-turbo on the semantic-inference classification tasks that
pipeline/translation_engine.py actually delegates to an LLM.

This is NOT an end-to-end translation eval (see eval/evaluate.py for that).
It isolates each individual LLM decision point (possession-type classification,
statif/qualitative adjective classification, reflexive-verb typing, etc.),
reproduces the EXACT prompt used in production, and scores each model's
answer against a hand-labeled gold answer.

Gold labels were assigned from the linguistic definitions written directly
in the production prompts (pipeline/translation_engine.py), so they reflect
what the pipeline itself considers "correct" -- not an external standard.

Usage:
  python eval/model_llm_comparison.py                 # run all tasks, all models
  python eval/model_llm_comparison.py --task possession_type
  python eval/model_llm_comparison.py --models qwen,gpt-3.5-turbo
"""

import argparse
import csv
import datetime
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (OLLAMA_GENERATE_URL, OPENAI_API_KEY,  # noqa: E402
                              GEMINI_API_KEY, GEMINI_MODEL)

import requests  # noqa: E402

# ---------------------------------------------------------------------------
# MODEL BACKENDS
# ---------------------------------------------------------------------------

QWEN_MODEL = 'qwen2.5:3b'

# Secondary local Ollama instance (default port 11434) holding free models
# that are NOT the production engine's Qwen instance (which runs on the
# OLLAMA_HOST port configured in config/settings.py, typically 11435).
OLLAMA_LOCAL_URL = os.getenv('OLLAMA_LOCAL_URL', 'http://localhost:11434') + '/api/generate'


def _call_ollama(url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> str:
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {'temperature': 0, 'num_predict': max_tokens},
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.json().get('response', '').strip()
    except Exception as e:
        return f'__ERROR__:{type(e).__name__}'


def call_qwen(prompt: str, max_tokens: int = 10, timeout: int = 60) -> str:
    return _call_ollama(OLLAMA_GENERATE_URL, QWEN_MODEL, prompt, max_tokens, timeout)


_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def call_openai(model: str, prompt: str, max_tokens: int = 10, timeout: int = 60) -> str:
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return (resp.choices[0].message.content or '').strip()
    except Exception as e:
        return f'__ERROR__:{type(e).__name__}:{e}'


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


_GEMINI_MIN_INTERVAL = 4.5  # seconds; free tier ~ 15 req/min
_gemini_last_call = 0.0


def call_gemini(prompt: str, max_tokens: int = 10, timeout: int = 60) -> str:
    global _gemini_last_call
    from google.genai import types
    client = _get_gemini_client()
    last_err = None
    for attempt in range(3):
        wait = _GEMINI_MIN_INTERVAL - (time.time() - _gemini_last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL or 'gemini-2.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0, max_output_tokens=max_tokens),
            )
            _gemini_last_call = time.time()
            return (resp.text or '').strip()
        except Exception as e:
            _gemini_last_call = time.time()
            last_err = e
            if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                time.sleep(8 * (attempt + 1))
                continue
            break
    return f'__ERROR__:{type(last_err).__name__}:{last_err}'


MODEL_BACKENDS = {
    'qwen': lambda prompt, max_tokens: call_qwen(prompt, max_tokens),
    'gpt-4': lambda prompt, max_tokens: call_openai('gpt-4', prompt, max_tokens),
    'gpt-3.5-turbo': lambda prompt, max_tokens: call_openai('gpt-3.5-turbo', prompt, max_tokens),
    'gemini': lambda prompt, max_tokens: call_gemini(prompt, max_tokens),
    'llama3.2': lambda prompt, max_tokens: _call_ollama(
        OLLAMA_LOCAL_URL, 'llama3.2', prompt, max_tokens, 60),
    'gemma2:2b': lambda prompt, max_tokens: _call_ollama(
        OLLAMA_LOCAL_URL, 'gemma2:2b', prompt, max_tokens, 60),
    'mistral': lambda prompt, max_tokens: _call_ollama(
        OLLAMA_LOCAL_URL, 'mistral', prompt, max_tokens, 60),
    'qwen2.5:0.5b': lambda prompt, max_tokens: _call_ollama(
        OLLAMA_LOCAL_URL, 'qwen2.5:0.5b', prompt, max_tokens, 60),
}

MODEL_LABELS = {
    'qwen': 'Qwen2.5:3b (production)',
    'gpt-4': 'GPT-4',
    'gpt-3.5-turbo': 'GPT-3.5-turbo',
    'gemini': f'Gemini ({GEMINI_MODEL or "gemini-1.5-flash"})',
    'llama3.2': 'Llama3.2 (local)',
    'gemma2:2b': 'Gemma2:2b (local)',
    'mistral': 'Mistral (local)',
    'qwen2.5:0.5b': 'Qwen2.5:0.5b (local, smaller)',
}

# ---------------------------------------------------------------------------
# TASK DEFINITIONS
# Each task reproduces the exact prompt template used in
# pipeline/translation_engine.py, with a hand-labeled gold set.
# Cases marked source='prompt_example' are literal examples quoted inside the
# production prompt itself (sanity floor -- any competent model should get
# these right). Cases marked source='held_out' are NOT mentioned in the
# prompt and test generalization of the classification rule.
# ---------------------------------------------------------------------------


def prompt_possession_type(lemma: str) -> str:
    return (
        f"Le nom français '{lemma}' est complément direct de 'avoir'.\n"
        f"Quelle catégorie lui correspond ?\n"
        f"AGE : durée ou âge (âge, ans, siècle, heure)\n"
        f"MATERIAL : objet physique concret qu'on peut tenir ou posséder "
        f"(voiture, maison, téléphone, argent, clé, vêtement, vélo, sac, outil, arme, couteau, bâton)\n"
        f"ABSTRACT : possession non physique — relation humaine, lien social, concept "
        f"ou DISPOSITION VOLITIONNELLE que le sujet peut mobiliser volontairement "
        f"(frère, ami, enfant, mari, famille, idée, droit, talent, chance, avis, "
        f"courage, confiance, patience, volonté, détermination, persévérance, orgueil)\n"
        f"EXPERIENCER : sensation physique externe subie par le corps — la sensation "
        f"est le sujet grammatical EN BAMBARA (faim, soif, chaud, froid, sommeil, "
        f"fièvre, nausée, douleur, mal, vertige, fatigue)\n"
        f"STATIF : état émotionnel PASSIF et INVOLONTAIRE — le sujet ne peut pas "
        f"le déclencher volontairement (test : 'Sois X!' est impossible ou absurde) "
        f"— s'exprime en bambara par une forme participiale "
        f"(peur, honte, joie, colère, jalousie, tristesse, regret, envie, nostalgie)\n"
        f"Réponds UNIQUEMENT par : AGE, MATERIAL, ABSTRACT, EXPERIENCER ou STATIF"
    )


def prompt_statif_adj(lemma: str) -> str:
    return (
        f'The French adjective "{lemma}" used predicatively — which Bambara construction?\n'
        f'QUALITE: permanent quality, physical property, or RELATIONAL property '
        f'that describes the subject. '
        f'Examples: grand, beau, fort, rapide, rouge, intelligent, '
        f'égal, semblable, différent, pareil, équivalent. '
        f'Bambara: subject + ka + adjective.\n'
        f'STATIF: temporary emotional or epistemic state the subject has entered. '
        f'Examples: fatigué, content, triste, prêt, malade, inquiet, libre, occupé, '
        f'sûr, certain, convaincu, conscient. '
        f'Bambara: adjective + -len/-nen dòn.\n'
        f'VALEUR: abstract truth-value presented as a FACT/NOUN (you would say '
        f'"c\'est la X"), NOT an adjective describing the subject. '
        f'Examples: vrai, faux, réel. '
        f'Bambara: noun dòn (presentative).\n'
        f'PARTICIPE: state resulting from a past action done to the subject. '
        f'Examples: blessé, fermé, cassé, ouvert, cuit. '
        f'Bambara: verb + -ra/-la/-na.\n'
        f'Reply with ONLY one word: QUALITE, STATIF, VALEUR, or PARTICIPE.'
    )


def prompt_classifying_adj(lemma: str) -> str:
    return (
        f'The French adjective "{lemma}" is used as an epithet (modifying a noun).\n'
        f'QUALIFIANT: describes a quality, property or characteristic of the noun — '
        f'grand, beau, vieux, rouge, rapide, chaud, intelligent, bon, mauvais, simple.\n'
        f'CLASSIFIANT: indicates a category, nationality, domain or type, NOT a quality — '
        f'français, international, européen, médical, électronique, national, politique.\n'
        f'Reply with ONLY one word: QUALIFIANT or CLASSIFIANT.'
    )


def prompt_reflexive_type(lemma: str) -> str:
    return (
        f'Verb: "{lemma}". Used with reflexive "se".\n'
        f'RECIPROCAL: Two or more participants perform the action on each other '
        f'(meet, fight, kiss, marry, see each other).\n'
        f'REFLEXIVE: The subject consciously and literally performs the action on themselves '
        f'as an object. This includes grooming/body-care '
        f'(wash, dress, shave, comb, hurt oneself, blame oneself, examine oneself).\n'
        f'PASSIVE: "se" has no semantic role; the subject undergoes the action '
        f'or the construction is impersonal/passive '
        f'(be sold, be called, be done, happen, be used).\n'
        f'SUBJECTIVE: "se" is an intrinsic part of the verb to express a state, change '
        f'of state, emotion, cognition, movement, or a completely non-literal meaning '
        f'(realize, remember, get angry, hurry, leave, wonder, concentrate, '
        f'make a mistake, get up, lie down, sit down, get bored).\n'
        f'Return one word only: RECIPROCAL, REFLEXIVE, PASSIVE, or SUBJECTIVE.'
    )


def prompt_relational_noun(lemma: str) -> str:
    return (
        f"En bambara, RÈGLE ABSOLUE : deux entités de même nature ne prennent JAMAIS 'ka'.\n"
        f"Les relations de parenté, de famille, et les relations sociales entre personnes "
        f"sont toujours INALIENABLES (sans 'ka') : père, mère, frère, sœur, fils, fille, "
        f"oncle, tante, cousin, grand-père, grand-mère, mari, femme, enfant, ami, ennemi, "
        f"voisin, collègue, patron, maître, chef, roi, dirigeant, responsable, leader, etc.\n"
        f"Les parties du corps sont aussi INALIENABLES (sans 'ka') : tête, bras, jambe, main, etc.\n"
        f"Seuls les objets physiques SÉPARABLES et TRANSFÉRABLES prennent 'ka' (possession ALIÉNABLE) : "
        f"maison, voiture, livre, vêtement, argent, champ, outil, etc.\n"
        f"Le mot français '{lemma}' représente-t-il une relation INALIENABLE (OUI) "
        f"ou un objet ALIÉNABLE (NON) ?\n"
        f"Réponds UNIQUEMENT par : OUI ou NON"
    )


def prompt_transitivity(lemma: str) -> str:
    return (
        f"Le verbe français '{lemma}' peut-il prendre un COD (complément d'objet direct) ?\n"
        f"ABSOLU  → jamais de COD (intransitif strict) : courir, dormir, régner, exister\n"
        f"ACTION  → COD possible (transitif) : manger, couper, aider, donner\n"
        f"Réponds UNIQUEMENT par ABSOLU ou ACTION."
    )


def prompt_privative_noun(lemma: str) -> str:
    return (
        f'Le nom français "{lemma}" est-il un NOM D\'ACTION '
        f'(dérivé d\'un verbe, représentant un processus ou une activité) '
        f'ou un NOM DE CHOSE (substance, objet, état) ?\n\n'
        f'NOM D\'ACTION : cuisson (de cuire), nettoyage (de nettoyer), '
        f'traitement (de traiter), construction, formation, réparation...\n'
        f'NOM DE CHOSE : sel, eau, sucre, argent, lumière, permission, bruit...\n\n'
        f'Réponds UNIQUEMENT par : ACTION ou CHOSE'
    )


def extract_first_label(raw: str, labels: list) -> str:
    if raw.startswith('__ERROR__'):
        return '__ERROR__'
    upper = raw.strip().upper()
    for lab in labels:
        if upper.startswith(lab):
            return lab
    for lab in labels:
        if lab in upper:
            return lab
    return upper.split()[0] if upper.split() else ''


TASKS = {
    'possession_type': {
        'prompt_fn': prompt_possession_type,
        'labels': ['AGE', 'MATERIAL', 'ABSTRACT', 'EXPERIENCER', 'STATIF'],
        'max_tokens': 5,
        'cases': [
            # -- prompt_example: literally listed as examples in the prompt
            ('âge', 'AGE', 'prompt_example'),
            ('siècle', 'AGE', 'prompt_example'),
            ('voiture', 'MATERIAL', 'prompt_example'),
            ('téléphone', 'MATERIAL', 'prompt_example'),
            ('frère', 'ABSTRACT', 'prompt_example'),
            ('courage', 'ABSTRACT', 'prompt_example'),
            ('faim', 'EXPERIENCER', 'prompt_example'),
            ('fièvre', 'EXPERIENCER', 'prompt_example'),
            ('peur', 'STATIF', 'prompt_example'),
            ('honte', 'STATIF', 'prompt_example'),
            # -- held_out: not in the prompt, tests generalization
            ('minute', 'AGE', 'held_out'),
            ('vélo', 'MATERIAL', 'held_out'),
            ('ordinateur', 'MATERIAL', 'held_out'),
            ('collègue', 'ABSTRACT', 'held_out'),
            ('espoir', 'ABSTRACT', 'held_out'),
            ('vertige', 'EXPERIENCER', 'held_out'),
            ('frisson', 'EXPERIENCER', 'held_out'),
            ('nostalgie', 'STATIF', 'held_out'),
            ('rancune', 'STATIF', 'held_out'),
        ],
    },
    'statif_adj': {
        'prompt_fn': prompt_statif_adj,
        'labels': ['QUALITE', 'STATIF', 'VALEUR', 'PARTICIPE'],
        'max_tokens': 5,
        'cases': [
            ('grand', 'QUALITE', 'prompt_example'),
            ('rouge', 'QUALITE', 'prompt_example'),
            ('fatigué', 'STATIF', 'prompt_example'),
            ('malade', 'STATIF', 'prompt_example'),
            ('vrai', 'VALEUR', 'prompt_example'),
            ('faux', 'VALEUR', 'prompt_example'),
            ('cassé', 'PARTICIPE', 'prompt_example'),
            ('ouvert', 'PARTICIPE', 'prompt_example'),
            ('intelligent', 'QUALITE', 'held_out'),
            ('rapide', 'QUALITE', 'held_out'),
            ('content', 'STATIF', 'held_out'),
            ('inquiet', 'STATIF', 'held_out'),
            ('réel', 'VALEUR', 'held_out'),
            ('fermé', 'PARTICIPE', 'held_out'),
            ('cuit', 'PARTICIPE', 'held_out'),
        ],
    },
    'classifying_adj': {
        'prompt_fn': prompt_classifying_adj,
        'labels': ['QUALIFIANT', 'CLASSIFIANT'],
        'max_tokens': 5,
        'cases': [
            ('grand', 'QUALIFIANT', 'prompt_example'),
            ('beau', 'QUALIFIANT', 'prompt_example'),
            ('chaud', 'QUALIFIANT', 'prompt_example'),
            ('français', 'CLASSIFIANT', 'prompt_example'),
            ('médical', 'CLASSIFIANT', 'prompt_example'),
            ('national', 'CLASSIFIANT', 'prompt_example'),
            ('intelligent', 'QUALIFIANT', 'held_out'),
            ('simple', 'QUALIFIANT', 'held_out'),
            ('rouge', 'QUALIFIANT', 'held_out'),
            ('africain', 'CLASSIFIANT', 'held_out'),
            ('juridique', 'CLASSIFIANT', 'held_out'),
            ('scolaire', 'CLASSIFIANT', 'held_out'),
        ],
    },
    'reflexive_type': {
        'prompt_fn': prompt_reflexive_type,
        'labels': ['RECIPROCAL', 'REFLEXIVE', 'PASSIVE', 'SUBJECTIVE'],
        'max_tokens': 5,
        'cases': [
            ('rencontrer', 'RECIPROCAL', 'prompt_example'),
            ('marier', 'RECIPROCAL', 'prompt_example'),
            ('laver', 'REFLEXIVE', 'prompt_example'),
            ('raser', 'REFLEXIVE', 'prompt_example'),
            ('vendre', 'PASSIVE', 'prompt_example'),
            ('appeler', 'PASSIVE', 'prompt_example'),
            ('souvenir', 'SUBJECTIVE', 'prompt_example'),
            ('dépêcher', 'SUBJECTIVE', 'prompt_example'),
            ('battre', 'RECIPROCAL', 'held_out'),
            ('embrasser', 'RECIPROCAL', 'held_out'),
            ('habiller', 'REFLEXIVE', 'held_out'),
            ('coiffer', 'REFLEXIVE', 'held_out'),
            ('faire', 'PASSIVE', 'held_out'),
            ('fâcher', 'SUBJECTIVE', 'held_out'),
            ('lever', 'SUBJECTIVE', 'held_out'),
        ],
    },
    'relational_noun': {
        'prompt_fn': prompt_relational_noun,
        'labels': ['OUI', 'NON'],
        'max_tokens': 5,
        'cases': [
            ('père', 'OUI', 'prompt_example'),
            ('frère', 'OUI', 'prompt_example'),
            ('tête', 'OUI', 'prompt_example'),
            ('maison', 'NON', 'prompt_example'),
            ('voiture', 'NON', 'prompt_example'),
            ('livre', 'NON', 'prompt_example'),
            ('cousin', 'OUI', 'held_out'),
            ('bras', 'OUI', 'held_out'),
            ('ami', 'OUI', 'held_out'),
            ('vélo', 'NON', 'held_out'),
            ('téléphone', 'NON', 'held_out'),
            ('outil', 'NON', 'held_out'),
        ],
    },
    'transitivity': {
        'prompt_fn': prompt_transitivity,
        'labels': ['ABSOLU', 'ACTION'],
        'max_tokens': 5,
        'cases': [
            ('courir', 'ABSOLU', 'prompt_example'),
            ('dormir', 'ABSOLU', 'prompt_example'),
            ('exister', 'ABSOLU', 'prompt_example'),
            ('manger', 'ACTION', 'prompt_example'),
            ('couper', 'ACTION', 'prompt_example'),
            ('donner', 'ACTION', 'prompt_example'),
            ('tomber', 'ABSOLU', 'held_out'),
            ('naître', 'ABSOLU', 'held_out'),
            ('construire', 'ACTION', 'held_out'),
            ('réparer', 'ACTION', 'held_out'),
        ],
    },
    'privative_noun': {
        'prompt_fn': prompt_privative_noun,
        'labels': ['ACTION', 'CHOSE'],
        'max_tokens': 4,
        'cases': [
            ('cuisson', 'ACTION', 'prompt_example'),
            ('nettoyage', 'ACTION', 'prompt_example'),
            ('construction', 'ACTION', 'prompt_example'),
            ('sel', 'CHOSE', 'prompt_example'),
            ('argent', 'CHOSE', 'prompt_example'),
            ('bruit', 'CHOSE', 'prompt_example'),
            ('réparation', 'ACTION', 'held_out'),
            ('formation', 'ACTION', 'held_out'),
            ('eau', 'CHOSE', 'held_out'),
            ('lumière', 'CHOSE', 'held_out'),
        ],
    },
}

# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------


def run(task_names, model_names, out_dir):
    rows = []
    for task_name in task_names:
        task = TASKS[task_name]
        for lemma, gold, source in task['cases']:
            prompt = task['prompt_fn'](lemma)
            for model_name in model_names:
                backend = MODEL_BACKENDS[model_name]
                t0 = time.time()
                raw = backend(prompt, task['max_tokens'])
                dt = time.time() - t0
                pred = extract_first_label(raw, task['labels'])
                correct = (pred == gold)
                rows.append({
                    'task': task_name, 'lemma': lemma, 'gold': gold,
                    'source': source, 'model': model_name,
                    'raw_response': raw, 'predicted': pred,
                    'correct': correct, 'latency_s': round(dt, 2),
                })
                mark = '' if correct else ('' if pred == '__ERROR__' else '')
                print(f"[{task_name:18s}] {model_name:14s} {lemma:14s} "
                      f"gold={gold:12s} pred={pred:12s} {mark} ({dt:.1f}s)")

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'model_comparison_{ts}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    json_path = os.path.join(out_dir, f'model_comparison_{ts}_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_report(summary)
    print(f"\nRaw results: {csv_path}")
    print(f"Summary:     {json_path}")
    return rows, summary


def summarize(rows):
    per_model_task = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0})
    per_model = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0})
    for r in rows:
        key = (r['model'], r['task'])
        per_model_task[key]['total'] += 1
        per_model[r['model']]['total'] += 1
        if r['predicted'] == '__ERROR__':
            per_model_task[key]['errors'] += 1
            per_model[r['model']]['errors'] += 1
        elif r['correct']:
            per_model_task[key]['correct'] += 1
            per_model[r['model']]['correct'] += 1

    out = {'per_model_task': {}, 'per_model_overall': {}}
    for (model, task), d in per_model_task.items():
        out['per_model_task'].setdefault(task, {})[model] = {
            'accuracy': round(d['correct'] / d['total'], 3) if d['total'] else 0,
            'correct': d['correct'], 'total': d['total'], 'errors': d['errors'],
        }
    for model, d in per_model.items():
        out['per_model_overall'][model] = {
            'accuracy': round(d['correct'] / d['total'], 3) if d['total'] else 0,
            'correct': d['correct'], 'total': d['total'], 'errors': d['errors'],
        }
    return out


def print_report(summary):
    print("\n" + "=" * 70)
    print("PER-TASK ACCURACY")
    print("=" * 70)
    for task, models in summary['per_model_task'].items():
        print(f"\n{task}:")
        for model, d in models.items():
            print(f"  {model:16s} {d['accuracy']*100:5.1f}%  "
                  f"({d['correct']}/{d['total']}, {d['errors']} errors)")

    print("\n" + "=" * 70)
    print("OVERALL ACCURACY")
    print("=" * 70)
    for model, d in summary['per_model_overall'].items():
        print(f"  {model:16s} {d['accuracy']*100:5.1f}%  "
              f"({d['correct']}/{d['total']}, {d['errors']} errors)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='all',
                     help='Comma-separated task names, or "all"')
    ap.add_argument('--models', default='qwen,gemini,llama3.2,gemma2:2b,mistral,qwen2.5:0.5b',
                     help='Comma-separated model names')
    args = ap.parse_args()

    task_names = list(TASKS.keys()) if args.task == 'all' else args.task.split(',')
    model_names = args.models.split(',')
    for m in model_names:
        if m not in MODEL_BACKENDS:
            raise SystemExit(f"Unknown model '{m}'. Choices: {list(MODEL_BACKENDS)}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    run(task_names, model_names, out_dir)
