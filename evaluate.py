"""
evaluate.py
Module d'évaluation quantitative automatisé pour Kuma-MT.
Calcule les scores BLEU et chrF sur le jeu de données de test.
"""

import sys
import os
import sacrebleu
from pipeline.translation_engine import TranslationEngine

# Configuration du framework de test
TEST_SUITE = [
    {
        "id": "TC-001",
        "category": "Complex Multi-Chunk",
        "source": "Thierry Vircoulon revient pour « 20 Minutes » sur la situation explosive au Mali après les attaques djihadistes de ce week-end",
        "reference": "thierry vircoulon bɛ kɔ́sègin «míniti mugan» ye sɛ́rɛya kun kan mali la ce dɔ́gɔfurancɛ laban in ka bìnni djihadistew kɔfɛ"
    },
    {
        "id": "TC-002",
        "category": "Pure Nominal Cascade",
        "source": "La décision finale de la constitution du Mali",
        "reference": "mali ka sariyaba ka kili laban"
    },
    {
        "id": "TC-003",
        "category": "Transitive Standard",
        "source": "Je mange du riz",
        "reference": "n bɛ iri dún"
    },
    {
        "id": "TC-004",
        "category": "Future / Prospective",
        "source": "Je vais manger du riz demain",
        "reference": "n bɛ na iri dún síni"
    },
    {
        "id": "TC-005",
        "category": "Distative Coordination & Purposive",
        "source": "L’université de Kumamoto recrute actuellement des participants pour le « Programme d’apprentissage coopératif japonais », ainsi que des étudiants japonais pour la soutenir.",
        "reference": "kumamoto ka université bɛ participantsw recruter programme d'apprentissage coopératif japonais ye ni étudiants japonais kà la soutenir"
    }
]

def run_evaluation():
    print("=" * 90)
    print("                       KUMA-MT AUTOMATED EVALUATION ENGINE                     ")
    print("=" * 90)
    
    # Initialisation sécurisée de ton moteur neuro-symbolique
    try:
        engine = TranslationEngine()
    except Exception as e:
        print(f" Erreur lors de l'initialisation de TranslationEngine : {e}")
        sys.exit(1)
        
    hypotheses = []
    references = []
    
    print(f"\n Exécution des traductions sur {len(TEST_SUITE)} cas d'évaluation...")
    
    for case in TEST_SUITE:
        print(f"\n  [{case['id']}] Catégorie : {case['category']}")
        print(f"     SRC -> {case['source']}")
        
        # Traduction à chaud via ton pipeline unifié
        try:
            pred = engine.translate(case['source'])
        except Exception as e:
            print(f"     Crash lors de la traduction : {e}")
            pred = ""
            
        pred_clean = pred.strip()
        ref_clean = case['reference'].strip()
        
        print(f"     REF -> {ref_clean}")
        print(f"     HYP -> {pred_clean if pred_clean else '[Chaîne vide / Échec]'}")
        
        # Accumulation pour le calcul global
        hypotheses.append(pred_clean)
        # SacreBLEU attend une liste de listes pour les références (multi-références possibles)
        references.append([ref_clean])
        
        # Calcul local indicatif (Sentence-level BLEU / chrF)
        if pred_clean:
            local_bleu = sacrebleu.sentence_bleu(pred_clean, [ref_clean]).score
            local_chrf = sacrebleu.sentence_chrf(pred_clean, [ref_clean]).score
            print(f"     Scores locaux : BLEU = {local_bleu:.2f} | chrF = {local_chrf:.2f}")

    print("\n" + "=" * 90)
    print("                                  RAPPORT FINAL COMPILÉ                                ")
    print("=" * 90)
    
    # Calcul des métriques globales au niveau du corpus (Corpus-level)
    # Les références doivent être transposées pour SacreBLEU : une liste par version de référence
    refs_transposed = [list(x) for x in zip(*references)]
    
    try:
        corpus_bleu = sacrebleu.corpus_bleu(hypotheses, refs_transposed)
        corpus_chrf = sacrebleu.corpus_chrf(hypotheses, refs_transposed)
        
        print(f"\n  SCORE CORPUS BLEU : {corpus_bleu.score:.2f}")
        print(f"     (Détails n-grammes : {corpus_bleu.bp_entropy_summary() if hasattr(corpus_bleu, 'bp_entropy_summary') else corpus_bleu.prec_str})")
        print(f"  SCORE CORPUS chrF : {corpus_chrf.score:.2f}")
        print("\n" + "-" * 90)
        # print(" Tu peux insérer ces scores directement dans la Section 'Experiments & Results' de ton papier.")
    except Exception as e:
        print(f" Échec du calcul des métriques globales : {e}")
        
    print("=" * 90)

if __name__ == "__main__":
    run_evaluation()
