"""
eval/bambara_ud_reference.py — Arbre UD de RÉFÉRENCE pour la sortie Kuma-MT
================================================================================
Construit l'arbre UD "vrai" (POS + dependency relations) pour une phrase
bambara PRODUITE PAR KUMA-MT, de façon déterministe à partir de :
  - tree_meta (clause_type, tam, S/V/O/obl_all) — connu avec certitude, c'est
    Kuma lui-même qui a rempli ces slots pendant la génération ;
  - le schéma d'annotation documenté dans Aplonova & Tyers 2017/2018,
    "Towards a dependency-annotated treebank for Bambara"
    (https://aclanthology.org/W17-7618.pdf), confirmé par des exemples réels
    du treebank UD_Bambara (github.com/KatyaAplonova/UD_Bambara,
    conllu_checked/*.conllu — ex: 'ye'/'don' taggés VERB+root dans 21+
    occurrences réelles de constructions copulatives).

Pourquoi PAS le parseur UDPipe tiers (eval/bambara_udpipe.py) : vérifié sur
"n yé kàramɔgɔkɛ yé" (sortie Kuma réelle, équatif) — le modèle inverse
complètement les rôles : copule 'yé' taggée NOUN+obj, nom 'kàramɔgɔkɛ' taggé
VERB+root. La construction bare-copule ("N yé O yé") est probablement rare
dans le corpus d'entraînement CRB (contes/entretiens), donc le modèle n'a pas
de signal fiable dessus — inutilisable comme vérité terrain pour CETTE
construction, même s'il est correctement évalué (bambara_parser_eval.py)
contre SON PROPRE test set CRB.

Schéma par clause_type (§ = section du papier Aplonova & Tyers) :
  equative / identificatory  (§5.2, ex 3c) :
      S  = NOUN/PRON + nsubj  →  copule(yé/tɛ) = VERB + root
      →  O = NOUN + obl  →  2e yé = ADP + case
  locative                   (§5.2, ex 3b) :
      S = NOUN/PRON + nsubj  →  copule(bɛ/tɛ) = VERB + root
      →  OBL.head = NOUN + obl  →  OBL.marker = ADP + case
  existential_absolute / presentative (§5.2, ex 3a) :
      référent = NOUN + nsubj  →  copule(dòn/bɛ) = VERB + root
  simple (verbe dynamique)   (Table 2 + structure S-TAM-O-V) :
      S = NOUN/PRON + nsubj  →  TAM = AUX + aux  →  O = NOUN + obj
      →  V = VERB + root
  statif (verbe qualitatif)  (§5.1, ex 2a) :
      S = NOUN/PRON + nsubj  →  V+len = VERB + root (qualitatif)
      →  copule(dòn/tɛ) = AUX + aux  [marqueur predicatif qual, cf §5.1]

Limitation assumée : les clause_type hors de cette liste retournent None
(non couverts) — pas de génération hasardeuse pour des constructions non
vérifiées contre le papier/corpus.
"""

# Pronoms personnels bambara (Table 2 : Personal pronoun -> PRON) — ensemble
# fermé, permet de distinguer PRON/NOUN pour le slot S sans heuristique
# hasardeuse (contrairement à un test comme S.isupper(), toujours faux ici
# puisque le bambara n'a pas de majuscules grammaticales pour les pronoms).
_PERSONAL_PRONOUNS = {'n', 'ń', 'i', 'a', 'à', 'an', 'aw', 'u', 'ù', 'ne', 'né'}


def _subj_upos(text):
    return 'PRON' if text.strip().lower() in _PERSONAL_PRONOUNS else 'NOUN'


def build_reference_ud_tree(tree_meta: dict, output: str) -> list:
    """Retourne :
      [{'id','form','upos','deprel','head'}, ...]  si succès
      None  si le clause_type n'est pas couvert par ce module
      []    si le clause_type EST couvert mais la sortie ne contient pas
            les mots attendus dans le bon ordre (échec réel de structure)
    """
    if not output or not tree_meta:
        return None

    clause_type = (tree_meta.get('clause_type') or '').strip()
    tam = (tree_meta.get('tam') or '').strip()
    S = (tree_meta.get('S') or '').strip()
    V = (tree_meta.get('V') or '').strip()
    O = (tree_meta.get('O') or '').strip()
    obl_all = tree_meta.get('obl_all') or []

    # nodes ajoutés dans l'ORDRE DE SURFACE (gauche à droite) — indispensable
    # pour la résolution séquentielle des positions ci-dessous (sinon deux
    # occurrences identiques, ex: 'yé' du copule répété, s'alignent mal).
    # head_ref référence une clé libre (pas forcément déjà ajoutée) résolue
    # à la fin via key_to_id, ce qui permet à S (en tête de phrase) de
    # pointer vers le copule (root) qui n'est ajouté qu'ensuite.
    nodes = []   # [{'key','form','upos','deprel','head_key'}]

    def add(key, form, upos, deprel, head_key=None):
        if not form:
            return
        nodes.append({'key': key, 'form': form, 'upos': upos,
                      'deprel': deprel, 'head_key': head_key})

    if clause_type in ('equative', 'identificatory'):
        # §5.2 ex 3c : S nsubj -> copule(root,VERB) -> O obl -> 2e marqueur ADP case
        # Le marqueur de clôture est TOUJOURS littéralement 'yé', même à la
        # forme négative (KG template equative_neg = '{S} tɛ {O} yé', pas
        # '{S} tɛ {O} tɛ') — confirmé par échec sur "a tɛ porofesɛri yé" quand
        # ce nœud cherchait à tort un 2e `tam` (donc un 2e 'tɛ' inexistant).
        add('S', S, _subj_upos(S), 'nsubj', 'ROOT')
        add('ROOT', tam, 'VERB', 'root', None)
        add('O', O, 'NOUN', 'obl', 'ROOT')
        add('CASE', 'yé', 'ADP', 'case', 'O')

    elif clause_type == 'locative':
        # §5.2 ex 3b : S nsubj -> copule(root,VERB) -> OBL.head obl -> OBL.marker ADP case
        add('S', S, _subj_upos(S), 'nsubj', 'ROOT')
        add('ROOT', tam, 'VERB', 'root', None)
        if obl_all:
            add('OBL', obl_all[0]['head'], 'NOUN', 'obl', 'ROOT')
            add('CASE', obl_all[0]['marker'], 'ADP', 'case', 'OBL')

    elif clause_type in ('existential_absolute', 'existential_localized', 'presentative'):
        # §5.2 ex 3a : référent nsubj -> copule(root,VERB)
        referent = O or S
        add('REF', referent, 'NOUN', 'nsubj', 'ROOT')
        add('ROOT', tam, 'VERB', 'root', None)
        if clause_type == 'existential_localized' and obl_all:
            add('OBL', obl_all[0]['head'], 'NOUN', 'obl', 'ROOT')
            add('CASE', obl_all[0]['marker'], 'ADP', 'case', 'OBL')

    elif clause_type == 'simple':
        # S nsubj -> TAM aux -> O obj -> V root (structure S-TAM-O-V)
        add('S', S, _subj_upos(S), 'nsubj', 'ROOT')
        add('TAM', tam, 'AUX', 'aux', 'ROOT')
        add('O', O, 'NOUN', 'obj', 'ROOT')
        add('ROOT', V, 'VERB', 'root', None)

    elif clause_type == 'statif':
        # §5.1 : S nsubj -> V(qualitatif, +len) root -> copule aux
        add('S', S, _subj_upos(S), 'nsubj', 'ROOT')
        add('ROOT', V, 'VERB', 'root', None)
        add('TAM', tam, 'AUX', 'aux', 'ROOT')

    else:
        return None   # clause_type non couvert — pas de génération hasardeuse

    if not nodes:
        return None

    # Résoudre les positions dans la chaîne de sortie, EN ORDRE DE SURFACE
    # (chaque forme cherchée à partir de la fin du match précédent, pour
    # gérer les répétitions comme le copule 'yé' apparaissant deux fois).
    # IMPORTANT : un échec de résolution ici retourne [] (liste vide), PAS
    # None — la distinction compte pour les consommateurs scorés (cf.
    # eval/evaluate.py::ud_structure_ok) : None = clause_type non couvert
    # (exclu de la moyenne), [] = clause_type couvert MAIS la sortie ne
    # contient pas les mots attendus dans le bon ordre (échec réel, doit
    # compter comme faux, pas être silencieusement ignoré).
    resolved = []
    last_pos = -1
    for n in nodes:
        pos = output.find(n['form'], last_pos + 1)
        if pos == -1:
            return []
        last_pos = pos
        resolved.append({**n, '_pos': pos})

    key_to_id = {n['key']: i + 1 for i, n in enumerate(resolved)}
    result = []
    for i, n in enumerate(resolved):
        head_key = n['head_key']
        head_id = 0 if head_key is None else key_to_id.get(head_key, 0)
        result.append({
            'id': i + 1, 'form': n['form'], 'upos': n['upos'],
            'deprel': n['deprel'], 'head': head_id,
        })
    return result
