"""rules/steps/step4_objet/__init__.py — orchestrateur étape objet."""
from rules.steps.step4_objet import noun_phrase
from rules.steps.step4_objet import objet_standard
from rules.steps.step4_objet import relcl_post
from rules.steps.step4_objet import ownership


def run(T, tree, m, processed_indices, G_kg, NX_G,
        root_tok, root_noun, has_acl, _has_relcl,
        xcomp_adj_tok, clause_type_init, _has_question_mark, _is_copula_fn):

    _pi0 = set(processed_indices)
    noun_phrase.run(T, tree, m, processed_indices, G_kg,
                    root_noun, has_acl, _has_relcl)
    print(f'[S4-NP] added={sorted(processed_indices-_pi0)}')

    _pi1 = set(processed_indices)
    objet_standard.run(T, tree, m, processed_indices, G_kg, NX_G,
                       root_tok, root_noun, has_acl,
                       xcomp_adj_tok, clause_type_init, _has_question_mark)
    print(f'[S4-OBJ] added={sorted(processed_indices-_pi1)} m_O={m.get("O")}')

    _pi2 = set(processed_indices)
    relcl_post.run(T, tree, m, processed_indices, G_kg, NX_G)
    print(f'[S4-RELCL] added={sorted(processed_indices-_pi2)}')

    _pi3 = set(processed_indices)
    ownership.run(T, tree, m, processed_indices, G_kg, root_tok)
    print(f'[S4-OWN] added={sorted(processed_indices-_pi3)}')
