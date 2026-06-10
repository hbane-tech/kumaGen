"""rules/steps/step4_objet/__init__.py — orchestrateur étape objet."""
from rules.steps.step4_objet import noun_phrase
from rules.steps.step4_objet import objet_standard
from rules.steps.step4_objet import relcl_post
from rules.steps.step4_objet import ownership


def run(T, tree, m, processed_indices, G_kg, NX_G,
        root_tok, root_noun, has_acl, _has_relcl,
        xcomp_adj_tok, clause_type_init, _has_question_mark, _is_copula_fn):

    noun_phrase.run(T, tree, m, processed_indices, G_kg,
                    root_noun, has_acl, _has_relcl)

    objet_standard.run(T, tree, m, processed_indices, G_kg, NX_G,
                       root_tok, root_noun, has_acl,
                       xcomp_adj_tok, clause_type_init, _has_question_mark)

    relcl_post.run(T, tree, m, processed_indices, G_kg, NX_G)

    ownership.run(T, tree, m, processed_indices, G_kg, root_tok)
