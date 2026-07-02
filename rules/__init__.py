"""
rules/__init__.py
Point d'entrée public du package rules.

Import compatible avec l'ancien :
    from rules.r1_r60_engine import RuleEngine, build_tree, tree_to_bambara
Nouveau :
    from rules import RuleEngine, build_tree, tree_to_bambara
"""
from rules.core import (j, _resolve_tam, _is_copula, _is_avoir,
                        _GRAMMAR_FALLBACK, _GENITIVE_FALLBACK)
from rules.build_tree import build_tree
# from rules.tree_to_bambara import tree_to_bambara
from rules.renderers import tree_to_bambara
from rules.rule_engine import RuleEngine

__all__ = [
    'j', '_resolve_tam', '_is_copula', '_is_avoir',
    '_GRAMMAR_FALLBACK', '_GENITIVE_FALLBACK',
    'build_tree', 'tree_to_bambara', 'RuleEngine',
]
