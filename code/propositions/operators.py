# (c) This file is part of the course
# Mathematical Logic through Programming
# by Gonczarowski and Nisan.
# File name: propositions/operators.py

"""Syntactic conversion of propositional formulae to use only specific sets of
operators."""

from propositions.syntax import *
from propositions.semantics import *

def to_not_and_or(formula: Formula) -> Formula:
    """Syntactically converts the given formula to an equivalent formula that
    contains no constants or operators beyond ``'~'``, ``'&'``, and ``'|'``.

    Parameters:
        formula: formula to convert.

    Return:
        A formula that has the same truth table as the given formula, but
        contains no constants or operators beyond ``'~'``, ``'&'``, and
        ``'|'``.
    """
    substitution_map = {'-|': Formula.parse('~(p|q)'), '-&': Formula.parse('~(p&q)'),
                        'T': Formula.parse('(p|~p)'), 'F': Formula.parse('(p&~p)'),
                        '+': Formula.parse('((p|q)&~(p&q))'), '->': Formula.parse('(~p|(p&q))'),
                        '<->': Formula.parse('((p&q)|(~p&~q))')}
    new_formula = formula
    if is_unary(formula.root):
        new_formula = Formula(formula.root, to_not_and_or(formula.first))
    if is_binary(formula.root):
        new_formula = Formula(formula.root, to_not_and_or(formula.first), to_not_and_or(formula.second))
    return new_formula.substitute_operators(substitution_map)

def to_not_and(formula: Formula) -> Formula:
    """Syntactically converts the given formula to an equivalent formula that
    contains no constants or operators beyond ``'~'`` and ``'&'``.

    Parameters:
        formula: formula to convert.

    Return:
        A formula that has the same truth table as the given formula, but
        contains no constants or operators beyond ``'~'`` and ``'&'``.
    """
    substitution_map = {'|': Formula.parse('~(~p&~q)'), '&': Formula.parse('(p&q)'),
                        '~': Formula.parse('~p')}
    new_formula = to_not_and_or(formula)
    return new_formula.substitute_operators(substitution_map)

def to_nand(formula: Formula) -> Formula:
    """Syntactically converts the given formula to an equivalent formula that
    contains no constants or operators beyond ``'-&'``.

    Parameters:
        formula: formula to convert.

    Return:
        A formula that has the same truth table as the given formula, but
        contains no constants or operators beyond ``'-&'``.
    """
    substitution_map = {'&': Formula.parse('((p-&q)-&(p-&q))'),
                        '~': Formula.parse('(p-&p)')}
    new_formula = to_not_and(formula)
    return new_formula.substitute_operators(substitution_map)

def to_implies_not(formula: Formula) -> Formula:
    """Syntactically converts the given formula to an equivalent formula that
    contains no constants or operators beyond ``'->'`` and ``'~'``.

    Parameters:
        formula: formula to convert.

    Return:
        A formula that has the same truth table as the given formula, but
        contains no constants or operators beyond ``'->'`` and ``'~'``.
    """
    substitution_map = {'-&': Formula.parse('(p->~q)')}
    new_formula = to_nand(formula)
    return new_formula.substitute_operators(substitution_map)

def to_implies_false(formula: Formula) -> Formula:
    """Syntactically converts the given formula to an equivalent formula that
    contains no constants or operators beyond ``'->'`` and ``'F'``.

    Parameters:
        formula: formula to convert.

    Return:
        A formula that has the same truth table as the given formula, but
        contains no constants or operators beyond ``'->'`` and ``'F'``.
    """
    substitution_map = {'~': Formula.parse('(p->F)'), '->': Formula.parse('(p->q)')}
    new_formula = to_implies_not(formula)
    return new_formula.substitute_operators(substitution_map)
