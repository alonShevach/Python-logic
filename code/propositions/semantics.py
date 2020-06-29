# (c) This file is part of the course
# Mathematical Logic through Programming
# by Gonczarowski and Nisan.
# File name: propositions/semantics.py

"""Semantic analysis of propositional-logic constructs."""

from typing import AbstractSet, Iterable, Iterator, List, Mapping
from itertools import product

from propositions.syntax import *
from propositions.proofs import *

Model = Mapping[str, bool]

def is_model(model: Model) -> bool:
    """Checks if the given dictionary a model over some set of variables.

    Parameters:
        model: dictionary to check.

    Returns:
        ``True`` if the given dictionary is a model over some set of variables,
        ``False`` otherwise.
    """
    for key in model:
        if not (is_variable(key) and type(model[key]) is bool):
            return False
    return True

def variables(model: Model) -> AbstractSet[str]:
    """Finds all variables over which the given model is defined.

    Parameters:
        model: model to check.

    Returns:
        A set of all variables over which the given model is defined.
    """
    assert is_model(model)
    return model.keys()

def evaluate(formula: Formula, model: Model) -> bool:
    """Calculates the truth value of the given formula in the given model.

    Parameters:
        formula: formula to calculate the truth value of.
        model: model over (possibly a superset of) the variables of the formula,
            to calculate the truth value in.

    Returns:
        The truth value of the given formula in the given model.
    """
    assert is_model(model)
    assert formula.variables().issubset(variables(model))
    if formula.root in model.keys():
        return model.get(formula.root)
    elif is_constant(formula.root):
        if formula.root == 'T':
            return True
        else:
            return False
    else:
        if formula.root == '~':
            if evaluate(formula.first, model) == True:
                return False
            else:
                return True
        elif formula.root == '&':
            return (evaluate(formula.first, model) and evaluate(formula.second, model))
        elif formula.root == '|':
            return (evaluate(formula.first, model) or evaluate(formula.second, model))
        elif formula.root == '+':
            return (evaluate(formula.first, model) != evaluate(formula.second, model))
        elif formula.root == '<->':
            return (evaluate(formula.first, model) and evaluate(formula.second, model)) or (not evaluate(formula.first, model) and not evaluate(formula.second, model))
        elif formula.root == '-&':
            return not(evaluate(formula.first, model) and evaluate(formula.second, model))
        elif formula.root == '-|':
            return not (evaluate(formula.first, model) or evaluate(formula.second, model))
        else:
            return (evaluate(formula.first, model) == False or evaluate(formula.second, model) == True)

def all_models(variables: List[str]) -> Iterable[Model]:
    """Calculates all possible models over the given variables.

    Parameters:
        variables: list of variables over which to calculate the models.

    Returns:
        An iterable over all possible models over the given variables. The order
        of the models is lexicographic according to the order of the given
        variables, where False precedes True.

    Examples:
        >>> list(all_models(['p', 'q']))
        [{'p': False, 'q': False}, {'p': False, 'q': True}, {'p': True, 'q': False}, {'p': True, 'q': True}]
    """
    for v in variables:
        assert is_variable(v)
    n = len(variables)
    for p in product([False, True], repeat=n):
        Model = {}
        i = 0
        for var in variables:
            Model[var] = p[i]
            i += 1
        yield Model

def truth_values(formula: Formula, models: Iterable[Model]) -> Iterable[bool]:
    """Calculates the truth value of the given formula in each of the given
    model.

    Parameters:
        formula: formula to calculate the truth value of.
        model: iterable over models to calculate the truth value in.

    Returns:
        An iterable over the respective truth values of the given formula in
        each of the given models, in the order of the given models.
    """
    for model in models:
        yield evaluate(formula, model)

def print_truth_table(formula: Formula) -> None:
    """Prints the truth table of the given formula, with variable-name columns
    sorted alphabetically.

    Parameters:
        formula: formula to print the truth table of.

    Examples:
        >>> print_truth_table(Formula.parse('~(p&q76)'))
        | p | q76 | ~(p&q76) |
        |---|-----|----------|
        | F | F   | T        |
        | F | T   | T        |
        | T | F   | T        |
        | T | T   | F        |
    """
    all_vars_sorted = sorted(formula.variables())
    formula_str = formula.__repr__()
    for var in all_vars_sorted:
        print('|', var, '', end='')
    print('|', formula_str, '|')
    for var in all_vars_sorted:
        print('|' + (len(var) + 2)*'-', end='')
    print('|' + (len(formula_str) + 2)*'-' + '|')
    for model in all_models(all_vars_sorted):
        for var in all_vars_sorted:
            if model[var] == True:
                print('| T', (len(var)-1)*' ', end = '')
            else:
                print('| F', (len(var)-1)*' ', end = '')
        if evaluate(formula, model) == True:
            print('| T', (len(formula_str) - 1) * ' ', end='|\n')
        else:
            print('| F', (len(formula_str) - 1) * ' ', end='|\n')


def is_tautology(formula: Formula) -> bool:
    """Checks if the given formula is a tautology.

    Parameters:
        formula: formula to check.

    Returns:
        ``True`` if the given formula is a tautology, ``False`` otherwise.
    """
    for val in truth_values(formula, all_models(list(formula.variables()))):
        if val != True:
            return False
    return True

def is_contradiction(formula: Formula) -> bool:
    """Checks if the given formula is a contradiction.

    Parameters:
        formula: formula to check.

    Returns:
        ``True`` if the given formula is a contradiction, ``False`` otherwise.
    """
    new_formula = Formula('~',formula)
    if is_tautology(new_formula):
        return True
    else:
        return False

def is_satisfiable(formula: Formula) -> bool:
    """Checks if the given formula is satisfiable.

    Parameters:
        formula: formula to check.

    Returns:
        ``True`` if the given formula is satisfiable, ``False`` otherwise.
    """
    new_formula = Formula('~', formula)
    if not is_tautology(new_formula):
        return True
    else:
        return False

def synthesize_for_model(model: Model) -> Formula:
    """Synthesizes a propositional formula in the form of a single clause that
      evaluates to ``True`` in the given model, and to ``False`` in any other
      model over the same variables.

    Parameters:
        model: model in which the synthesized formula is to hold.

    Returns:
        The synthesized formula.
    """
    assert is_model(model)
    formula = None
    for var in model.keys():
        if model[var] == True:
            if formula == None:
                formula = Formula(var)
            else:
                var_formula = Formula(var)
                new_formula = Formula('&', formula, var_formula)
                formula = new_formula
        else:
            if formula == None:
                var_formula = Formula(var)
                formula = Formula('~', var_formula)
            else:
                var_formula = Formula(var)
                neg_var = Formula('~', var_formula)
                new_formula = Formula('&', formula, neg_var)
                formula = new_formula
    return formula

def synthesize(variables: List[str], values: Iterable[bool]) -> Formula:
    """Synthesizes a propositional formula in DNF over the given variables, from
    the given specification of which value the formula should have on each
    possible model over these variables.

    Parameters:
        variables: the set of variables for the synthesize formula.
        values: iterable over truth values for the synthesized formula in every
            possible model over the given variables, in the order returned by
            `all_models`\ ``(``\ `~synthesize.variables`\ ``)``.

    Returns:
        The synthesized formula.

    Examples:
        >>> formula = synthesize(['p', 'q'], [True, True, True, False])
        >>> for model in all_models(['p', 'q']):
        ...     evaluate(formula, model)
        True
        True
        True
        False
    """
    assert len(variables) > 0
    i = 0
    formula = None
    for model in all_models(variables):
        if values[i] == True:
            sub_formula = synthesize_for_model(model)
            if formula == None:
                formula = sub_formula
            else:
                new_formula = Formula('|', sub_formula, formula)
                formula = new_formula
        i += 1
    if formula == None:
        var_formula = Formula(variables[0])
        return Formula('&', var_formula, Formula('~', var_formula))
    return formula

# Tasks for Chapter 4

def evaluate_inference(rule: InferenceRule, model: Model) -> bool:
    """Checks if the given inference rule holds in the given model.

    Parameters:
        rule: inference rule to check.
        model: model to check in.

    Returns:
        ``True`` if the given inference rule holds in the given model, ``False``
        otherwise.
    """
    assert is_model(model)
    if len(rule.assumptions) == 0:
        return is_tautology(rule.conclusion)
    assumptions = True #holds whether all assumptions hold in this model
    for formula in rule.assumptions:
        if evaluate(formula, model) == False:
            assumptions = False
    if assumptions:
        if evaluate(rule.conclusion, model) == False:
            return False
    return True


def is_sound_inference(rule: InferenceRule) -> bool:
    """Checks if the given inference rule is sound, i.e., whether its
    conclusion is a semantically correct implication of its assumptions.

    Parameters:
        rule: inference rule to check.

    Returns:
        ``True`` if the given inference rule is sound, ``False`` otherwise.
    """
    for model in all_models(list(rule.variables())):
        if evaluate_inference(rule, model) == False:
            return False
    return True