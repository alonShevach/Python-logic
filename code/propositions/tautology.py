# (c) This file is part of the course
# Mathematical Logic through Programming
# by Gonczarowski and Nisan.
# File name: propositions/tautology.py

"""The Tautology Theorem and its implications."""

from typing import List, Union

from logic_utils import frozendict

from propositions.syntax import *
from propositions.proofs import *
from propositions.deduction import *
from propositions.semantics import *
from propositions.operators import *
from propositions.axiomatic_systems import *

def formulae_capturing_model(model: Model) -> List[Formula]:
    """Computes the formulae that capture the given model: ``'``\ `x`\ ``'``
    for each variable `x` that is assigned the value ``True`` in the given
    model, and ``'~``\ `x`\ ``'`` for each variable x that is assigned the value
    ``False``.

    Parameters:
        model: model to construct the formulae for.

    Returns:
        A list of the constructed formulae, ordered alphabetically by variable
        name.

    Examples:
        >>> formulae_capturing_model({'p2': False, 'p1': True, 'q': True})
        [p1, ~p2, q]
    """
    assert is_model(model)
    all_formulas = []
    for val in sorted(model):
        if model[val]:
            all_formulas.append(Formula(val))
        else:
            all_formulas.append(Formula('~', Formula(val)))
    return all_formulas

def prove_in_model(formula: Formula, model:Model) -> Proof:
    """Either proves the given formula or proves its negation, from the formulae
    that capture the given model.

    Parameters:
        formula: formula that contains no constants or operators beyond ``'->'``
            and ``'~'``, whose affirmation or negation is to prove.
        model: model from whose formulae to prove.

    Returns:
        If the given formula evaluates to ``True`` in the given model, then
        a proof of the formula, otherwise a proof of ``'~``\ `formula`\ ``'``.
        The returned proof is from the formulae that capture the given model, in
        the order returned by `formulae_capturing_model`\ ``(``\ `model`\ ``)``,
        via `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM`.
    """
    assert formula.operators().issubset({'->', '~'})
    assert is_model(model)
    formula_evaluation = evaluate(formula, model)
    capturing_model = formulae_capturing_model(model)
    inference_rule_to_prove = InferenceRule(capturing_model, formula)
    # if formula evaluates to False we will proof the negation of formula
    if not formula_evaluation:
        inference_rule_to_prove = InferenceRule(capturing_model, Formula('~', formula))
    if formula.root in model:
        if model[formula.root]:
            return Proof(inference_rule_to_prove, AXIOMATIC_SYSTEM, [Proof.Line(formula)])
        else:
            return Proof(inference_rule_to_prove, AXIOMATIC_SYSTEM,[Proof.Line(Formula('~', formula))])
    else:
        if formula.root == '->':
            if formula_evaluation:
                if not evaluate(formula.first, model):
                    #recursively prove the negation of formula.first
                    neg_formula_first = Formula('~', formula.first)
                    proof_of_neg_first = prove_in_model(neg_formula_first, model)
                    return prove_corollary(proof_of_neg_first, formula, I2)
                else:
                    #recursively prove formula.second
                    proof_of_sec = prove_in_model(formula.second, model)
                    return prove_corollary(proof_of_sec, formula, I1)
            else:
                #recursively prove formula.first
                proof_of_first = prove_in_model(formula.first, model)
                #recursively prove the negation of formula.second
                neg_formula_sec = Formula('~', formula.second)
                proof_of_neg_sec = prove_in_model(neg_formula_sec, model)
                return combine_proofs(proof_of_first, proof_of_neg_sec, Formula('~', formula), NI)
        else:
            if formula_evaluation:
                # recursively prove formula.first
                return prove_in_model(formula.first, model)
            else:
                proof_of_first = prove_in_model(formula.first, model)
                return prove_corollary(proof_of_first, Formula('~', formula), NN)


def reduce_assumption(proof_from_affirmation: Proof,
                      proof_from_negation: Proof) -> Proof:
    """Combines the given two proofs, both of the same formula `conclusion` and
    from the same assumptions except that the last assumption of the latter is
    the negation of that of the former, into a single proof of `conclusion` from
    only the common assumptions.

    Parameters:
        proof_from_affirmation: valid proof of `conclusion` from one or more
            assumptions, the last of which is an assumption `assumption`.
        proof_of_negation: valid proof of `conclusion` from the same assumptions
            and inference rules of `proof_from_affirmation`, but with the last
            assumption being ``'~``\ `assumption` ``'`` instead of `assumption`.

    Returns:
        A valid proof of `conclusion` from only the assumptions common to the
        given proofs (i.e., without the last assumption of each), via the same
        inference rules of the given proofs and in addition
        `~propositions.axiomatic_systems.MP`,
        `~propositions.axiomatic_systems.I0`,
        `~propositions.axiomatic_systems.I1`,
        `~propositions.axiomatic_systems.D`, and
        `~propositions.axiomatic_systems.R`.

    Examples:
        If the two given proofs are of ``['p', 'q'] ==> '(q->p)'`` and of
        ``['p', '~q'] ==> ('q'->'p')``, then the returned proof is of
        ``['p'] ==> '(q->p)'``.
    """
    assert proof_from_affirmation.is_valid()
    assert proof_from_negation.is_valid()
    assert proof_from_affirmation.statement.conclusion == \
           proof_from_negation.statement.conclusion
    assert len(proof_from_affirmation.statement.assumptions) > 0
    assert len(proof_from_negation.statement.assumptions) > 0
    assert proof_from_affirmation.statement.assumptions[:-1] == \
           proof_from_negation.statement.assumptions[:-1]
    assert Formula('~', proof_from_affirmation.statement.assumptions[-1]) == \
           proof_from_negation.statement.assumptions[-1]
    assert proof_from_affirmation.rules == proof_from_negation.rules
    affirmation_without_last_assumption = remove_assumption(proof_from_affirmation)
    negation_without_last_assumption = remove_assumption(proof_from_negation)
    conclusion = proof_from_negation.statement.conclusion
    return combine_proofs(affirmation_without_last_assumption, negation_without_last_assumption, conclusion, R)

def prove_tautology(tautology: Formula, model: Model = frozendict()) -> Proof:
    """Proves the given tautology from the formulae that capture the given
    model.

    Parameters:
        tautology: tautology that contains no constants or operators beyond
            ``'->'`` and ``'~'``, to prove.
        model: model over a (possibly empty) prefix (with respect to the
            alphabetical order) of the variables of `tautology`, from whose
            formulae to prove.

    Returns:
        A valid proof of the given tautology from the formulae that capture the
        given model, in the order returned by
        `formulae_capturing_model`\ ``(``\ `model`\ ``)``, via
        `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM`.

    Examples:
        If the given model is the empty dictionary, then the returned proof is
        of the given tautology from no assumptions.
    """
    assert is_tautology(tautology)
    assert tautology.operators().issubset({'->', '~'})
    assert is_model(model)
    assert sorted(tautology.variables())[:len(model)] == sorted(model.keys())
    tautology_variables = list(sorted(tautology.variables()))
    if len(model) == len(tautology_variables):
        return prove_in_model(tautology, model)
    else:
        #craete a model with added assumption, assigned to be True, and another model
        #with the same added assumption assigned to be False
        model_with_negation =  dict(model)
        model_with_negation[tautology_variables[len(model)]] = False
        model_with_affirmation = dict(model)
        model_with_affirmation[tautology_variables[len(model)]] = True
        proof_from_affirmation = prove_tautology(tautology, model_with_affirmation)
        proof_from_negation = prove_tautology(tautology, model_with_negation)
        return reduce_assumption(proof_from_affirmation, proof_from_negation)


def proof_or_counterexample(formula: Formula) -> Union[Proof, Model]:
    """Either proves the given formula or finds a model in which it does not
    hold.

    Parameters:
        formula: formula that contains no constants or operators beyond ``'->'``
            and ``'~'``, to either prove or find a counterexample for.

    Returns:
        If the given formula is a tautology, then an assumptionless proof of the
        formula via `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM`,
        otherwise a model in which the given formula does not hold.
    """
    assert formula.operators().issubset({'->', '~'})
    if is_tautology(formula):
        return prove_tautology(formula, {})
    for model in all_models(list(formula.variables())):
        if not evaluate(formula, model):
            return model

def encode_as_formula(rule: InferenceRule) -> Formula:
    """Encodes the given inference rule as a formula consisting of a chain of
    implications.

    Parameters:
        rule: inference rule to encode.

    Returns:
        The formula encoding the given rule.

    Examples:
        >>> encode_as_formula(InferenceRule([Formula('p1'), Formula('p2'),
        ...                                  Formula('p3'), Formula('p4')],
        ...                                 Formula('q')))
        (p1->(p2->(p3->(p4->q))))
        >>> encode_as_formula(InferenceRule([], Formula('q')))
        q
    """
    if len(rule.assumptions) == 0:
        return rule.conclusion
    formula = Formula('->', rule.assumptions[-1], rule.conclusion)
    for assumption in reversed(rule.assumptions[:-1]):
        formula = Formula('->', assumption, formula)
    return formula

def prove_sound_inference(rule: InferenceRule) -> Proof:
    """Proves the given sound inference rule.

    Parameters:
        rule: sound inference rule whose assumptions and conclusion that contain
            no constants or operators beyond ``'->'`` and ``'~'``, to prove.

    Returns:
        A valid assumptionless proof of the given sound inference rule via
        `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM`.
    """
    assert is_sound_inference(rule)
    for formula in rule.assumptions + (rule.conclusion,):
        assert formula.operators().issubset({'->', '~'})
    formula_of_rule = encode_as_formula(rule)
    proof_of_formula = prove_tautology(formula_of_rule, {})
    all_lines = list(proof_of_formula.lines)
    num_of_assumptions = 0 #num of assumptions added to the proof
    while all_lines[-1].formula != rule.conclusion:
        all_lines.append(Proof.Line(rule.assumptions[num_of_assumptions]))
        all_lines.append(Proof.Line(all_lines[-2].formula.second, MP, (len(all_lines)-1, len(all_lines)-2)))
        num_of_assumptions += 1
    return Proof(rule, AXIOMATIC_SYSTEM, all_lines)


def model_or_inconsistency(formulae: List[Formula]) -> Union[Model, Proof]:
    """Either finds a model in which all the given formulae hold, or proves
    ``'~(p->p)'`` from these formula.

    Parameters:
        formulae: formulae that use only the operators ``'->'`` and ``'~'``, to
            either find a model for or prove ``'~(p->p)'`` from.

    Returns:
        A model in which all of the given formulae hold if such exists,
        otherwise a proof of '~(p->p)' from the given formulae via
        `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM`.
    """
    for formula in formulae:
        assert formula.operators().issubset({'->', '~'})
    # Task 6.5
    all_vars  = set()
    for formula in formulae:
        all_vars.update(formula.variables())
    for model in all_models(list(all_vars)):
        correct_model = True
        for formula in formulae:
            if not evaluate(formula, model):
                correct_model = False
        if correct_model:
            return model
    conclusion = Formula.parse('~(p->p)')
    rule_to_prove = InferenceRule(formulae, conclusion)
    return prove_sound_inference(rule_to_prove)


def prove_in_model_full(formula: Formula, model: Model) -> Proof:
    """Either proves the given formula or proves its negation, from the formulae
    that capture the given model.

    Parameters:
        formula: formula that contains no operators beyond ``'->'``, ``'~'``,
            ``'&'``, and ``'|'``, whose affirmation or negation is to prove.
        model: model from whose formulae to prove.

    Returns:
        If the given formula evaluates to ``True`` in the given model, then
        a proof of the formula, otherwise a proof of ``'~``\ `formula`\ ``'``.
        The returned proof is from the formulae that capture the given model, in
        the order returned by `formulae_capturing_model`\ ``(``\ `model`\ ``)``,
        via `~propositions.axiomatic_systems.AXIOMATIC_SYSTEM_FULL`.
    """
    assert formula.operators().issubset({'T', 'F', '->', '~', '&', '|'})
    assert is_model(model)
    # Optional Task 6.6
