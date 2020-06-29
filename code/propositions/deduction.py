# (c) This file is part of the course
# Mathematical Logic through Programming
# by Gonczarowski and Nisan.
# File name: propositions/deduction.py

"""Useful proof manipulation maneuvers in propositional logic."""

from propositions.syntax import *
from propositions.proofs import *
from propositions.axiomatic_systems import *

def prove_corollary(antecedent_proof: Proof, consequent: Formula,
                    conditional: InferenceRule) -> Proof:
    """Converts the given proof of a formula `antecedent` into a proof of the
    given formula `consequent` by using the given assumptionless inference rule
    of which ``'(``\ `antecedent`\ ``->``\ `consequent`\ ``)'`` is a
    specialization.

    Parameters:
        antecedent_proof: valid proof of `antecedent`.
        consequent: formula to prove.
        conditional: assumptionless inference rule of which the assumptionless
            inference rule with conclusion
            ``'(``\ `antecedent`\ ``->``\ `consequent`\ ``)'`` is a
            specialization.

    Returns:
        A valid proof of `consequent` from the same assumptions as the given
        proof, via the same inference rules as the given proof and in addition
        `~propositions.axiomatic_systems.MP` and `conditional`.
    """
    assert antecedent_proof.is_valid()
    assert InferenceRule([],
                         Formula('->', antecedent_proof.statement.conclusion,
                                 consequent)).is_specialization_of(conditional)
    # Task 5.3a
    all_lines = list(antecedent_proof.lines)
    inference_rule_to_prove = InferenceRule(antecedent_proof.statement.assumptions, consequent)
    conditional_specialization = Formula('->', antecedent_proof.statement.conclusion, consequent)
    all_lines.append(Proof.Line(conditional_specialization, conditional, ()))
    len_of_antecedent_proof = len(antecedent_proof.lines)
    all_lines.append(Proof.Line(consequent, MP, (len_of_antecedent_proof - 1, len_of_antecedent_proof)))
    new_rules = list(antecedent_proof.rules) + [MP, conditional]
    return Proof(inference_rule_to_prove, set(new_rules), all_lines)

def combine_proofs(antecedent1_proof: Proof, antecedent2_proof: Proof,
                   consequent: Formula, double_conditional: InferenceRule) -> \
        Proof:
    """Combines the given proofs of two formulae `antecedent1` and `antecedent2`
    into a proof of the given formula `consequent` by using the given
    assumptionless inference rule of which
    ``('``\ `antecedent1`\ ``->(``\ `antecedent2`\ ``->``\ `consequent`\ ``))'``
    is a specialization.

    Parameters:
        antecedent1_proof: valid proof of `antecedent1`.
        antecedent2_proof: valid proof of `antecedent2` from the same
            assumptions and inference rules as `antecedent1_proof`.
        consequent: formula to prove.
        double_conditional: assumptionless inference rule of which the
            assumptionless inference rule with conclusion
            ``'(``\ `antecedent1`\ ``->(``\ `antecedent2`\ ``->``\ `consequent`\ ``))'``
            is a specialization.

    Returns:
        A valid proof of `consequent` from the same assumptions as the given
        proofs, via the same inference rules as the given proofs and in addition
        `~propositions.axiomatic_systems.MP` and `conditional`.
    """
    assert antecedent1_proof.is_valid()
    assert antecedent2_proof.is_valid()
    assert antecedent1_proof.statement.assumptions == \
           antecedent2_proof.statement.assumptions
    assert antecedent1_proof.rules == antecedent2_proof.rules
    assert InferenceRule(
        [], Formula('->', antecedent1_proof.statement.conclusion,
        Formula('->', antecedent2_proof.statement.conclusion, consequent))
        ).is_specialization_of(double_conditional)

    # first add the proof of antecedent1
    all_lines = list(antecedent1_proof.lines)
    inference_rule_to_prove = InferenceRule(antecedent1_proof.statement.assumptions, consequent)
    double_conditional_specialization = Formula('->', antecedent1_proof.statement.conclusion,
                                         Formula('->', antecedent2_proof.statement.conclusion, consequent))
    all_lines.append(Proof.Line(double_conditional_specialization, double_conditional, ()))
    # save the index of the line of which we concluded a -> (b -> c)
    num_line_of_double_conditional = len(all_lines) - 1
    # we will conclude from 'a' and 'a -> (b -> c)': '(b -> c)'
    # but first we will create the appropriate formula to deduce (b->c)
    conclusion_of_double_conditional = Formula('->', antecedent2_proof.statement.conclusion, consequent)
    num_line_of_conclusion_of_antecedent1 = len(antecedent1_proof.lines) - 1
    all_lines.append(Proof.Line(conclusion_of_double_conditional, MP,
                                (num_line_of_conclusion_of_antecedent1, num_line_of_double_conditional)))
    # save the index of the line of which we concluded (b -> c)
    num_line_of_sub_conclusion = len(all_lines) - 1
    # add the proof of antecedent2
    new_antecedent2_proof_lines = []
    for line in antecedent2_proof.lines:
        indices = None
        if not line.is_assumption():
            indices = tuple(i + len(all_lines) for i in line.assumptions)
        new_line = Proof.Line(line.formula, line.rule, indices)
        new_antecedent2_proof_lines.append(new_line)
    all_lines += new_antecedent2_proof_lines
    num_line_of_conclusion_of_antecedent2 = len(all_lines) - 1
    all_lines.append(Proof.Line(consequent, MP,
                                (num_line_of_conclusion_of_antecedent2, num_line_of_sub_conclusion)))

    new_rules = list(antecedent1_proof.rules) + list(antecedent2_proof.rules) + [MP, double_conditional]
    return Proof(inference_rule_to_prove, set(new_rules), all_lines)


def remove_assumption(proof: Proof) -> Proof:
    """Converts a proof of some `conclusion` formula, the last assumption of
    which is an assumption `assumption`, into a proof of
    ``'(``\ `assumption`\ ``->``\ `conclusion`\ ``)'`` from the same assumptions
    except `assumption`.

    Parameters:
        proof: valid proof to convert, with at least one assumption, via some
            set of inference rules all of which have no assumptions except
            perhaps `~propositions.axiomatic_systems.MP`.

    Return:
        A valid proof of ``'(``\ `assumptions`\ ``->``\ `conclusion`\ ``)'``
        from the same assumptions as the given proof except the last one, via
        the same inference rules as the given proof and in addition
        `~propositions.axiomatic_systems.MP`,
        `~propositions.axiomatic_systems.I0`,
        `~propositions.axiomatic_systems.I1`, and
        `~propositions.axiomatic_systems.D`.
    """        
    assert proof.is_valid()
    assert len(proof.statement.assumptions) > 0
    for rule in proof.rules:
        assert rule == MP or len(rule.assumptions) == 0
    assumptions_without_fi = list(proof.statement.assumptions)[:-1]
    fi = proof.statement.assumptions[-1]
    all_lines = []
    # we will keep a map that maps a line number in proofs.lines to the number
    # of lines added to the new proofs lines
    indices_map = {}
    for num_line, line in enumerate(proof.lines):
        if line.formula in assumptions_without_fi:
            new_formula_with_I1 = Formula('->', line.formula, Formula('->', fi, line.formula))
            new_formula_with_MP = Formula('->', fi, line.formula)
            all_lines.append(Proof.Line(line.formula, None, None))
            index_of_assumption = len(all_lines) - 1
            all_lines.append(Proof.Line(new_formula_with_I1, I1, ()))
            all_lines.append(Proof.Line(new_formula_with_MP, MP, (index_of_assumption, index_of_assumption+1)))
            indices_map[num_line] = 3
        elif line.formula == fi:
            new_formula = Formula('->', fi, fi)
            all_lines.append(Proof.Line(new_formula, I0, ()))
            indices_map[num_line] = 1
        elif line.rule == MP:
            first_formula_of_original_mp = proof.lines[line.assumptions[0]].formula
            # print('first_formula_of_original_mp ', first_formula_of_original_mp)
            new_formula_with_D = Formula('->', Formula('->', fi, first_formula_of_original_mp),
                                         Formula('->', fi, line.formula))
            new_formula_with_MP = Formula('->', fi, line.formula)
            first_assumption_index = -1 #starts with -1 since indices starts with 0
            for i in range(line.assumptions[0] + 1):
                first_assumption_index += indices_map[i]
            sec_assumption_index = -1
            for i in range(line.assumptions[1] + 1):
                sec_assumption_index += indices_map[i]
            # this is the formula that represents the relevant specialization of D
            formula_of_D = Formula('->', all_lines[sec_assumption_index].formula, new_formula_with_D)
            #we will add it to the proof and keep it's index
            all_lines.append(Proof.Line(formula_of_D, D, ()))
            index_of_formula_of_D = len(all_lines) - 1
            all_lines.append(Proof.Line(new_formula_with_D, MP, (sec_assumption_index, index_of_formula_of_D)))
            cur_index = len(all_lines) - 1
            all_lines.append(Proof.Line(new_formula_with_MP, MP, (first_assumption_index, cur_index)))
            indices_map[num_line] = 3
        else:
            new_formula_with_I1 = Formula('->', line.formula, Formula('->', fi, line.formula))
            new_formula_with_MP = Formula('->', fi, line.formula)
            all_lines.append(line)
            index_of_formula = len(all_lines) - 1
            all_lines.append(Proof.Line(new_formula_with_I1, I1, ()))
            all_lines.append(Proof.Line(new_formula_with_MP, MP, (index_of_formula, index_of_formula + 1)))
            indices_map[num_line] = 3
    new_conclusion = Formula('->', fi, proof.statement.conclusion)
    rule_to_prove = InferenceRule(tuple(assumptions_without_fi), new_conclusion)
    new_rules = list(proof.rules) + [MP, I0, I1, D]
    return Proof(rule_to_prove, set(new_rules), all_lines)

def proof_from_inconsistency(proof_of_affirmation: Proof,
                             proof_of_negation: Proof, conclusion: Formula) -> \
        Proof:
    """Combines the given proofs of a formula `affirmation` and its negation
    ``'~``\ `affirmation`\ ``'`` into a proof of the given formula.

    Parameters:
        proof_of_affirmation: valid proof of `affirmation`.
        proof_of_negation: valid proof of ``'~``\ `affirmation`\ ``'`` from the
            same assumptions and inference rules of `proof_of_affirmation`.

    Returns:
        A valid proof of `conclusion` from the same assumptions as the given
        proofs, via the same inference rules as the given proofs and in addition
        `~propositions.axiomatic_systems.MP` and
        `~propositions.axiomatic_systems.I2`.
    """
    assert proof_of_affirmation.is_valid()
    assert proof_of_negation.is_valid()
    assert proof_of_affirmation.statement.assumptions == \
           proof_of_negation.statement.assumptions
    assert Formula('~', proof_of_affirmation.statement.conclusion) == \
           proof_of_negation.statement.conclusion
    assert proof_of_affirmation.rules == proof_of_negation.rules
    return combine_proofs(proof_of_negation, proof_of_affirmation, conclusion, I2)

def prove_by_contradiction(proof: Proof) -> Proof:
    """Converts the given proof of ``'~(p->p)'``, the last assumption of which
    is an assumption ``'~``\ `formula`\ ``'``, into a proof of `formula` from
    the same assumptions except ``'~``\ `formula`\ ``'``.

    Parameters:
        proof: valid proof of ``'~(p->p)'`` to convert, the last assumption of
            which is of the form ``'~``\ `formula`\ ``'``, via some set of
            inference rules all of which have no assumptions except perhaps
            `~propositions.axiomatic_systems.MP`.

    Return:
        A valid proof of `formula` from the same assumptions as the given proof
        except the last one, via the same inference rules as the given proof and
        in addition `~propositions.axiomatic_systems.MP`,
        `~propositions.axiomatic_systems.I0`,
        `~propositions.axiomatic_systems.I1`,
        `~propositions.axiomatic_systems.D`, and
        `~propositions.axiomatic_systems.N`.
    """
    assert proof.is_valid()
    assert proof.statement.conclusion == Formula.parse('~(p->p)')
    assert len(proof.statement.assumptions) > 0
    assert proof.statement.assumptions[-1].root == '~'
    for rule in proof.rules:
        assert rule == MP or len(rule.assumptions) == 0
    proof_without_last_assumption = remove_assumption(proof)
    all_lines = list(proof_without_last_assumption.lines)
    index_of_first_conclusion = len(all_lines) - 1
    new_rules = list(proof_without_last_assumption.rules) + [N]
    conclusion_of_proof_without_last_assumption = proof_without_last_assumption.statement.conclusion
    conclusion_of_N = Formula('->', conclusion_of_proof_without_last_assumption.second.first,
                                   conclusion_of_proof_without_last_assumption.first.first)
    formula_of_N = Formula('->', conclusion_of_proof_without_last_assumption, conclusion_of_N)
    all_lines.append(Proof.Line(formula_of_N, N, ()))
    index_of_formula_of_N = len(all_lines) - 1
    all_lines.append(Proof.Line(conclusion_of_N, MP, (index_of_first_conclusion,index_of_formula_of_N)))
    #now we will prove that p->p
    formula = Formula('->', Formula('p'), Formula('p'))
    all_lines.append(Proof.Line(formula, I0, ()))
    all_lines.append(Proof.Line(proof.statement.assumptions[-1].first, MP, (len(all_lines)-1, index_of_formula_of_N+1)))
    new_conclusion = proof.statement.assumptions[-1].first
    new_statement = InferenceRule(proof.statement.assumptions[:-1], new_conclusion)
    return Proof(new_statement, set(new_rules), all_lines)
