# (c) This file is part of the course
# Mathematical Logic through Programming
# by Gonczarowski and Nisan.
# File name: predicates/deduction.py

"""Useful proof manipulation maneuvers in predicate logic."""

from predicates.syntax import *
from predicates.proofs import *
from predicates.prover import *


def remove_assumption(proof: Proof, assumption: Formula,
                      print_as_proof_forms: bool = False) -> Proof:
    """Converts the given proof of some `conclusion` formula, an assumption of
    which is `assumption`, to a proof of
    ``'(``\ `assumption`\ ``->``\ `conclusion`\ ``)'`` from the same assumptions
    except `assumption`.

    Parameters:
        proof: valid proof to convert, from assumptions/axioms that include
            `~predicates.prover.Prover.AXIOMS`.
        assumption: formula that is a simple assumption (i.e., without any
            templates) of the given proof, such that no line of the given proof
            is a UG line over a variable that is free in this assumption.

    Returns:
        A valid proof of ``'(``\ `assumption`\ ``->``\ `conclusion`\ ``)'``
        from the same assumptions/axioms as the given proof except `assumption`.
    """
    assert proof.is_valid()
    assert Schema(assumption) in proof.assumptions
    assert proof.assumptions.issuperset(Prover.AXIOMS)
    for line in proof.lines:
        if isinstance(line, Proof.UGLine):
            assert line.formula.variable not in assumption.free_variables()
    # Task 11.1
    assumptions = Prover.AXIOMS.union(proof.assumptions) - {Schema(assumption)}
    prover = Prover(assumptions)
    new_dict = dict()
    for i, line in enumerate(proof.lines):
        if isinstance(line, Proof.AssumptionLine):
            if line.formula == assumption:
                assump_line = prover.add_tautology(Formula("->", assumption, assumption))
                new_dict[i] = assump_line
                continue
            assump_line = prover.add_instantiated_assumption(line.formula, line.assumption, line.instantiation_map)
            line_num = prover.add_tautological_implication(Formula("->", assumption, line.formula), {assump_line})
            new_dict[i] = line_num
        elif isinstance(line, Proof.MPLine):
            mp_num = prover.add_tautological_implication(Formula("->", assumption, line.formula),
                                                         {new_dict[line.antecedent_line_number],
                                                          new_dict[line.conditional_line_number]})
            new_dict[i] = mp_num
        elif isinstance(line, Proof.UGLine):
            first_form = Formula("A", line.formula.variable,
                                 Formula("->", assumption, proof.lines[line.predicate_line_number].formula))
            ug_form = Formula("->", assumption, line.formula)
            ug_line = prover.add_ug(first_form, new_dict[line.predicate_line_number])
            us_line = prover.add_instantiated_assumption(Formula("->", first_form, ug_form), Prover.US,
                                                         {'x': line.formula.variable, 'R': Prover.replace_var_in_str(
                                                             proof.lines[line.predicate_line_number].formula.__repr__(),
                                                             line.formula.variable, '_'), 'Q': assumption})
            mp_line = prover.add_mp(ug_form, ug_line, us_line)
            new_dict[i] = mp_line
        elif isinstance(line, Proof.TautologyLine):
            taut = prover.add_tautology(line.formula)
            ug_num = prover.add_tautological_implication(Formula("->", assumption, line.formula), {taut})
            new_dict[i] = ug_num
    return prover.qed()


def proof_by_way_of_contradiction(proof: Proof, assumption: Formula,
                                  print_as_proof_forms: bool = False) -> Proof:
    """Converts the given proof of a contradiction, an assumption of which is
    `assumption`, to a proof of ``'~``\ `assumption`\ ``'`` from the same
    assumptions except `assumption`.

    Parameters:
        proof: valid proof of a contradiction (i.e., a formula whose negation is
            a tautology) to convert, from assumptions/axioms that include
            `~predicates.prover.Prover.AXIOMS`.
        assumption: formula that is a simple assumption (i.e., without any
            templates) of the given proof, such that no line of the given proof
            is a UG line over a variable that is free in this assumption.

    Return:
        A valid proof of ``'~``\ `assumption`\ ``'`` from the same
        assumptions/axioms as the given proof except `assumption`.
    """
    assert proof.is_valid()
    assert Schema(assumption) in proof.assumptions
    assert proof.assumptions.issuperset(Prover.AXIOMS)
    for line in proof.lines:
        if isinstance(line, Proof.UGLine):
            assert line.formula.variable not in assumption.free_variables()
    # Task 11.2
    assumptions = Prover.AXIOMS.union(proof.assumptions) - {Schema(assumption)}
    prover = Prover(assumptions)
    new_proof = remove_assumption(proof, assumption)
    conc_line = prover.add_proof(new_proof.conclusion, new_proof)
    taut_line = prover.add_tautology(Formula('~', proof.conclusion))
    conc_taut = prover.add_tautology(
        Formula("->", new_proof.conclusion, Formula("->", Formula('~', proof.conclusion), Formula('~', assumption))))
    mp1 = prover.add_mp(Formula("->", Formula('~', proof.conclusion), Formula('~', assumption)), conc_line, conc_taut)
    mp2 = prover.add_mp(Formula('~', assumption), taut_line, mp1)
    return prover.qed()
