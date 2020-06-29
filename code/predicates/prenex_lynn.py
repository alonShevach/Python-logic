# (c) This file is part of the course
# Mathematical Logic through Programming
# by Gonczarowski and Nisan.
# File name: predicates/prenex.py

"""Conversion of predicate-logic formulas into prenex normal form."""

from typing import Tuple

from logic_utils import fresh_variable_name_generator

from predicates.syntax import *
from predicates.proofs import *
from predicates.prover import *

#: Additional axioms of quantification for first-order predicate logic.
ADDITIONAL_QUANTIFICATION_AXIOMS = (
    Schema(Formula.parse('((~Ax[R(x)]->Ex[~R(x)])&(Ex[~R(x)]->~Ax[R(x)]))'),
           {'x', 'R'}),
    Schema(Formula.parse('((~Ex[R(x)]->Ax[~R(x)])&(Ax[~R(x)]->~Ex[R(x)]))'),
           {'x', 'R'}),
    Schema(Formula.parse('(((Ax[R(x)]&Q())->Ax[(R(x)&Q())])&'
                         '(Ax[(R(x)&Q())]->(Ax[R(x)]&Q())))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Ex[R(x)]&Q())->Ex[(R(x)&Q())])&'
                         '(Ex[(R(x)&Q())]->(Ex[R(x)]&Q())))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Q()&Ax[R(x)])->Ax[(Q()&R(x))])&'
                         '(Ax[(Q()&R(x))]->(Q()&Ax[R(x)])))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Q()&Ex[R(x)])->Ex[(Q()&R(x))])&'
                         '(Ex[(Q()&R(x))]->(Q()&Ex[R(x)])))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Ax[R(x)]|Q())->Ax[(R(x)|Q())])&'
                         '(Ax[(R(x)|Q())]->(Ax[R(x)]|Q())))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Ex[R(x)]|Q())->Ex[(R(x)|Q())])&'
                         '(Ex[(R(x)|Q())]->(Ex[R(x)]|Q())))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Q()|Ax[R(x)])->Ax[(Q()|R(x))])&'
                         '(Ax[(Q()|R(x))]->(Q()|Ax[R(x)])))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Q()|Ex[R(x)])->Ex[(Q()|R(x))])&'
                         '(Ex[(Q()|R(x))]->(Q()|Ex[R(x)])))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Ax[R(x)]->Q())->Ex[(R(x)->Q())])&'
                         '(Ex[(R(x)->Q())]->(Ax[R(x)]->Q())))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Ex[R(x)]->Q())->Ax[(R(x)->Q())])&'
                         '(Ax[(R(x)->Q())]->(Ex[R(x)]->Q())))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Q()->Ax[R(x)])->Ax[(Q()->R(x))])&'
                         '(Ax[(Q()->R(x))]->(Q()->Ax[R(x)])))'), {'x','R','Q'}),
    Schema(Formula.parse('(((Q()->Ex[R(x)])->Ex[(Q()->R(x))])&'
                         '(Ex[(Q()->R(x))]->(Q()->Ex[R(x)])))'), {'x','R','Q'}),
    Schema(Formula.parse('(((R(x)->Q(x))&(Q(x)->R(x)))->'
                         '((Ax[R(x)]->Ay[Q(y)])&(Ay[Q(y)]->Ax[R(x)])))'),
           {'x', 'y', 'R', 'Q'}),
    Schema(Formula.parse('(((R(x)->Q(x))&(Q(x)->R(x)))->'
                         '((Ex[R(x)]->Ey[Q(y)])&(Ey[Q(y)]->Ex[R(x)])))'),
           {'x', 'y', 'R', 'Q'}))

def is_quantifier_free(formula: Formula) -> bool:
    """Checks if the given formula contains any quantifiers.

    Parameters:
        formula: formula to check.

    Returns:
        ``False`` if the given formula contains any quantifiers, ``True``
        otherwise.
    """
    # Task 11.3.1
    if is_equality(formula.root) or is_relation(formula.root):
        return True
    elif is_unary(formula.root):
        return is_quantifier_free(formula.first)
    elif is_binary(formula.root):
        return is_quantifier_free(formula.first) and is_quantifier_free(formula.second)
    return False

def is_in_prenex_normal_form(formula: Formula) -> bool:
    """Checks if the given formula is in prenex normal form.

    Parameters:
        formula: formula to check.

    Returns:
        ``True`` if the given formula in prenex normal form, ``False``
        otherwise.
    """
    # Task 11.3.2
    rel_formula = formula
    while is_quantifier(rel_formula.root):
        rel_formula = rel_formula.predicate
    if is_unary(rel_formula.root):
        return is_quantifier_free(rel_formula.first)
    elif is_binary(rel_formula.root):
        return is_quantifier_free(rel_formula.first) and is_quantifier_free(rel_formula.second)
    return (is_relation(rel_formula.root) or is_equality(rel_formula.root))


def equivalence_of(formula1: Formula, formula2: Formula) -> Formula:
    """States the equivalence of the two given formulas as a formula.

    Parameters:
        formula1: first of the formulas the equivalence of which is to be
            stated.
        formula2: second of the formulas the equivalence of which is to be
            stated.

    Returns:
        The formula ``'((``\ `formula1`\ ``->``\ `formula2`\ ``)&(``\ `formula2`\ ``->``\ `formula1`\ ``))'``.
    """
    return Formula('&', Formula('->', formula1, formula2),
                   Formula('->', formula2, formula1))

def has_uniquely_named_variables(formula: Formula) -> bool:
    """Checks if the given formula has uniquely named variables.

    Parameters:
        formula: formula to check.

    Returns:
        ``False`` if in the given formula some variable name has both quantified
        and free occurrences or is quantified by more than one quantifier,
        ``True`` otherwise.
    """
    forbidden_variables = set(formula.free_variables())
    def has_uniquely_named_variables_helper(formula: Formula) -> bool:
        if is_unary(formula.root):
            return has_uniquely_named_variables_helper(formula.first)
        elif is_binary(formula.root):
            return has_uniquely_named_variables_helper(formula.first) and \
                   has_uniquely_named_variables_helper(formula.second)
        elif is_quantifier(formula.root):
            if formula.variable in forbidden_variables:
                return False
            forbidden_variables.add(formula.variable)
            return has_uniquely_named_variables_helper(formula.predicate)
        else:
            assert is_relation(formula.root) or is_equality(formula.root)
            return True

    return has_uniquely_named_variables_helper(formula)


def uniquely_rename_quantified_variables(formula: Formula) -> \
        Tuple[Formula, Proof]:
    """Converts the given formula to an equivalent formula with uniquely named
    variables, and proves the equivalence of these two formulas.

    Parameters:
        formula: formula to convert, which contains no variable names starting
            with ``z``.

    Returns:
        A pair. The first element of the pair is a formula equivalent to the
        given formula, with the exact same structure but with the additional
        property of having uniquely named variables, obtained by consistently
        replacing each variable name that is bound in the given formula with a
        new variable name obtained by calling
        `next`\ ``(``\ `~logic_utils.fresh_variable_name_generator`\ ``)``. The
        second element of the pair is a proof of the equivalence of the given
        formula and the returned formula (i.e., a proof of
        `equivalence_of`\ ``(``\ `formula`\ ``,``\ `returned_formula`\ ``)``)
        via `~predicates.prover.Prover.AXIOMS` and
        `ADDITIONAL_QUANTIFICATION_AXIOMS`.
    """
    # Task 11.5
    if is_unary(formula.root):
        prover = Prover(Prover.AXIOMS.union(ADDITIONAL_QUANTIFICATION_AXIOMS))
        new_first, proof = uniquely_rename_quantified_variables(formula.first)
        new_formula = Formula(formula.root, new_first)
        conclusion_of_proof = prover.add_proof(proof.conclusion, proof)
        equivalence = equivalence_of(formula, new_formula)
        prover.add_tautological_implication(equivalence, {conclusion_of_proof})
        return (new_formula, prover.qed())
    elif is_binary(formula.root):
        prover = Prover(Prover.AXIOMS.union(ADDITIONAL_QUANTIFICATION_AXIOMS))
        new_first, proof_of_first = uniquely_rename_quantified_variables(formula.first)
        new_sec, proof_of_sec = uniquely_rename_quantified_variables(formula.second)
        new_formula = Formula(formula.root, new_first, new_sec)
        conclusion_of_first = prover.add_proof(proof_of_first.conclusion, proof_of_first)
        equivalence_of_first = equivalence_of(formula.first, new_first)
        i = prover.add_tautological_implication(equivalence_of_first, {conclusion_of_first})
        conclusion_of_sec = prover.add_proof(proof_of_sec.conclusion, proof_of_sec)
        equivalence_of_sec = equivalence_of(formula.second, new_sec)
        j = prover.add_tautological_implication(equivalence_of_sec, {conclusion_of_sec})
        equivalence = equivalence_of(formula, new_formula)
        prover.add_tautological_implication(equivalence, {i, j})
        return (new_formula, prover.qed())
    elif is_relation(formula.root) or is_equality(formula.root):
        prover = Prover(Prover.AXIOMS.union(ADDITIONAL_QUANTIFICATION_AXIOMS))
        prover.add_tautology(equivalence_of(formula, formula))
        return (formula, prover.qed())
    elif is_quantifier(formula.root):
        prover = Prover(Prover.AXIOMS.union(ADDITIONAL_QUANTIFICATION_AXIOMS))
        new_var = next(fresh_variable_name_generator)
        new_predicate, proof = uniquely_rename_quantified_variables(formula.predicate)
        replaced_predicate_formula = new_predicate.substitute({formula.variable: Term(new_var)})
        new_formula = Formula(formula.root, new_var, replaced_predicate_formula)
        equivalence = equivalence_of(formula, new_formula)
        conclusion_of_proof = prover.add_proof(proof.conclusion, proof)
        rel_assumption = ADDITIONAL_QUANTIFICATION_AXIOMS[14]
        if formula.root == 'E':
            rel_assumption = ADDITIONAL_QUANTIFICATION_AXIOMS[15]
        instantiated_assumption = Formula('->',proof.conclusion, equivalence)
        instantiation_map = {'x': formula.variable, 'y': new_var,
                             'R': formula.predicate.substitute({formula.variable: Term('_')}),
                             'Q': replaced_predicate_formula.substitute({new_var: Term('_')})}
        i_a = prover.add_instantiated_assumption(instantiated_assumption, rel_assumption, instantiation_map)
        prover.add_mp(equivalence, conclusion_of_proof, i_a)
        return (new_formula, prover.qed())

def pull_out_quantifications_across_negation(formula: Formula) -> \
        Tuple[Formula, Proof]:
    """Converts the given formula with uniquely named variables of the form
    ``'~``\ `Q1`\ `x1`\ ``[``\ `Q2`\ `x2`\ ``[``...\ `Qn`\ `xn`\ ``[``\ `inner_formula`\ ``]``...\ ``]]'``
    to an equivalent formula of the form
    ``'``\ `Q'1`\ `x1`\ ``[``\ `Q'2`\ `x2`\ ``[``...\ `Q'n`\ `xn`\ ``[~``\ `inner_formula`\ ``]``...\ ``]]'``,
    and proves the equivalence of these two formulas.

    Parameters:
        formula: formula to convert, whose root is a negation, i.e., which is of
            the form
            ``'~``\ `Q1`\ `x1`\ ``[``\ `Q2`\ `x2`\ ``[``...\ `Qn`\ `xn`\ ``[``\ `inner_formula`\ ``]``...\ ``]]'``
            where `n`>=0, each `Qi` is a quantifier, each `xi` is a variable
            name, and `inner_formula` does not start with a quantifier.

    Returns:
        A pair. The first element of the pair is a formula equivalent to the
        given formula, but of the form
        ``'``\ `Q'1`\ `x1`\ ``[``\ `Q'2`\ `x2`\ ``[``...\ `Q'n`\ `xn`\ ``[~``\ `inner_formula`\ ``]``...\ ``]]'``
        where each `Q'i` is a quantifier, and where the `xi` variable names and
        `inner_formula` are the same as in the given formula. The second element
        of the pair is a proof of the equivalence of the given formula and the
        returned formula (i.e., a proof of
        `equivalence_of`\ ``(``\ `formula`\ ``,``\ `returned_formula`\ ``)``)
        via `~predicates.prover.Prover.AXIOMS` and
        `ADDITIONAL_QUANTIFICATION_AXIOMS`.

    Examples:
        >>> formula = Formula.parse('~Ax[Ey[R(x,y)]]')
        >>> returned_formula, proof = pull_out_quantifications_across_negation(
        ...     formula)
        >>> returned_formula
        Ex[Ay[~R(x,y)]]
        >>> proof.is_valid()
        True
        >>> proof.conclusion == equivalence_of(formula, returned_formula)
        True
        >>> proof.assumptions == Prover.AXIOMS.union(
        ...     ADDITIONAL_QUANTIFICATION_AXIOMS)
        True
    """
    assert is_unary(formula.root)
    # Task 11.6
    prover = Prover(Prover.AXIOMS.union(ADDITIONAL_QUANTIFICATION_AXIOMS))
    if not is_quantifier(formula.first.root):
        prover.add_tautology(equivalence_of(formula, formula))
        return (formula, prover.qed())
    neg_of_sec = Formula('~', formula.first.predicate)
    new_first, proof = pull_out_quantifications_across_negation(neg_of_sec)
    conclusion_of_proof = prover.add_proof(proof.conclusion, proof)

    rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[0]
    rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[15]
    new_quantifier = 'E'
    if formula.first.root == 'E':
        new_quantifier = 'A'
        rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[1]
        rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[14]

    quantified_new_first = Formula(new_quantifier, formula.first.variable, new_first)
    new_formula = Formula(new_quantifier, formula.first.variable, neg_of_sec)
    equivalence1 = equivalence_of(new_formula, quantified_new_first)
    instantiated_assumption1 = Formula('->', proof.conclusion, equivalence1)
    instantiation_map1 = {'x': formula.first.variable, 'y': formula.first.variable,
                         'R': neg_of_sec.substitute({formula.first.variable: Term('_')}),
                         'Q': new_first.substitute({formula.first.variable: Term('_')})}
    i1 = prover.add_instantiated_assumption(instantiated_assumption1, rel_assumption2, instantiation_map1)
    i2 = prover.add_mp(equivalence1, conclusion_of_proof, i1)
    equivalence2 = equivalence_of(formula, new_formula)
    i3 = prover.add_instantiated_assumption(equivalence2, rel_assumption1,
                                           {'x': formula.first.variable,
                                            'R': formula.first.predicate.substitute({formula.first.variable: Term('_')})})
    equivalence3 = equivalence_of(formula, quantified_new_first)
    prover.add_tautological_implication(equivalence3, {i2, i3})
    return (quantified_new_first, prover.qed())


def pull_out_quantifications_from_left_across_binary_operator(formula:
                                                              Formula) -> \
        Tuple[Formula, Proof]:
    """Converts the given formula with uniquely named variables of the form
    ``'(``\ `Q1`\ `x1`\ ``[``\ `Q2`\ `x2`\ ``[``...\ `Qn`\ `xn`\ ``[``\ `inner_first`\ ``]``...\ ``]]``\ `*`\ `second`\ ``)'``
    to an equivalent formula of the form
    ``'``\ `Q'1`\ `x1`\ ``[``\ `Q'2`\ `x2`\ ``[``...\ `Q'n`\ `xn`\ ``[(``\ `inner_first`\ `*`\ `second`\ ``)]``...\ ``]]'``
    and proves the equivalence of these two formulas.

    Parameters:
        formula: formula with uniquely named variables to convert, whose root
            is a binary operator, i.e., which is of the form
            ``'(``\ `Q1`\ `x1`\ ``[``\ `Q2`\ `x2`\ ``[``...\ `Qn`\ `xn`\ ``[``\ `inner_first`\ ``]``...\ ``]]``\ `*`\ `second`\ ``)'``
            where `*` is a binary operator, `n`>=0, each `Qi` is a quantifier,
            each `xi` is a variable name, and `inner_first` does not start with
            a quantifier.

    Returns:
        A pair. The first element of the pair is a formula equivalent to the
        given formula, but of the form
        ``'``\ `Q'1`\ `x1`\ ``[``\ `Q'2`\ `x2`\ ``[``...\ `Q'n`\ `xn`\ ``[(``\ `inner_first`\ `*`\ `second`\ ``)]``...\ ``]]'``
        where each `Q'i` is a quantifier, and where the operator `*`, the `xi`
        variable names, `inner_first`, and `second` are the same as in the given
        formula. The second element of the pair is a proof of the equivalence of
        the given formula and the returned formula (i.e., a proof of
        `equivalence_of`\ ``(``\ `formula`\ ``,``\ `returned_formula`\ ``)``)
        via `~predicates.prover.Prover.AXIOMS` and
        `ADDITIONAL_QUANTIFICATION_AXIOMS`.

    Examples:
        >>> formula = Formula.parse('(Ax[Ey[R(x,y)]]&Ez[P(1,z)])')
        >>> returned_formula, proof = pull_out_quantifications_from_left_across_binary_operator(
        ...     formula)
        >>> returned_formula
        Ax[Ey[(R(x,y)&Ez[P(1,z)])]]
        >>> proof.is_valid()
        True
        >>> proof.conclusion == equivalence_of(formula, returned_formula)
        True
        >>> proof.assumptions == Prover.AXIOMS.union(
        ...     ADDITIONAL_QUANTIFICATION_AXIOMS)
        True
    """
    assert has_uniquely_named_variables(formula)
    assert is_binary(formula.root)
    # Task 11.7.1
    prover = Prover(Prover.AXIOMS.union(ADDITIONAL_QUANTIFICATION_AXIOMS))
    if not is_quantifier(formula.first.root):
        prover.add_tautology(equivalence_of(formula, formula))
        return (formula, prover.qed())
    predicate = formula.first.predicate
    new_binary_formula = Formula(formula.root, predicate, formula.second)
    new_first, proof = pull_out_quantifications_from_left_across_binary_operator(new_binary_formula)
    conclusion_of_proof = prover.add_proof(proof.conclusion, proof)

    rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[2]
    rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[14]
    new_quantifier = 'A'
    if formula.root == '&':
        if formula.first.root == 'E':
            new_quantifier = 'E'
            rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[3]
            rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[15]
    elif formula.root == '|':
        rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[6]
        rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[14]
        new_quantifier = 'A'
        if formula.first.root == 'E':
            new_quantifier = 'E'
            rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[7]
            rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[15]
    elif formula.root == '->':
        rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[10]
        rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[15]
        new_quantifier = 'E'
        if formula.first.root == 'E':
            new_quantifier = 'A'
            rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[11]
            rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[14]

    quantified_new_first = Formula(new_quantifier, formula.first.variable, new_first)
    new_formula = Formula(new_quantifier, formula.first.variable, new_binary_formula)
    equivalence1 = equivalence_of(new_formula, quantified_new_first)
    instantiated_assumption1 = Formula('->', proof.conclusion, equivalence1)
    instantiation_map1 = {'x': formula.first.variable,
                          'y': formula.first.variable,
                          'R': new_binary_formula.substitute(
                              {formula.first.variable: Term('_')}),
                          'Q': new_first.substitute(
                              {formula.first.variable: Term('_')})}
    i1 = prover.add_instantiated_assumption(instantiated_assumption1,
                                            rel_assumption2,
                                            instantiation_map1)
    i2 = prover.add_mp(equivalence1, conclusion_of_proof, i1)
    equivalence2 = equivalence_of(formula, new_formula)
    i3 = prover.add_instantiated_assumption(equivalence2, rel_assumption1,
                                            {'x': formula.first.variable,
                                             'R': predicate.substitute({formula.first.variable: Term('_')}),
                                             'Q': formula.second.substitute({formula.first.variable: Term('_')})})
    equivalence3 = equivalence_of(formula, quantified_new_first)
    prover.add_tautological_implication(equivalence3, {i2, i3})
    return (quantified_new_first, prover.qed())

    
def pull_out_quantifications_from_right_across_binary_operator(formula:
                                                               Formula) -> \
        Tuple[Formula, Proof]:
    """Converts the given formula with uniquely named variables of the form
    ``'(``\ `first`\ `*`\ `Q1`\ `x1`\ ``[``\ `Q2`\ `x2`\ ``[``...\ `Qn`\ `xn`\ ``[``\ `inner_second`\ ``]``...\ ``]])'``
    to an equivalent formula of the form
    ``'``\ `Q'1`\ `x1`\ ``[``\ `Q'2`\ `x2`\ ``[``...\ `Q'n`\ `xn`\ ``[(``\ `first`\ `*`\ `inner_second`\ ``)]``...\ ``]]'``
    and proves the equivalence of these two formulas.

    Parameters:
        formula: formula with uniquely named variables to convert, whose root
            is a binary operator, i.e., which is of the form
            ``'(``\ `first`\ `*`\ `Q1`\ `x1`\ ``[``\ `Q2`\ `x2`\ ``[``...\ `Qn`\ `xn`\ ``[``\ `inner_second`\ ``]``...\ ``]])'``
            where `*` is a binary operator, `n`>=0, each `Qi` is a quantifier,
            each `xi` is a variable name, and `inner_second` does not start with
            a quantifier.

    Returns:
        A pair. The first element of the pair is a formula equivalent to the
        given formula, but of the form
        ``'``\ `Q'1`\ `x1`\ ``[``\ `Q'2`\ `x2`\ ``[``...\ `Q'n`\ `xn`\ ``[(``\ `first`\ `*`\ `inner_second`\ ``)]``...\ ``]]'``
        where each `Q'i` is a quantifier, and where the operator `*`, the `xi`
        variable names, `first`, and `inner_second` are the same as in the given
        formula. The second element of the pair is a proof of the equivalence of
        the given formula and the returned formula (i.e., a proof of
        `equivalence_of`\ ``(``\ `formula`\ ``,``\ `returned_formula`\ ``)``)
        via `~predicates.prover.Prover.AXIOMS` and
        `ADDITIONAL_QUANTIFICATION_AXIOMS`.

    Examples:
        >>> formula = Formula.parse('(Ax[Ey[R(x,y)]]|Ez[P(1,z)])')
        >>> returned_formula, proof = pull_out_quantifications_from_right_across_binary_operator(
        ...     formula)
        >>> returned_formula
        Ez[(Ax[Ey[R(x,y)]]|P(1,z))]
        >>> proof.is_valid()
        True
        >>> proof.conclusion == equivalence_of(formula, returned_formula)
        True
        >>> proof.assumptions == Prover.AXIOMS.union(
        ...     ADDITIONAL_QUANTIFICATION_AXIOMS)
        True
    """
    assert has_uniquely_named_variables(formula)
    assert is_binary(formula.root)
    # Task 11.7.2
    prover = Prover(Prover.AXIOMS.union(ADDITIONAL_QUANTIFICATION_AXIOMS))
    if not is_quantifier(formula.second.root):
        prover.add_tautology(equivalence_of(formula, formula))
        return (formula, prover.qed())
    predicate = formula.second.predicate
    new_binary_formula = Formula(formula.root, formula.first, predicate)
    new_second, proof = pull_out_quantifications_from_right_across_binary_operator(new_binary_formula)
    conclusion_of_proof = prover.add_proof(proof.conclusion, proof)

    rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[4]
    rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[14]
    new_quantifier = 'A'
    if formula.root == '&':
        if formula.second.root == 'E':
            new_quantifier = 'E'
            rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[5]
            rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[15]
    elif formula.root == '|':
        rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[8]
        rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[14]
        new_quantifier = 'A'
        if formula.second.root == 'E':
            new_quantifier = 'E'
            rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[9]
            rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[15]
    elif formula.root == '->':
        rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[12]
        rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[14]
        new_quantifier = 'A'
        if formula.second.root == 'E':
            new_quantifier = 'E'
            rel_assumption1 = ADDITIONAL_QUANTIFICATION_AXIOMS[13]
            rel_assumption2 = ADDITIONAL_QUANTIFICATION_AXIOMS[15]

    quantified_new_second = Formula(new_quantifier, formula.second.variable,
                                   new_second)
    new_formula = Formula(new_quantifier, formula.second.variable,
                          new_binary_formula)
    equivalence1 = equivalence_of(new_formula, quantified_new_second)
    instantiated_assumption1 = Formula('->', proof.conclusion, equivalence1)
    instantiation_map1 = {'x': formula.second.variable,
                          'y': formula.second.variable,
                          'R': new_binary_formula.substitute(
                              {formula.second.variable: Term('_')}),
                          'Q': new_second.substitute(
                              {formula.second.variable: Term('_')})}
    i1 = prover.add_instantiated_assumption(instantiated_assumption1,
                                            rel_assumption2,
                                            instantiation_map1)
    i2 = prover.add_mp(equivalence1, conclusion_of_proof, i1)
    equivalence2 = equivalence_of(formula, new_formula)
    i3 = prover.add_instantiated_assumption(equivalence2, rel_assumption1,
                                            {'x': formula.second.variable,
                                             'R': predicate.substitute({formula.second.variable: Term('_')}),
                                             'Q': formula.first.substitute({formula.second.variable: Term('_')})})
    equivalence3 = equivalence_of(formula, quantified_new_second)
    prover.add_tautological_implication(equivalence3, {i2, i3})
    return (quantified_new_second, prover.qed())

def pull_out_quantifications_across_binary_operator(formula: Formula) -> \
        Tuple[Formula, Proof]:
    """Converts the given formula with uniquely named variables of the form
    ``'(``\ `Q1`\ `x1`\ ``[``\ `Q2`\ `x2`\ ``[``...\ `Qn`\ `xn`\ ``[``\ `inner_first`\ ``]``...\ ``]]``\ `*`\ `P1`\ `y1`\ ``[``\ `P2`\ `y2`\ ``[``...\ `Pm`\ `ym`\ ``[``\ `inner_second`\ ``]``...\ ``]])'``
    to an equivalent formula of the form
    ``'``\ `Q'1`\ `x1`\ ``[``\ `Q'2`\ `x2`\ ``[``...\ `Q'n`\ `xn`\ ``[``\ `P'1`\ `y1`\ ``[``\ `P'2`\ `y2`\ ``[``...\ `P'm`\ `ym`\ ``[(``\ `inner_first`\ `*`\ `inner_second`\ ``)]``...\ ``]]]``...\ ``]]'``
    and proves the equivalence of these two formulas.

    Parameters:
        formula: formula with uniquely named variables to convert, whose root
            is a binary operator, i.e., which is of the form
            ``'(``\ `Q1`\ `x1`\ ``[``\ `Q2`\ `x2`\ ``[``...\ `Qn`\ `xn`\ ``[``\ `inner_first`\ ``]``...\ ``]]``\ `*`\ `P1`\ `y1`\ ``[``\ `P2`\ `y2`\ ``[``...\ `Pm`\ `ym`\ ``[``\ `inner_second`\ ``]``...\ ``]])'``
            where `*` is a binary operator, `n`>=0, `m`>=0, each `Qi` and `Pi`
            is a quantifier, each `xi` and `yi` is a variable name, and neither
            `inner_first` nor `inner_second` starts with a quantifier.

    Returns:
        A pair. The first element of the pair is a formula equivalent to the
        given formula, but of the form
        ``'``\ `Q'1`\ `x1`\ ``[``\ `Q'2`\ `x2`\ ``[``...\ `Q'n`\ `xn`\ ``[``\ `P'1`\ `y1`\ ``[``\ `P'2`\ `y2`\ ``[``...\ `P'm`\ `ym`\ ``[(``\ `inner_first`\ `*`\ `inner_second`\ ``)]``...\ ``]]]``...\ ``]]'``
        where each `Q'i` and `P'i` is a quantifier, and where the operator `*`,
        the `xi` and `yi` variable names, `inner_first`, and `inner_second` are
        the same as in the given formula. The second element of the pair is a
        proof of the equivalence of the given formula and the returned formula
        (i.e., a proof of
        `equivalence_of`\ ``(``\ `formula`\ ``,``\ `returned_formula`\ ``)``)
        via `~predicates.prover.Prover.AXIOMS` and
        `ADDITIONAL_QUANTIFICATION_AXIOMS`.

    Examples:
        >>> formula = Formula.parse('(Ax[Ey[R(x,y)]]->Ez[P(1,z)])')
        >>> returned_formula, proof = pull_out_quantifications_across_binary_operator(
        ...     formula)
        >>> returned_formula
        Ex[Ay[Ez[(R(x,y)->P(1,z))]]]
        >>> proof.is_valid()
        True
        >>> proof.conclusion == equivalence_of(formula, returned_formula)
        True
        >>> proof.assumptions == Prover.AXIOMS.union(
        ...     ADDITIONAL_QUANTIFICATION_AXIOMS)
        True
    """
    assert has_uniquely_named_variables(formula)
    assert is_binary(formula.root)
    # Task 11.8
    prover = Prover(Prover.AXIOMS.union(ADDITIONAL_QUANTIFICATION_AXIOMS))
    new_left, left_proof = pull_out_quantifications_from_left_across_binary_operator(formula)
    rel_formula = new_left
    all_quantifiers_of_left = []
    all_variables_of_left = []
    while is_quantifier(rel_formula.root):
        all_quantifiers_of_left.append(rel_formula.root)
        all_variables_of_left.append(rel_formula.variable)
        rel_formula = rel_formula.predicate
    new_right, right_proof = pull_out_quantifications_from_right_across_binary_operator(rel_formula)
    right_conclusion = prover.add_proof(right_proof.conclusion, right_proof)
    rel_equivalence = right_proof.conclusion
    for i in reversed(range(len(all_quantifiers_of_left))):
        rel_quantifier = all_quantifiers_of_left[i]
        rel_var = all_variables_of_left[i]
        equivalence = equivalence_of(Formula(rel_quantifier, rel_var, rel_equivalence.first.first),
                                     Formula(rel_quantifier, rel_var, rel_equivalence.first.second))
        instantiated_assumption = Formula('->', rel_equivalence, equivalence)
        rel_assumption = ADDITIONAL_QUANTIFICATION_AXIOMS[14]
        if rel_quantifier == 'E':
            rel_assumption = ADDITIONAL_QUANTIFICATION_AXIOMS[15]
        i = prover.add_instantiated_assumption(instantiated_assumption, rel_assumption,
                                               {'x': rel_var, 'y': rel_var,
                                                'R': rel_equivalence.first.first.substitute({rel_var: Term('_')}),
                                                'Q': rel_equivalence.first.second.substitute({rel_var: Term('_')})})
        right_conclusion = prover.add_mp(equivalence, right_conclusion, i)
        rel_equivalence = equivalence
    left_conclusion = prover.add_proof(left_proof.conclusion, left_proof)
    new_equivalence = equivalence_of(formula, rel_equivalence.first.second)
    prover.add_tautological_implication(new_equivalence, {left_conclusion, right_conclusion})
    return (new_equivalence.first.second, prover.qed())


def to_prenex_normal_form_from_uniquely_named_variables(formula: Formula) -> \
        Tuple[Formula, Proof]:
    """Converts the given formula with uniquely named variables to an equivalent
    formula in prenex normal form, and proves the equivalence of these two
    formulas.

    Parameters:
        formula: formula with uniquely named variables to convert.

    Returns:
        A pair. The first element of the pair is a formula equivalent to the
        given formula, but in prenex normal form. The second element of the pair
        is a proof of the equivalence of the given formula and the returned
        formula (i.e., a proof of
        `equivalence_of`\ ``(``\ `formula`\ ``,``\ `returned_formula`\ ``)``)
        via `~predicates.prover.Prover.AXIOMS` and
        `ADDITIONAL_QUANTIFICATION_AXIOMS`.

    Examples:
        >>> formula = Formula.parse('(~(Ax[Ey[R(x,y)]]->Ez[P(1,z)])|S(w))')
        >>> returned_formula, proof = to_prenex_normal_form_from_uniquely_named_variables(
        ...     formula)
        >>> returned_formula
        Ax[Ey[Az[(~(R(x,y)->P(1,z))|S(w))]]]
        >>> proof.is_valid()
        True
        >>> proof.conclusion == equivalence_of(formula, returned_formula)
        True
        >>> proof.assumptions == Prover.AXIOMS.union(
        ...     ADDITIONAL_QUANTIFICATION_AXIOMS)
        True
    """
    assert has_uniquely_named_variables(formula)
    # Task 11.9

def to_prenex_normal_form(formula: Formula) -> Tuple[Formula, Proof]:
    """Converts the given formula to an equivalent formula in prenex normal
    form, and proves the equivalence of these two formulas.

    Parameters:
        formula: formula to convert, which contains no variable names starting
            with ``z``.

    Returns:
        A pair. The first element of the pair is a formula equivalent to the
        given formula, but in prenex normal form. The second element of the pair
        is a proof of the equivalence of the given formula and the returned
        formula (i.e., a proof of
        `equivalence_of`\ ``(``\ `formula`\ ``,``\ `returned_formula`\ ``)``)
        via `~predicates.prover.Prover.AXIOMS` and
        `ADDITIONAL_QUANTIFICATION_AXIOMS`.
    """
    # Task 11.10
