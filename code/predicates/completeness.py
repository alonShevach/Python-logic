# (c) This file is part of the course
# Mathematical Logic through Programming
# by Gonczarowski and Nisan.
# File name: predicates/completeness.py

from typing import AbstractSet, Container, Set, Union

from logic_utils import fresh_constant_name_generator

from predicates.syntax import *
from predicates.semantics import *
from predicates.proofs import *
from predicates.prover import *
from predicates.deduction import *
from predicates.prenex import *
from itertools import product


def get_constants(formulas: AbstractSet[Formula]) -> Set[str]:
    """Finds all constant names in the given formulas.

    Parameters:
        formulas: formulas to find all constant names in.

    Returns:
        A set of all constant names used in one or more of the given formulas.
    """
    constants = set()
    for formula in formulas:
        constants.update(formula.constants())
    return constants


def is_closed(sentences: AbstractSet[Formula]) -> bool:
    """Checks whether the given set of prenex-normal-form sentences is closed.

    Parameters:
        sentences: set of prenex-normal-form sentences to check.

    Returns:
        ``True`` if the given set of sentences is primitively, universally, and
        existentially closed, ``False`` otherwise.
    """
    for sentence in sentences:
        assert is_in_prenex_normal_form(sentence) and \
               len(sentence.free_variables()) == 0
    return is_primitively_closed(sentences) and \
           is_universally_closed(sentences) and \
           is_existentially_closed(sentences)


def is_primitively_closed(sentences: AbstractSet[Formula]) -> bool:
    """Checks whether the given set of prenex-normal-form sentences is
    primitively closed.

    Parameters:
        sentences: set of prenex-normal-form sentences to check.

    Returns:
        ``True`` if for every n-ary relation name from the given sentences, and
        for every n (not necessarily distinct) constant names from the given
        sentences, either the invocation of this relation name over these
        constant names (in order), or the negation of this invocation, is one of
        the given sentences, ``False`` otherwise.
    """
    for sentence in sentences:
        assert is_in_prenex_normal_form(sentence) and \
               len(sentence.free_variables()) == 0
    # Task 12.1.1
    const_set = set()
    relation_set = set()
    str_sents = set()
    for sentence in sentences:
        const_set = const_set.union(sentence.constants())
        relation_set = relation_set.union(sentence.relations())
        if not is_quantifier(sentence.root):
            str_sents.add(sentence.__repr__())
    const_set = set(int(const) for const in const_set)
    for relation, arg_num in relation_set:
        for args in product(const_set, repeat=arg_num):
            if (relation + str(args)).replace(" ", "") in str_sents:
                str_sents.remove((relation + str(args)).replace(" ", ""))
            elif "~" + (relation + str(args)).replace(" ", "") in str_sents:
                str_sents.remove("~" + (relation + str(args)).replace(" ", ""))
            else:
                return False
    return True


def is_universally_closed(sentences: AbstractSet[Formula]) -> bool:
    """Checks whether the given set of prenex-normal-form sentences is
    universally closed.

    Parameters:
        sentences: set of prenex-normal-form sentences to check.

    Returns:
        ``True`` if for every universally quantified sentence of the given
        sentences, and for every constant name from the given sentences, the
        predicate of this quantified sentence, with every free occurrence of the
        universal quantification variable replaced with this constant name, is
        one of the given sentences, ``False`` otherwise.
    """
    for sentence in sentences:
        assert is_in_prenex_normal_form(sentence) and \
               len(sentence.free_variables()) == 0
    # Task 12.1.2
    return check_for_exist_and_quant(sentences, False)


def check_for_exist_and_quant(sentences, is_exist):
    """
    Checks whether the given set of prenex-normal-form sentences is
    universally closed.
    Checks for E and A quantifier
    :param sentences: set of prenex-normal-form sentences to check.
    :param is_exist: True if it is a E quantifier, false if A
    :return: ``True`` if for every universally quantified sentence of the given
        sentences, and for every constant name from the given sentences, the
        predicate of this quantified sentence, with every free occurrence of the
        universal quantification variable replaced with this constant name, is
        one of the given sentences, ``False`` otherwise.
    """
    const_set = set()
    sent_set = set()
    for sentence in sentences:
        const_set = const_set.union(sentence.constants())
        sent_set.add(sentence.__repr__())
    for sentence in sentences:
        if not is_quantifier(sentence.root) or (sentence.root == "E" and not is_exist) or (
                is_exist and sentence.root == "A"):
            continue
        var = sentence.variable
        found_inst = False
        for const in const_set:
            form = sentence.predicate.substitute({var: Term(const)})
            found_inst = form.__repr__() in sent_set
            if not found_inst and not is_exist:
                return False
            if found_inst and is_exist:
                break
        if not found_inst:
            return False
    return True


def is_existentially_closed(sentences: AbstractSet[Formula]) -> bool:
    """Checks whether the given set of prenex-normal-form sentences is
    existentially closed.

    Parameters:
        sentences: set of prenex-normal-form sentences to check.

    Returns:
        ``True`` if for every existentially quantified sentence of the given
        sentences there exists a constant name such that the predicate of this
        quantified sentence, with every free occurrence of the existential
        quantification variable replaced with this constant name, is one of the
        given sentences, ``False`` otherwise.
    """
    for sentence in sentences:
        assert is_in_prenex_normal_form(sentence) and \
               len(sentence.free_variables()) == 0
    # Task 12.1.3
    return check_for_exist_and_quant(sentences, True)


def find_unsatisfied_quantifier_free_sentence(sentences: Container[Formula],
                                              model: Model[str],
                                              unsatisfied: Formula) -> Formula:
    """
    Given a closed set of prenex-normal-form sentences, given a model whose
    universe is the set of all constant names from the given sentences, and
    given a sentence from the given set that the given model does not satisfy,
    finds a quantifier-free sentence from the given set that the given model
    does not satisfy.
    
    Parameters:
        sentences: closed set of prenex-normal-form sentences, which is only to
            be accessed using containment queries, i.e., using the ``in``
            operator as in:

            >>> if sentence in sentences:
            ...     print('contained!')

        model: model for all element names from the given sentences, whose
            universe is `get_constants`\ ``(``\ `sentences`\ ``)``.
        unsatisfied: sentence (which possibly contains quantifiers) from the
            given sentences that is not satisfied by the given model.

    Returns:
        A quantifier-free sentence from the given sentences that is not
        satisfied by the given model.
    """
    # We assume that every sentence in sentences is of type formula, is in
    # prenex normal form, and has no free variables, and furthermore that the
    # set of constants that appear somewhere in sentences is model.universe;
    # but we cannot assert these since we cannot iterate over sentences.
    for constant in model.universe:
        assert is_constant(constant)
    assert is_in_prenex_normal_form(unsatisfied)
    assert len(unsatisfied.free_variables()) == 0
    assert unsatisfied in sentences
    assert not model.evaluate_formula(unsatisfied)
    # Task 12.2

    return find_unsatisfied_helper(model, sentences, unsatisfied)


def find_unsatisfied_helper(model, sentences, unsatisfied):
    if not is_quantifier(unsatisfied.root):
        if unsatisfied in sentences and not model.evaluate_formula(unsatisfied):
            return unsatisfied
        return False
    for const in model.universe:
        form_or_false = find_unsatisfied_helper(model, sentences,
                                                unsatisfied.predicate.substitute({unsatisfied.variable: Term(const)},
                                                                                 {}))
        if form_or_false is not False:
            return form_or_false
    return False


def get_primitives(quantifier_free: Formula) -> Set[Formula]:
    """Finds all primitive subformulas of the given quantifier-free formula.

    Parameters:
        quantifier_free: quantifier-free formula whose subformulas are to
            be searched.

    Returns:
        The primitive subformulas (i.e., relation invocations) of the given
        quantifier-free formula.

    Examples:
        The primitive subformulas of ``'(R(c1,d)|~(Q(c1)->~R(c2,a)))'`` are
        ``'R(c1,d)'``, ``'Q(c1)'``, and ``'R(c2,a)'``.
    """
    assert is_quantifier_free(quantifier_free)
    # Task 12.3.1
    new_set = set()
    if is_binary(quantifier_free.root):
        first_set = get_primitives(quantifier_free.first)
        second_set = get_primitives(quantifier_free.second)
        return first_set.union(second_set)
    elif is_unary(quantifier_free.root):
        return get_primitives(quantifier_free.first)
    new_set.add(quantifier_free)
    return new_set


def model_or_inconsistency(sentences: AbstractSet[Formula]) -> \
        Union[Model[str], Proof]:
    """Either finds a model in which the given closed set of prenex-normal-form
    sentences holds, or proves a contradiction from these sentences.

    Parameters:
        sentences: closed set of prenex-normal-form sentences to either find a
            model for or prove a contradiction from.

    Returns:
        A model in which all of the given sentences hold if such exists,
        otherwise a valid proof of  a contradiction from the given formulas via
        `~predicates.prover.Prover.AXIOMS`.
    """
    assert is_closed(sentences)
    # Task 12.3.2
    universe = get_constants(sentences)
    const_meaning = dict()
    for const in universe:
        const_meaning[const] = const
    relation_meaning = dict()
    for sentence in sentences:
        if is_relation(sentence.root):
            if sentence.root not in relation_meaning:
                relation_meaning[sentence.root] = set()
            args = tuple(str(arg) for arg in sentence.arguments)
            relation_meaning[sentence.root].add(args)
    model = Model(universe, const_meaning, relation_meaning)
    model_fits = True
    unjust_sentence = None
    for sentence in sentences:
        if not model.evaluate_formula(sentence):
            model_fits = False
            unjust_sentence = sentence
    if model_fits:
        return model
    prover = Prover(Prover.AXIOMS.union(sentences))
    cont_sentence = find_unsatisfied_quantifier_free_sentence(sentences, model, unjust_sentence)
    cont_line = prover.add_assumption(cont_sentence)
    primitive_set = get_primitives(cont_sentence)
    pos_set = set()
    pos_set.add(cont_line)
    cont_form = None
    for form in primitive_set:
        if form in sentences:
            form_num = prover.add_assumption(form)
            pos_set.add(form_num)
            cont_form = form
        if Formula("~", form) in sentences:
            form_num = prover.add_assumption(Formula("~", form))
            pos_set.add(form_num)
            cont_form = form
    prover.add_tautological_implication(Formula("&", cont_form, Formula("~", cont_form)), pos_set)
    return prover.qed()


def combine_contradictions(proof_from_affirmation: Proof,
                           proof_from_negation: Proof) -> Proof:
    """Combines the given two proofs of contradictions, both from the same
    assumptions/axioms except that the latter has an extra assumption that is
    the negation of an extra assumption that the former has, into a single proof
    of a contradiction from only the common assumptions/axioms.

    Parameters:
        proof_from_affirmation: valid proof of a contradiction from one or more
            assumptions/axioms that are all sentences and that include
            `~predicates.prover.Prover.AXIOMS`.
        proof_from_negation: valid proof of a contradiction from the same
            assumptions/axioms of `proof_from_affirmation`, but with one
            simple assumption `assumption` replaced with its negation
            ``'~``\ `assumption` ``'``.

    Returns:
        A valid proof of a contradiction from only the assumptions/axioms common
        to the given proofs (i.e., without `assumption` or its negation).
    """
    assert proof_from_affirmation.is_valid()
    assert proof_from_negation.is_valid()
    common_assumptions = proof_from_affirmation.assumptions.intersection(
        proof_from_negation.assumptions)
    assert len(common_assumptions) == \
           len(proof_from_affirmation.assumptions) - 1
    assert len(common_assumptions) == len(proof_from_negation.assumptions) - 1
    affirmed_assumption = list(
        proof_from_affirmation.assumptions.difference(common_assumptions))[0]
    negated_assumption = list(
        proof_from_negation.assumptions.difference(common_assumptions))[0]
    assert len(affirmed_assumption.templates) == 0
    assert len(negated_assumption.templates) == 0
    assert negated_assumption.formula == \
           Formula('~', affirmed_assumption.formula)
    assert proof_from_affirmation.assumptions.issuperset(Prover.AXIOMS)
    assert proof_from_negation.assumptions.issuperset(Prover.AXIOMS)
    for assumption in common_assumptions.union({affirmed_assumption,
                                                negated_assumption}):
        assert len(assumption.formula.free_variables()) == 0
    # Task 12.4
    neg_assump = proof_by_way_of_contradiction(proof_from_affirmation, affirmed_assumption.formula)
    pos_assump = proof_by_way_of_contradiction(proof_from_negation, negated_assumption.formula)
    prover = Prover(common_assumptions)
    neg_line = prover.add_proof(negated_assumption.formula, neg_assump)
    neg_neg_form = Formula("~", negated_assumption.formula)
    neg_neg_line = prover.add_proof(neg_neg_form, pos_assump)
    prover.add_tautological_implication(Formula("&", neg_neg_form, negated_assumption.formula),
                                        {neg_line, neg_neg_line})
    return prover.qed()


def eliminate_universal_instantiation_assumption(proof: Proof, constant: str,
                                                 instantiation: Formula,
                                                 universal: Formula) -> Proof:
    """Converts the given proof of a contradiction, whose assumptions/axioms
    include `universal` and `instantiation`, where the latter is a universal
    instantiation of the former, to a proof of a contradiction from the same
    assumptions without `instantiation`.

    Parameters:
        proof: valid proof of a contradiction from one or more
            assumptions/axioms that are all sentences and that include
            `~predicates.prover.Prover.AXIOMS`.
        universal: assumption of the given proof that is universally quantified.
        instantiation: assumption of the given proof that is obtained from the
            predicate of `universal` by replacing all free occurrences of the
            universal quantification variable by some constant.

    Returns:
        A valid proof of a contradiction from the assumptions/axioms of the
        proof except `instantiation`.
    """
    assert proof.is_valid()
    assert is_constant(constant)
    assert Schema(instantiation) in proof.assumptions
    assert Schema(universal) in proof.assumptions
    assert universal.root == 'A'
    assert instantiation == \
           universal.predicate.substitute({universal.variable: Term(constant)})
    for assumption in proof.assumptions:
        assert len(assumption.formula.free_variables()) == 0
    # Task 12.5
    assumps = set(proof.assumptions) - {Schema(instantiation)}
    prover = Prover(assumps)
    not_inst_proof = proof_by_way_of_contradiction(proof, instantiation)
    univ_line = prover.add_assumption(universal)
    ui_line = prover.add_universal_instantiation(instantiation, univ_line, Term(constant))
    not_inst = Formula("~", instantiation)
    not_inst_line = prover.add_proof(not_inst, not_inst_proof)
    prover.add_tautological_implication(Formula("&", instantiation, not_inst), {ui_line, not_inst_line})
    return prover.qed()


def universal_closure_step(sentences: AbstractSet[Formula]) -> Set[Formula]:
    """Augments the given sentences with all universal instantiations of each
    universally quantified sentence from these sentences, with respect to all
    constant names from these sentences.

    Parameters:
        sentences: prenex-normal-form sentences to augment with their universal
            instantiations.

    Returns:
        A set of all of the given sentences, and in addition any formula that
        can be obtained replacing in the predicate of any universally quantified
        sentence from the given sentences, all occurrences of the quantification
        variable with some constant from the given sentences.
    """
    for sentence in sentences:
        assert is_in_prenex_normal_form(sentence) and \
               len(sentence.free_variables()) == 0
    # Task 12.6
    consts_set = get_constants(sentences)
    new_set = set(sentences)
    for sentence in sentences:
        if sentence.root == "A":
            for const in consts_set:
                new_set.add(sentence.predicate.substitute({sentence.variable: Term(const)}))
    return new_set


def replace_constant(proof: Proof, constant: str, variable: str = 'zz') -> \
        Proof:
    """Replaces all occurrences of the given constant in the given proof with
    the given variable.

    Parameters:
        proof: a valid proof.
        constant: a constant name that does not appear as a template constant
            name in any of the assumptions of the given proof.
        variable: a variable name that does not appear anywhere in given the
            proof or in its assumptions.

    Returns:
        A valid proof where every occurrence of the given constant name in the
        given proof and in its assumptions is replaced with the given variable
        name.
    """
    assert proof.is_valid()
    assert is_constant(constant)
    assert is_variable(variable)
    for assumption in proof.assumptions:
        assert constant not in assumption.templates
        assert variable not in assumption.formula.variables()
    for line in proof.lines:
        assert variable not in line.formula.variables()
    # Task 12.7.1
    sub_map = {constant: Term(variable)}
    new_assumptions = set()
    new_lines = []
    for assumption in proof.assumptions:
        replaced_formula = assumption.formula.substitute(sub_map)
        new_templates = set(assumption.templates)
        # if constant in new_templates:
        #     new_templates.remove(constant)
        #     new_templates.add(variable)
        new_assumptions.add(Schema(replaced_formula, new_templates))
    new_conclusion = proof.conclusion.substitute(sub_map)
    for line in proof.lines:
        if isinstance(line, Proof.MPLine):
            new_formula = line.formula.substitute(sub_map)
            new_lines.append(Proof.MPLine(new_formula, line.antecedent_line_number, line.conditional_line_number))
        elif isinstance(line, Proof.AssumptionLine):
            new_formula = line.formula.substitute(sub_map)
            new_instantiation_map = dict(line.instantiation_map)
            for key in new_instantiation_map:
                # if new_instantiation_map[key] == Term(constant):
                #     new_instantiation_map[key] = Term(variable)
                if isinstance(new_instantiation_map[key], Formula) or isinstance(new_instantiation_map[key], Term):
                    new_instantiation_map[key] = new_instantiation_map[key].substitute(sub_map)
            new_line_assumptions = Schema(line.assumption.formula.substitute(sub_map), line.assumption.templates)
            new_lines.append(Proof.AssumptionLine(new_formula, new_line_assumptions, new_instantiation_map))
        elif isinstance(line, Proof.UGLine):
            new_formula = line.formula.substitute(sub_map)
            new_lines.append(Proof.UGLine(new_formula, line.predicate_line_number))
        elif isinstance(line, Proof.TautologyLine):
            new_formula = line.formula.substitute(sub_map)
            new_lines.append(Proof.TautologyLine(new_formula))
    return Proof(new_assumptions, new_conclusion, new_lines)


def eliminate_existential_witness_assumption(proof: Proof, constant: str,
                                             witness: Formula,
                                             existential: Formula) -> Proof:
    """Converts the given proof of a contradiction, whose assumptions/axioms
    include `existential` and `witness`, where the latter is an existential
    witness of the former, to a proof of a contradiction from the same
    assumptions without `witness`.

    Parameters:
        proof: valid proof of a contradiction from one or more
            assumptions/axioms that are all sentences and that include
            `~predicates.prover.Prover.AXIOMS`.
        existential: assumption of the given proof that is existentially
            quantified.
        witness: assumption of the given proof that is obtained from the
            predicate of `existential` by replacing all free occurrences of the
            existential quantification variable by some constant that does not
            appear in any assumption of the given proof except for this
            assumption.

    Returns:
        A valid proof of a contradiction from the assumptions/axioms of the
        proof except `witness`.
    """
    assert proof.is_valid()
    assert is_constant(constant)
    assert Schema(witness) in proof.assumptions
    assert Schema(existential) in proof.assumptions
    assert existential.root == 'E'
    assert witness == \
           existential.predicate.substitute(
               {existential.variable: Term(constant)})
    for assumption in proof.assumptions:
        assert len(assumption.formula.free_variables()) == 0
    for assumption in proof.assumptions.difference({Schema(witness)}):
        assert constant not in assumption.formula.constants()
    # Task 12.7.2
    new_proof = replace_constant(proof, constant)
    replaced_witness = witness.substitute({constant: Term('zz')})
    proof_by_contradiction_without_witness = proof_by_way_of_contradiction(new_proof, replaced_witness)
    prover = Prover(proof_by_contradiction_without_witness.assumptions)
    conc_of_proof_by_contr = proof_by_contradiction_without_witness.conclusion
    new_conclusion = Formula('&', witness, Formula('~', witness))
    mp_conc = Formula('->', conc_of_proof_by_contr.first, new_conclusion)
    step1 = prover.add_proof(proof_by_contradiction_without_witness.conclusion, proof_by_contradiction_without_witness)
    step2 = prover.add_tautology(Formula('->', conc_of_proof_by_contr, mp_conc))
    step3 = prover.add_mp(mp_conc, step1, step2)
    ins_map = {'zz': Term(existential.variable)}
    step4 = prover.add_free_instantiation(mp_conc.substitute(ins_map), step3, ins_map)
    step5 = prover.add_assumption(existential)
    step6 = prover.add_existential_derivation(new_conclusion, step5, step4)
    return prover.qed()

def existential_closure_step(sentences: AbstractSet[Formula]) -> Set[Formula]:
    """Augments the given sentences with an existential witness that uses a new
    constant name, for each existentially quantified sentences from these
    sentences for which an existential witness is missing.

    Parameters:
        sentences: prenex-normal-form sentences to augment with any missing
            existential witnesses.

    Returns:
        A set of all of the given sentences, and in addition for every
        existentially quantified sentence from the given sentences, a formula
        obtained from the predicate of that quantified sentence by replacing all
        occurrences of the quantification variable with a new constant name
        obtained by calling
        `next`\ ``(``\ `~logic_utils.fresh_constant_name_generator`\ ``)``.
    """
    for sentence in sentences:
        assert is_in_prenex_normal_form(sentence) and \
               len(sentence.free_variables()) == 0
    # Task 12.8
    new_set = set(sentences)
    new_constant = next(fresh_constant_name_generator)
    for formula in sentences:
        if is_quantifier(formula.root):
            sub_formula = formula.predicate.substitute({formula.variable: Term(new_constant)})
            if not sub_formula in sentences:
                new_set.add(sub_formula)
            new_constant = next(fresh_constant_name_generator)
    return new_set
