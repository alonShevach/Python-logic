# (c) This file is part of the course
# Mathematical Logic through Programming
# by Gonczarowski and Nisan.
# File name: predicates/functions.py

"""Syntactic conversion of first-order formulas to not use functions and
equality."""

from typing import AbstractSet, List, Set

from logic_utils import fresh_variable_name_generator

from predicates.syntax import *
from predicates.semantics import *


def function_name_to_relation_name(function: str) -> str:
    """Converts the given function name to a canonically corresponding relation
    name.

    Parameters:
        function: function name to convert.

    Returns:
        A relation name that is the same as the given function name, except that
        its first letter is capitalized.
    """
    assert is_function(function)
    return function[0].upper() + function[1:]


def relation_name_to_function_name(relation: str) -> str:
    """Converts the given relation name to a canonically corresponding function
    name.

    Parameters:
        relation: relation name to convert.

    Returns:
        A function name `function` such that
        `function_name_to_relation_name`\ ``(``\ `function`\ ``)`` is the given
        relation name.
    """
    assert is_relation(relation)
    return relation[0].lower() + relation[1:]


def replace_functions_with_relations_in_model(model: Model[T]) -> Model[T]:
    """Converts the given model to a canonically corresponding model without any
    function meanings, replacing each function meaning with a canonically
    corresponding relation meaning.

    Parameters:
        model: model to convert, such that there exist no canonically
            corresponding function name and relation name that both have
            meanings in this model.

    Return:
        A model obtained from the given model by replacing every function
        meaning of a function name with a relation meaning of the canonically
        corresponding relation name, such that the relation meaning contains
        any tuple ``(``\ `x1`\ ``,``...\ ``,``\ `xn`\ ``)``  if and only if `x1`
        is the output of the function meaning for the arguments
        ``(``\ `x2`\ ``,``...\ ``,``\ `xn`\ ``)``.
    """
    for function in model.function_meanings:
        assert function_name_to_relation_name(function) not in \
               model.relation_meanings
    # Task 8.1
    new_relation_meaning = dict(model.relation_meanings)
    for function, args in model.function_meanings.items():
        temp = set()
        for k, v in args.items():
            temp.add(tuple(v) + k)
        new_relation_meaning[function_name_to_relation_name(function)] = tuple(temp)
    return Model(model.universe, model.constant_meanings, new_relation_meaning, dict())


def replace_relations_with_functions_in_model(model: Model[T],
                                              original_functions:
                                              AbstractSet[str]) -> \
        Union[Model[T], None]:
    """Converts the given model with no function meanings to a canonically
    corresponding model with meanings for the given function names, having each
    new function meaning replace a canonically corresponding relation meaning.

    Parameters:
        model: model to convert, that contains no function meanings.
        original_functions: function names for the model to convert to,
            such that no relation name that canonically corresponds to any of
            these function names has a meaning in the given model.

    Returns:
        A model `model` with the given function names such that
        `replace_functions_with_relations_in_model`\ ``(``\ `model`\ ``)``
        is the given model, or ``None`` if no such model exists.
    """
    for function in original_functions:
        assert is_function(function)
        assert function not in model.function_meanings
        assert function_name_to_relation_name(function) in \
               model.relation_meanings
    # Task 8.2
    new_function_meaning = dict()
    new_relation_meaning = dict()
    for relation, args in model.relation_meanings.items():
        temp = dict()
        if relation_name_to_function_name(relation) not in original_functions:
            new_relation_meaning[relation] = args
            continue
        for arg in args:
            if len(arg) != 0:
                value = arg[0]
                keys = tuple(arg[1:])
                if keys in temp.keys() and temp[keys] != value:
                    return None
                temp[keys] = value
        if len(temp) != len(model.universe) ** (model.relation_arities[relation] - 1):
            return None
        new_function_meaning[relation_name_to_function_name(relation)] = copy.copy(temp)
    return Model(model.universe, model.constant_meanings, new_relation_meaning,
                 new_function_meaning)


def compile_term(term: Term) -> List[Formula]:
    """Syntactically compiles the given term into a list of single-function
    invocation steps.

    Parameters:
        term: term to compile, whose root is a function invocation, and that
            contains no variable names starting with ``z``.

    Returns:
        A list of steps, each of which is a formula of the form
        ``'``\ `y`\ ``=``\ `f`\ ``(``\ `x1`\ ``,``...\ ``,``\ `xn`\ ``)'``,
        where `y` is a new variable name obtained by calling
        `next`\ ``(``\ `~logic_utils.fresh_variable_name_generator`\ ``)``, `f`
        is a function name, and each of the `x`\ `i` is either a constant name
        or a variable name. If `x`\ `i` is a new variable name, then it is also
        the left-hand side of a previous step, where all of the steps "leading
        up to" `x1` precede those "leading up" to `x2`, etc. If all the returned
        steps hold in any model, then the left-hand-side variable of the last
        returned step evaluates in that model to the value of the given term.
    """
    assert is_function(term.root)
    # Task 8.3
    func_lst = []
    compile_term_helper(term, func_lst)
    return func_lst


def compile_term_helper(term, func_lst):
    if is_constant(term.root) or is_variable(term.root):
        return term.root
    if len(term.arguments) == 1 and (
            is_variable(term.arguments[0].root) or is_constant(term.arguments[0].root)):
        var = next(fresh_variable_name_generator)
        func_lst.append(Formula("=", [Term(var), term]))
        return var
    args_lst = []
    for arg in term.arguments:
        args_lst.append(Term(compile_term_helper(arg, func_lst)))
    var = next(fresh_variable_name_generator)
    func_lst.append(Formula("=", [Term(var), Term(term.root, args_lst)]))
    return var


def replace_functions_with_relations_in_formula(formula: Formula) -> Formula:
    """Syntactically converts the given formula to a formula that does not
    contain any function invocations, and is "one-way equivalent" in the sense
    that the former holds in a model if and only if the latter holds in the
    canonically corresponding model with no function meanings.

    Parameters:
        formula: formula to convert, that contains no variable names starting
            with ``z``, and such that there exist no canonically corresponding
            function name and relation name that are both invoked in this
            formula.

    Returns:
        A formula such that the given formula holds in any model `model` if and
        only if the returned formula holds in
        `replace_function_with_relations_in_model`\ ``(``\ `model`\ ``)``.
    """
    assert len({function_name_to_relation_name(function) for
                function, arity in formula.functions()}.intersection(
        {relation for relation, arity in formula.relations()})) == 0
    for variable in formula.variables():
        assert variable[0] != 'z'
    # Task 8.4
    if is_variable(formula.root) or is_constant(formula.root):
        return formula
    if is_unary(formula.root):
        return Formula(formula.root, replace_functions_with_relations_in_formula(formula.first))
    if is_quantifier(formula.root):
        return Formula(formula.root, formula.variable,
                       replace_functions_with_relations_in_formula(formula.predicate))
    if is_binary(formula.root):
        return Formula(formula.root, replace_functions_with_relations_in_formula(formula.first),
                       replace_functions_with_relations_in_formula(formula.second))
    if is_relation(formula.root) or formula.root == "=":
        form_lst = []
        args_dict = dict()
        for term in formula.arguments:
            if not is_function(term.root):
                continue
            form_lst += compile_term(term)
        new_args = []
        for item in form_lst:
            func_name = item.arguments[1].root
            func_name = function_name_to_relation_name(func_name)
            new_relation = Formula(func_name,
                                   tuple([item.arguments[0]]) + item.arguments[1].arguments)
            new_args.append(new_relation)
            args_dict[item.arguments[1].__repr__()] = item.arguments[0].__repr__()
        new_form_args = []
        for term in formula.arguments:
            if not is_function(term.root):
                new_form_args.append(term)
                continue
            term_str = term.root + "("
            term_str = get_term_str(term, args_dict, term_str)
            if term_str[-1] == ",":
                term_str = term_str[:-1]
                term_str += ")"
            new_form_args.append(Term.parse(args_dict[term_str]))
        return replace_relation_or_equality(formula, form_lst, 0, new_args, new_form_args)


def get_term_str(term, args_dict, term_str):
    for arg in term.arguments:
        if not is_function(arg.root):
            term_str += arg.__repr__() + ","
        else:
            if arg.__repr__() in args_dict:
                term_str += args_dict[arg.__repr__()] + ","
            else:
                term_str = get_term_str(arg, args_dict, term_str)
    return term_str


def replace_relation_or_equality(formula, form_lst, i, new_args, new_form_args):
    if i == len(form_lst):
        return Formula(formula.root, new_form_args)
    return Formula("A", form_lst[i].arguments[0].__repr__(), Formula("->", new_args[i],
                                                                     replace_relation_or_equality(
                                                                         formula, form_lst, i + 1,
                                                                         new_args, new_form_args)))


def replace_functions_with_relations_in_formulas(formulas:
AbstractSet[Formula]) -> \
        Set[Formula]:
    """Syntactically converts the given set of formulas to a set of formulas
    that do not contain any function invocations, and is "two-way
    equivalent" in the sense that:

    1. The former holds in a model if and only if the latter holds in the
       canonically corresponding model with no function meanings.
    2. The latter holds in a model if and only if that model has a
       canonically corresponding model with meanings for the functions of the
       former, and the former holds in that model.

    Parameters:
        formulas: formulas to convert, that contain no variable names starting
            with ``z``, and such that there exist no canonically corresponding
            function name and relation name that are both invoked in these
            formulas.

    Returns:
        A set of formulas, one for each given formula as well as one additional
        formula for each relation name that replaces a function name from the
        given formulas, such that:

        1. The given formulas holds in a model `model` if and only if the
           returned formulas holds in
           `replace_functions_with_relations_in_model`\ ``(``\ `model`\ ``)``.
        2. The returned formulas holds in a model `model` if and only if
           `replace_relations_with_functions_in_model`\ ``(``\ `model`\ ``,``\ `original_functions`\ ``)``,
           where `original_functions` are all the function names in the given
           formulas, is a model and the given formulas hold in it.
    """
    assert len(set.union(*[{function_name_to_relation_name(function) for
                            function, arity in formula.functions()}
                           for formula in formulas]).intersection(
        set.union(*[{relation for relation, arity in
                     formula.relations()} for
                    formula in formulas]))) == 0
    for formula in formulas:
        for variable in formula.variables():
            assert variable[0] != 'z'
    # Task 8.5
    all_formulas = set()
    function_names = set()
    for formula in formulas:
        all_formulas.add(replace_functions_with_relations_in_formula(formula))
        function_names.update(formula.functions())
    for func in function_names:
        first_con_str = ''
        sec_con_str = ''
        var_names = []
        for i in range(func[1]):
            var_names.append('x' + str(i+1))
        for var in var_names:
            first_con_str += 'A' + var + '['
            sec_con_str += 'A' + var + '['
        first_con_str += 'Ez[' + function_name_to_relation_name(func[0]) + '(z,'
        sec_con_str += 'Az1[Az2[((' + function_name_to_relation_name(func[0]) + '(z1,'
        for var in var_names:
            first_con_str += var + ','
            sec_con_str += var + ','
        first_con_str = first_con_str[:-1] + ')' + ']'*(len(var_names)+1)
        sec_con_str = sec_con_str[:-1] + ')&' + function_name_to_relation_name(func[0]) + '(z2,'
        for var in var_names:
            sec_con_str += var + ','
        sec_con_str = sec_con_str[:-1] + '))->z1=z2)' + ']'*(len(var_names)+2)
        first_con = Formula.parse(first_con_str)
        sec_con = Formula.parse(sec_con_str)
        all_formulas.add(Formula('&', first_con, sec_con))
    return all_formulas


def replace_equality_with_SAME_in_formulas(formulas: AbstractSet[Formula]) -> \
        Set[Formula]:
    """Syntactically converts the given set of formulas to a canonically
    corresponding set of formulas that do not contain any equalities, consisting
    of the following formulas:

    1. A formula for each of the given formulas, where each equality is
       replaced with a matching invocation of the relation name ``'SAME'``.
    2. Formula(s) that ensure that in any model for the returned formulas,
       the meaning of the relation name ``'SAME'`` is reflexive, symmetric, and
       transitive.
    3. For each relation name from the given formulas, formula(s) that ensure
       that in any model for the returned formulas, the meaning of this
       relation name respects the meaning of the relation name ``'SAME'``.

    Parameters:
        formulas: formulas to convert, that contain no function names and do not
            contain the relation name ``'SAME'``.

    Returns:
        The converted set of formulas.
    """
    for formula in formulas:
        assert len(formula.functions()) == 0
        assert 'SAME' not in \
               {relation for relation, arity in formula.relations()}
    # Task 8.6
    all_formulas = set()
    all_relations_in_formulas = set()
    for formula in formulas:
        all_relations_in_formulas.update(formula.relations())
        if not '=' in formula.__repr__():
            all_formulas.add(formula)
        else:
            new_formula = replace_equality_with_SAME_in_one_formula(formula)
            all_formulas.add(new_formula)
    # reflexivity
    all_formulas.add(Formula.parse('Ax[SAME(x,x)]'))
    # symmetry
    all_formulas.add(Formula.parse('Ax[Ay[((SAME(x,y)->SAME(y,x))&(SAME(y,x)->SAME(x,y)))'))
    # transitivity
    all_formulas.add(Formula.parse('Ax[Ay[Az[((SAME(x,y)&SAME(y,z))->SAME(y,z))]]]'))
    for rel in all_relations_in_formulas:
        x_vars = []
        y_vars = []
        for i in range(rel[1]):
            x_vars.append(Term('x' + str(i+1)))
            y_vars.append(Term('y' + str(i+1)))
        all_same_relations = []
        for i in range(len(x_vars)):
            same_relation = Formula('SAME', (x_vars[i], y_vars[i]))
            all_same_relations.append(same_relation)
        deduced_formula = Formula('->', Formula(rel[0], x_vars), Formula(rel[0], y_vars))
        same_formula = all_same_relations[0]
        i = 1
        while i < len(all_same_relations):
            new_same_formula = Formula('&', same_formula, all_same_relations[i])
            same_formula = new_same_formula
            i += 1
        formula_without_quantifiers = Formula('->', same_formula, deduced_formula)
        rel_str = ''
        for var in x_vars + y_vars:
            rel_str += 'A' + var.__repr__() + '['
        rel_str += formula_without_quantifiers.__repr__() + ']'*len(x_vars)*2
        all_formulas.add(Formula.parse(rel_str))
    return all_formulas

def replace_equality_with_SAME_in_one_formula(formula: Formula) -> Formula:
    if is_variable(formula.root) or is_constant(formula.root) or is_relation(formula.root):
        return formula
    elif is_unary(formula.root):
        return Formula(formula.root, replace_equality_with_SAME_in_one_formula(formula.first))
    elif is_binary(formula.root):
        new_first = replace_equality_with_SAME_in_one_formula(formula.first)
        new_sec = replace_equality_with_SAME_in_one_formula(formula.second)
        return Formula(formula.root, new_first, new_sec)
    elif is_quantifier(formula.root):
        return Formula(formula.root, formula.variable, replace_equality_with_SAME_in_one_formula(formula.predicate))
    elif formula.root == '=':
        return Formula('SAME', formula.arguments)

def add_SAME_as_equality_in_model(model: Model[T]) -> Model[T]:
    """Adds a meaning for the relation name ``'SAME'`` in the given model, that
    canonically corresponds to equality in the given model.

    Parameters:
        model: model that has no meaning for the relation name ``'SAME'``, to
            add the meaning to.

    Return:
        A model obtained from the given model by adding a meaning for the
        relation name ``'SAME'``, that contains precisely all pairs
        ``(``\ `x`\ ``,``\ `x`\ ``)`` for every element `x` of the universe of
        the given model.
    """
    assert 'SAME' not in model.relation_meanings
    # Task 8.7
    new_model_relation = dict(model.relation_meanings)
    SAME_meaning = set()
    for element in model.universe:
        SAME_meaning.add((element, element))
    new_model_relation['SAME'] = SAME_meaning
    return Model(model.universe, model.constant_meanings, new_model_relation, model.function_meanings)


def make_equality_as_SAME_in_model(model: Model[T]) -> Model[T]:
    """Converts the given model to a model where equality coincides with the
    meaning of ``'SAME'`` in the given model, in the sense that any set of
    formulas holds in the returned model if and only if its canonically
    corresponding set of formulas that do not contain equality holds in the
    given model.

    Parameters:
        model: model to convert, that contains no function meanings, and
            contains a meaning for the relation name ``'SAME'`` that is
            reflexive, symmetric, transitive, and respected by the meanings
            of all other relation names.

    Returns:
        A model that is a model for any set `formulas` if and only if
        the given model is a model for
        `replace_equality_with_SAME`\ ``(``\ `formulas`\ ``)``. The universe of
        the returned model corresponds to the equivalence classes of the
        ``'SAME'`` relation in the given model.
    """
    assert 'SAME' in model.relation_meanings and \
           model.relation_arities['SAME'] == 2
    assert len(model.function_meanings) == 0
    # Task 8.8
    all_classes = dict()
    for pair in model.relation_meanings['SAME']:
        added_to_dict = False
        for key in all_classes.keys():
            if pair[0] in all_classes[key] or pair[1] in all_classes[key]:
                all_classes[key].add(pair[0])
                all_classes[key].add(pair[1])
                added_to_dict = True
        if not added_to_dict:
            all_classes[pair[0]] = {pair[0], pair[1]}
    new_universe = set(all_classes.keys())
    new_constant_meanings = dict(model.constant_meanings)
    for const in model.constant_meanings:
        for key in all_classes.keys():
            if model.constant_meanings[const] in all_classes[key]:
                new_constant_meanings[const] = key
    new_relation_meanings = dict()
    new_keys = list(model.relation_meanings.keys())
    new_keys.remove('SAME')
    for rel_name in new_keys:
        for rel in model.relation_meanings[rel_name]:
            new_rel = ()
            for var in rel:
                if var in new_universe:
                    new_rel += (var,)
                else:
                    for key in all_classes:
                        if var in all_classes[key]:
                            new_rel += (key,)
            if rel_name not in new_relation_meanings:
                new_relation_meanings[rel_name] = {new_rel}
            else:
                new_relation_meanings[rel_name].add(new_rel)
    return Model(new_universe,new_constant_meanings, new_relation_meanings, model.function_meanings)
