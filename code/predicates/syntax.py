# (c) This file is part of the course
# Mathematical Logic through Programming
# by Gonczarowski and Nisan.
# File name: predicates/syntax.py

"""Syntactic handling of first-order formulas and terms."""

from __future__ import annotations
from typing import AbstractSet, Mapping, Optional, Sequence, Set, Tuple, Union

from logic_utils import fresh_variable_name_generator, frozen

from propositions.syntax import Formula as PropositionalFormula, \
    is_variable as is_propositional_variable


class ForbiddenVariableError(Exception):
    """Raised by `Term.substitute` and `Formula.substitute` when a substituted
    term contains a variable name that is forbidden in that context."""

    def __init__(self, variable_name: str) -> None:
        """Initializes a `ForbiddenVariableError` from its offending variable
        name.

        Parameters:
            variable_name: variable name that is forbidden in the context in
                which a term containing it was to be substituted.
        """
        assert is_variable(variable_name)
        self.variable_name = variable_name


def is_constant(s: str) -> bool:
    """Checks if the given string is a constant name.

    Parameters:
        s: string to check.

    Returns:
        ``True`` if the given string is a constant name, ``False`` otherwise.
    """
    return (((s[0] >= '0' and s[0] <= '9') or (s[0] >= 'a' and s[0] <= 'd'))
            and s.isalnum()) or s == '_'


def is_variable(s: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        s: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return s[0] >= 'u' and s[0] <= 'z' and s.isalnum()


def is_function(s: str) -> bool:
    """Checks if the given string is a function name.

    Parameters:
        s: string to check.

    Returns:
        ``True`` if the given string is a function name, ``False`` otherwise.
    """
    return s[0] >= 'f' and s[0] <= 't' and s.isalnum()


@frozen
class Term:
    """An immutable first-order term in tree representation, composed from
    variable names and constant names, and function names applied to them.

    Attributes:
        root (`str`): the constant name, variable name, or function name at the
            root of the term tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments to the root, if the root is a function name.
    """
    root: str
    arguments: Optional[Tuple[Term, ...]]

    def __init__(self, root: str,
                 arguments: Optional[Sequence[Term]] = None) -> None:
        """Initializes a `Term` from its root and root arguments.

        Parameters:
            root: the root for the formula tree.
            arguments: the arguments to the root, if the root is a function
                name.
        """
        if is_constant(root) or is_variable(root):
            assert arguments is None
            self.root = root
        else:
            assert is_function(root)
            assert arguments is not None
            self.root = root
            self.arguments = tuple(arguments)
            assert len(self.arguments) > 0

    def __repr__(self) -> str:
        """Computes the string representation of the current term.

        Returns:
            The standard string representation of the current term.
        """
        # Task 7.1
        if is_variable(self.root) or is_constant(self.root):
            return self.root
        else:
            term_repr = self.root + "("
            if not self.arguments:
                return term_repr + ")"
            for arg in self.arguments:
                term_repr += arg.__repr__() + ','
            term_repr = term_repr[:-1] + ')'
            return term_repr

    def __eq__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Term` object that equals the
            current term, ``False`` otherwise.
        """
        return isinstance(other, Term) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current term with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Term` object or does not
            equal the current term, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def parse_prefix(s: str) -> Tuple[Term, str]:
        """Parses a prefix of the given string into a term.

        Parameters:
            s: string to parse, which has a prefix that is a valid
                representation of a term.

        Returns:
            A pair of the parsed term and the unparsed suffix of the string. If
            the given string has as a prefix a constant name (e.g., ``'c12'``)
            or a variable name (e.g., ``'x12'``), then the parsed prefix will be
            that entire name (and not just a part of it, such as ``'x1'``).
        """
        # Task 7.3.1
        if is_variable(s[0]):
            var = s[0]
            i = 1
            while (i <= len(s) - 1 and (s[i].isdigit() or s[i].isalpha())):
                var += s[i]
                i += 1
            return (Term(var), s[len(var):])
        elif is_constant(s[0]):
            if s[0] == '_':
                return (Term(s[0]), s[1:])
            else:
                const = s[0]
                i = 1
                while (i <= len(s) - 1 and (s[i].isdigit() or s[i].isalpha())):
                    const += s[i]
                    i += 1
                return (Term(const), s[len(const):])
        else:
            split_by_first_par = s.split('(', 1)
            function_name = split_by_first_par[0]
            if is_function(function_name):
                all_terms = []  # save the term objects in the function
                arg, left_string = Term.parse_prefix(split_by_first_par[1])
                while left_string[0] == ',':
                    all_terms.append(arg)
                    arg, left_string = Term.parse_prefix(left_string[1:])
                if left_string[0] == ')':
                    all_terms.append(arg)
                    return (Term(function_name, all_terms), left_string[1:])

    @staticmethod
    def parse(s: str) -> Term:
        """Parses the given valid string representation into a term.

        Parameters:
            s: string to parse.

        Returns:
            A term whose standard string representation is the given string.
        """
        # Task 7.3.2
        return Term.parse_prefix(s)[0]

    def constants(self) -> Set[str]:
        """Finds all constant names in the current term.

        Returns:
            A set of all constant names used in the current term.
        """
        if is_constant(self.root):
            return {self.root}
        elif is_variable(self.root):
            return set()
        else:
            all_constants = set()
            for arg in self.arguments:
                all_constants.update(arg.constants())
            return all_constants

    def variables(self) -> Set[str]:
        """Finds all variable names in the current term.

        Returns:
            A set of all variable names used in the current term.
        """
        if is_variable(self.root):
            return {self.root}
        elif is_constant(self.root):
            return set()
        else:
            all_variables = set()
            for arg in self.arguments:
                all_variables.update(arg.variables())
            return all_variables

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current term, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current term.
        """
        if is_variable(self.root) or is_constant(self.root):
            return set()
        else:
            all_functions = {(self.root, len(self.arguments))}
            for arg in self.arguments:
                all_functions.update(arg.functions())
            return all_functions

    def substitute(self, substitution_map: Mapping[str, Term],
                   forbidden_variables: AbstractSet[str] = frozenset()) -> Term:
        """Substitutes in the current term, each constant name `name` or
        variable name `name` that is a key in `substitution_map` with the term
        `substitution_map[name]`.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variables not allowed in substitution terms.

        Returns:
            The term resulting from performing all substitutions. Only
            constant names and variable names originating in the current term
            are substituted (i.e., those originating in one of the specified
            substitutions are not subjected to additional substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable from `forbidden_variables`.

        Examples:
            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'y'})
            f(c,plus(d,x))
            >>> Term.parse('f(x,c)').substitute(
            ...     {'c': Term.parse('plus(d,y)')}, {'y'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for element_name in substitution_map:
            assert is_constant(element_name) or is_variable(element_name)
        for variable in forbidden_variables:
            assert is_variable(variable)
        # Task 9.1
        for key in substitution_map:
            vars_in_term = substitution_map[key].variables()
            for var in vars_in_term:
                if var in forbidden_variables:
                    raise (ForbiddenVariableError(var))
        if is_variable(self.root) or is_constant(self.root):
            if self.root in substitution_map:
                return substitution_map[self.root]
            return self
        else:
            new_args = []
            for arg in self.arguments:
                new_args.append(arg.substitute(substitution_map, forbidden_variables))
            return Term(self.root, new_args)


def is_equality(s: str) -> bool:
    """Checks if the given string is the equality relation.

    Parameters:
        s: string to check.

    Returns:
        ``True`` if the given string is the equality relation, ``False``
        otherwise.
    """
    return s == '='


def is_relation(s: str) -> bool:
    """Checks if the given string is a relation name.

    Parameters:
        s: string to check.

    Returns:
        ``True`` if the given string is a relation name, ``False`` otherwise.
    """
    return s[0] >= 'F' and s[0] <= 'T' and s.isalnum()


def is_unary(s: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        s: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return s == '~'


def is_binary(s: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        s: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return s == '&' or s == '|' or s == '->'


def is_quantifier(s: str) -> bool:
    """Checks if the given string is a quantifier.

    Parameters:
        s: string to check.

    Returns:
        ``True`` if the given string is a quantifier, ``False`` otherwise.
    """
    return s == 'A' or s == 'E'


@frozen
class Formula:
    """An immutable first-order formula in tree representation, composed from
    relation names applied to first-order terms, and operators and
    quantifications applied to them.

    Attributes:
        root (`str`): the relation name, equality relation, operator, or
            quantifier at the root of the formula tree.
        arguments (`~typing.Optional`\\[`~typing.Tuple`\\[`Term`, ...]]): the
            arguments to the root, if the root is a relation name or the
            equality relation.
        first (`~typing.Optional`\\[`Formula`]): the first operand to the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second
            operand to the root, if the root is a binary operator.
        variable (`~typing.Optional`\\[`str`]): the variable name quantified by
            the root, if the root is a quantification.
        predicate (`~typing.Optional`\\[`Formula`]): the predicate quantified by
            the root, if the root is a quantification.
    """
    root: str
    arguments: Optional[Tuple[Term, ...]]
    first: Optional[Formula]
    second: Optional[Formula]
    variable: Optional[str]
    predicate: Optional[Formula]

    def __init__(self, root: str,
                 arguments_or_first_or_variable: Union[Sequence[Term],
                                                       Formula, str],
                 second_or_predicate: Optional[Formula] = None) -> None:
        """Initializes a `Formula` from its root and root arguments, root
        operands, or root quantified variable and predicate.

        Parameters:
            root: the root for the formula tree.
            arguments_or_first_or_variable: the arguments to the the root, if
                the root is a relation name or the equality relation; the first
                operand to the root, if the root is a unary or binary operator;
                the variable name quantified by the root, if the root is a
                quantification.
            second_or_predicate: the second operand to the root, if the root is
                a binary operator; the predicate quantified by the root, if the
                root is a quantification.
        """
        if is_equality(root) or is_relation(root):
            # Populate self.root and self.arguments
            assert second_or_predicate is None
            assert isinstance(arguments_or_first_or_variable, Sequence) and \
                   not isinstance(arguments_or_first_or_variable, str)
            self.root, self.arguments = \
                root, tuple(arguments_or_first_or_variable)
            if is_equality(root):
                assert len(self.arguments) == 2
        elif is_unary(root):
            # Populate self.first
            assert isinstance(arguments_or_first_or_variable, Formula) and \
                   second_or_predicate is None
            self.root, self.first = root, arguments_or_first_or_variable
        elif is_binary(root):
            # Populate self.first and self.second
            assert isinstance(arguments_or_first_or_variable, Formula) and \
                   second_or_predicate is not None
            self.root, self.first, self.second = \
                root, arguments_or_first_or_variable, second_or_predicate
        else:
            assert is_quantifier(root)
            # Populate self.variable and self.predicate
            assert isinstance(arguments_or_first_or_variable, str) and \
                   is_variable(arguments_or_first_or_variable) and \
                   second_or_predicate is not None
            self.root, self.variable, self.predicate = \
                root, arguments_or_first_or_variable, second_or_predicate

    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        if is_equality(self.root):
            return self.arguments[0].__repr__() + self.root + self.arguments[1].__repr__()
        elif is_relation(self.root):
            formula_repr = self.root + '('
            if not self.arguments:
                return formula_repr + ')'
            for arg in self.arguments:
                formula_repr += arg.__repr__() + ','
            formula_repr = formula_repr[:-1] + ')'
            return formula_repr
        elif is_unary(self.root):
            return '~' + self.first.__repr__()
        elif is_binary(self.root):
            return '(' + self.first.__repr__() + self.root + self.second.__repr__() + ')'
        elif is_quantifier(self.root):
            return self.root + self.variable + '[' + self.predicate.__repr__() + ']'

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def parse_prefix(s: str) -> Tuple[Formula, str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            s: string to parse, which has a prefix that is a valid
                representation of a formula.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a term followed by an equality
            followed by a constant name (e.g., ``'c12'``) or by a variable name
            (e.g., ``'x12'``), then the parsed prefix will include that entire
            name (and not just a part of it, such as ``'x1'``).
        """
        # Task 7.4.1
        if is_unary(s[0]):
            form, leftover = Formula.parse_prefix(s[1:])
            form = Formula(s[0], form)
            return form, leftover
        if s[0] == "(":
            return Formula.parse_prefix_for_binary(s)
        if is_quantifier(s[0]):
            quant = s[0]
            term, form = s[1:].split("[", 1)
            form, leftover = Formula.parse_prefix(form)
            return Formula(quant, term, form), leftover[1:]
        if is_relation(s[0]):
            return Formula.parse_prefix_for_relation(s)
        else:
            term1, equal, leftover = s.partition("=")
            term2, suffix = Term.parse_prefix(leftover)
            return Formula(equal, [Term.parse(term1), term2]), suffix

    @staticmethod
    def parse_prefix_for_relation(s):
        i = 0
        while s[i] != "(":
            i += 1
        relation_oper = s[:i]
        term_lst = []
        if s[i + 1] == ")":
            return Formula(relation_oper, term_lst), s[i + 2:]
        arg, left_string = Term.parse_prefix(s[i + 1:])
        while left_string[0] == ',':
            term_lst.append(arg)
            arg, left_string = Term.parse_prefix(left_string[1:])
        if left_string[0] == ')':
            term_lst.append(arg)
        return Formula(relation_oper, term_lst), left_string[1:]

    @staticmethod
    def parse_prefix_for_binary(s):
        """
        An helper method for the parse prefix method, this method will
        parse_prefix a given string that starts with parenthesis, and will check if it
        is a binary operator.
        :param s: a string that starts with a parenthesis.
        :return: upon success A tuple of Formula and string, None and error msg otherwise.
        """
        form, suffix = Formula.parse_prefix(s[1:])
        if form is None:
            return None, suffix
        if is_binary(suffix[0]):
            bin_oper = suffix[0]
            sec_form, new_suffix = Formula.parse_prefix(suffix[1:])
        elif len(suffix) > 1 and is_binary(suffix[:2]):
            bin_oper = suffix[:2]
            sec_form, new_suffix = Formula.parse_prefix(suffix[2:])
        else:
            return None, "No binary operator found for the amount of parenthesis"
        if sec_form is None:
            return None, new_suffix
        if new_suffix == "" or new_suffix[0] != ")":
            return None, "Parenthesis missing"
        return Formula(bin_oper, form, sec_form), new_suffix[1:]

    @staticmethod
    def parse(s: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            s: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        # Task 7.4.2
        return Formula.parse_prefix(s)[0]

    def constants(self) -> Set[str]:
        """Finds all constant names in the current formula.

        Returns:
            A set of all constant names used in the current formula.
        """
        # Task 7.6.1
        const_set = set()
        if is_constant(self.root):
            const_set.add(self.root)
            return const_set
        if is_unary(self.root):
            return self.first.constants()
        if is_binary(self.root):
            const_set1 = self.first.constants()
            const_set2 = self.second.constants()
            const_set = const_set1.union(const_set2)
            return const_set
        if is_relation(self.root):
            for term in self.arguments:
                const_set = const_set.union(term.constants())
            return const_set
        if is_quantifier(self.root):
            return self.predicate.constants()
        if self.root == "=":
            const_set1 = self.arguments[0].constants()
            const_set2 = self.arguments[1].constants()
            const_set = const_set1.union(const_set2)
            return const_set
        return const_set

    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        # Task 7.6.2
        var_set = set()
        if is_variable(self.root):
            var_set.add(self.root)
            return var_set
        if is_unary(self.root):
            return self.first.variables()
        if is_binary(self.root):
            var_set1 = self.first.variables()
            var_set2 = self.second.variables()
            var_set = var_set1.union(var_set2)
            return var_set
        if is_relation(self.root):
            for term in self.arguments:
                var_set = var_set.union(term.variables())
            return var_set
        if is_quantifier(self.root):
            var_set.add(self.variable)
            var_set = var_set.union(self.predicate.variables())
            return var_set
        if self.root == "=":
            var_set1 = self.arguments[0].variables()
            var_set2 = self.arguments[1].variables()
            var_set = var_set1.union(var_set2)
            return var_set
        return var_set

    def free_variables(self) -> Set[str]:
        """Finds all variable names that are free in the current formula.

        Returns:
            A set of all variable names used in the current formula not only
            within a scope of a quantification on those variable names.
        """
        # Task 7.6.3
        var_set = set()
        if is_variable(self.root):
            var_set.add(self.root)
            return var_set
        if is_unary(self.root):
            return self.first.free_variables()
        if is_binary(self.root):
            var_set1 = self.first.free_variables()
            var_set2 = self.second.free_variables()
            var_set = var_set1.union(var_set2)
            return var_set
        if is_relation(self.root):
            for term in self.arguments:
                var_set = var_set.union(term.variables())
            return var_set
        if is_quantifier(self.root):
            var_set = var_set.union(self.predicate.free_variables())
            var_set = var_set - set([self.variable])
            return var_set
        if self.root == "=":
            var_set1 = self.arguments[0].variables()
            var_set2 = self.arguments[1].variables()
            var_set = var_set1.union(var_set2)
            return var_set
        return var_set

    def functions(self) -> Set[Tuple[str, int]]:
        """Finds all function names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of function name and arity (number of arguments) for
            all function names used in the current formula.
        """
        # Task 7.6.4
        func_set = set()
        if is_unary(self.root):
            return self.first.functions()
        if is_binary(self.root):
            func_set1 = self.first.functions()
            func_set2 = self.second.functions()
            func_set = func_set1.union(func_set2)
            return func_set
        if is_relation(self.root):
            for term in self.arguments:
                func_set = func_set.union(term.functions())
            return func_set
        if is_quantifier(self.root):
            func_set = func_set.union(self.predicate.functions())
            return func_set
        if self.root == "=":
            func_set1 = self.arguments[0].functions()
            func_set2 = self.arguments[1].functions()
            func_set = func_set1.union(func_set2)
            return func_set

    def relations(self) -> Set[Tuple[str, int]]:
        """Finds all relation names in the current formula, along with their
        arities.

        Returns:
            A set of pairs of relation name and arity (number of arguments) for
            all relation names used in the current formula.
        """
        # Task 7.6.5
        relation_set = set()
        if is_relation(self.root):
            relation_set.add((self.root, len(self.arguments)))
            return relation_set
        if is_unary(self.root):
            return self.first.relations()
        if is_binary(self.root):
            rel_set1 = self.first.relations()
            rel_set2 = self.second.relations()
            relation_set = rel_set1.union(rel_set2)
            return relation_set
        if is_quantifier(self.root):
            return self.predicate.relations()
        return relation_set

    def substitute(self, substitution_map: Mapping[str, Term],
                   forbidden_variables: AbstractSet[str] = frozenset()) -> \
            Formula:
        """Substitutes in the current formula, each constant name `name` or free
        occurrence of variable name `name` that is a key in `substitution_map`
        with the term `substitution_map[name]`.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.
            forbidden_variables: variables not allowed in substitution terms.

        Returns:
            The formula resulting from performing all substitutions. Only
            constant names and variable names originating in the current formula
            are substituted (i.e., those originating in one of the specified
            substitutions are not subjected to additional substitutions).

        Raises:
            ForbiddenVariableError: If a term that is used in the requested
                substitution contains a variable from `forbidden_variables`
                or a variable occurrence that becomes bound when that term is
                substituted into the current formula.

        Examples:
            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,x)'), 'x': Term.parse('c')}, {'z'})
            Ay[c=plus(d,x)]
            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,z)')}, {'z'})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: z
            >>> Formula.parse('Ay[x=c]').substitute(
            ...     {'c': Term.parse('plus(d,y)')})
            Traceback (most recent call last):
              ...
            predicates.syntax.ForbiddenVariableError: y
        """
        for element_name in substitution_map:
            assert is_constant(element_name) or is_variable(element_name)
        for variable in forbidden_variables:
            assert is_variable(variable)
        # Task 9.2
        for key in substitution_map:
            vars_in_term = substitution_map[key].variables()
            for var in vars_in_term:
                if var in forbidden_variables:
                    raise (ForbiddenVariableError(var))
        if is_quantifier(self.root):
            updated_forbidden_variables = set(forbidden_variables)
            updated_forbidden_variables.add(self.variable)
            new_map = {}
            for key in substitution_map:
                if key in self.free_variables() or key in self.constants():
                    new_map[key] = substitution_map[key]
            new_formula = self.predicate.substitute(new_map, updated_forbidden_variables)
            return Formula(self.root, self.variable, new_formula)
        elif self.root == '=' or is_relation(self.root):
            new_args = []
            for arg in self.arguments:
                new_args.append(arg.substitute(substitution_map, forbidden_variables))
            return Formula(self.root, new_args)
        elif is_unary(self.root):
            return Formula(self.root, self.first.substitute(substitution_map, forbidden_variables))
        elif is_binary(self.root):
            new_first = self.first.substitute(substitution_map, forbidden_variables)
            new_second = self.second.substitute(substitution_map, forbidden_variables)
            return Formula(self.root, new_first, new_second)
        return self

    def propositional_skeleton(self) -> Tuple[PropositionalFormula,
                                              Mapping[str, Formula]]:
        """Computes a propositional skeleton of the current formula.

        Returns:
            A pair. The first element of the pair is a propositional formula
            obtained from the current formula by substituting every (outermost)
            subformula that has a relation or quantifier at its root with an
            atomic propositional formula, consistently such that multiple equal
            such (outermost) subformulas are substituted with the same atomic
            propositional formula. The atomic propositional formulas used for
            substitution are obtained, from left to right, by calling
            `next`\ ``(``\ `~logic_utils.fresh_variable_name_generator`\ ``)``.
            The second element of the pair is a map from each atomic
            propositional formula to the subformula for which it was
            substituted.
        """
        # Task 9.8
        form_dict = dict()
        return self.propositional_skeleton_helper(form_dict)

    def propositional_skeleton_helper(self, form_dict):
        """
        A recursive helper for the propositional_skeleton method.
        :param form_dict: a dictionary of types {str : formula} mapping each variable to
        the formula he assigned for.
        :return:
            A pair. The first element of the pair is a propositional formula
            obtained from the current formula by substituting every (outermost)
            subformula that has a relation or quantifier at its root with an
            atomic propositional formula, consistently such that multiple equal
            such (outermost) subformulas are substituted with the same atomic
            propositional formula. The atomic propositional formulas used for
            substitution are obtained, from left to right, by calling
            `next`\ ``(``\ `~logic_utils.fresh_variable_name_generator`\ ``)``.
            The second element of the pair is a map from each atomic
            propositional formula to the subformula for which it was
            substituted.
        """
        if is_unary(self.root):
            form_var, form_dict = self.first.propositional_skeleton_helper(form_dict)
            str_form = PropositionalFormula(self.root, form_var)
            return str_form, form_dict
        if is_binary(self.root):
            form_var1, form_dict = self.first.propositional_skeleton_helper(form_dict)
            form_var2, form_dict = self.second.propositional_skeleton_helper(form_dict)
            str_form = PropositionalFormula(self.root, form_var1, form_var2)
            return str_form, form_dict
        form_var = False
        for k, v in form_dict.items():
            if v == self:
                form_var = k
        if not form_var:
            form_var = next(fresh_variable_name_generator)
            form_dict[form_var] = self
        return PropositionalFormula(form_var), form_dict

    @staticmethod
    def from_propositional_skeleton(skeleton: PropositionalFormula,
                                    substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Computes a first-order formula from a propositional skeleton and a
        substitution map.

        Arguments:
            skeleton: propositional skeleton for the formula to compute.
            substitution_map: a map from each atomic propositional subformula
                of the given skeleton to a first-order formula.

        Returns:
            A first-order formula obtained from the given propositional skeleton
            by substituting each atomic propositional subformula with the formula
            mapped to it by the given map.
        """
        for key in substitution_map:
            assert is_propositional_variable(key)
        # Task 9.10
        if is_variable(skeleton.root):
            return substitution_map[skeleton.root]
        if is_unary(skeleton.root):
            return Formula(skeleton.root,
                           Formula.from_propositional_skeleton(skeleton.first, substitution_map))
        if is_binary(skeleton.root):
            first = Formula.from_propositional_skeleton(skeleton.first, substitution_map)
            second = Formula.from_propositional_skeleton(skeleton.second, substitution_map)
            return Formula(skeleton.root, first, second)