# (c) This file is part of the course
# Mathematical Logic through Programming
# by Gonczarowski and Nisan.
# File name: propositions/syntax.py

"""Syntactic handling of propositional formulae."""

from __future__ import annotations
from typing import Mapping, Optional, Set, Tuple, Union

from logic_utils import frozen

def is_variable(s: str) -> bool:
    """Checks if the given string is an atomic proposition.

    Parameters:
        s: string to check.

    Returns:
        ``True`` if the given string is an atomic proposition, ``False``
        otherwise.
    """
    return s[0] >= 'p' and s[0] <= 'z' and (len(s) == 1 or s[1:].isdigit())

def is_constant(s: str) -> bool:
    """Checks if the given string is a constant.

    Parameters:
        s: string to check.

    Returns:
        ``True`` if the given string is a constant, ``False`` otherwise.
    """
    return s == 'T' or s == 'F'

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
    # return s == '&' or s == '|' or s == '->'
    # For Chapter 3:
    return s in {'&', '|',  '->', '+', '<->', '-&', '-|'}

@frozen
class Formula:
    """An immutable propositional formula in tree representation.

    Attributes:
        root (`str`): the constant, atomic proposition, or operator at the root
            of the formula tree.
        first (`~typing.Optional`\\[`Formula`]): the first operand to the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand to the
            root, if the root is a binary operator.
    """
    root: str
    first: Optional[Formula]
    second: Optional[Formula]

    def __init__(self, root: str, first: Optional[Formula] = None,
                 second: Optional[Formula] = None) -> None:
        """Initializes a `Formula` from its root and root operands.

        Parameters:
            root: the root for the formula tree.
            first: the first operand to the root, if the root is a unary or
                binary operator.
            second: the second operand to the root, if the root is a binary
                operator.
        """
        if is_variable(root) or is_constant(root):
            assert first is None and second is None
            self.root = root
        elif is_unary(root):
            assert type(first) is Formula and second is None
            self.root, self.first = root, first
        else:
            assert is_binary(root) and type(first) is Formula and \
                   type(second) is Formula
            self.root, self.first, self.second = root, first, second

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
            does not equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        if is_variable(self.root) or is_constant(self.root):
            return self.root
        elif is_unary(self.root):
            return self.root + self.first.__repr__()
        elif is_binary(self.root):
            return "(" + self.first.__repr__() + self.root + self.second.__repr__() + ")"


    def variables(self) -> Set[str]:
        """Finds all atomic propositions (variables) in the current formula.

        Returns:
            A set of all atomic propositions used in the current formula.
        """
        all_variables = set()
        if is_variable(self.root):
            all_variables.add(self.root)
        elif is_unary(self.root):
            all_variables.update(self.first.variables())
        elif is_binary(self.root):
            all_variables.update(self.first.variables())
            all_variables.update(self.second.variables())
        return all_variables

    def operators(self) -> Set[str]:
        """Finds all operators in the current formula.

        Returns:
            A set of all operators (including ``'T'`` and ``'F'``) used in the
            current formula.
        """
        all_operators = set()
        if is_unary(self.root):
            all_operators.add(self.root)
            all_operators.update(self.first.operators())
        elif is_binary(self.root):
            all_operators.add(self.root)
            all_operators.update(self.first.operators())
            all_operators.update(self.second.operators())
        elif is_constant(self.root):
            all_operators.add(self.root)
        return all_operators

    @staticmethod
    def parse_prefix(s: str) -> Tuple[Union[Formula, None], str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            s: string to parse.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the first token of the string is a variable name (e.g.,
            ``'x12'``), then the parsed prefix will be that entire variable name
            (and not just a part of it, such as ``'x1'``). If no prefix of the
            given string is a valid standard string representation of a formula
            then returned pair should be of ``None`` and an error message, where
            the error message is a string with some human-readable content.
        """
        if len(s) <= 0:
            return (None, 'Wrong format of formula')
        elif len(s) == 1:
            if is_constant(s) or is_variable(s):
                return (Formula(s), '')
            else:
                return (None, 'Wrong format of formula')
        else:
            if is_variable(s[0]):
                var = s[0]
                i = 1
                while (i <= len(s) - 1 and s[i].isdigit()):
                    var += s[i]
                    i += 1
                if len(s) > len(var):
                    return (Formula(var), s[len(var):])
                else:
                    return (Formula(var), '')
            elif is_constant(s[0]):
                return (Formula(s[0]), s[1:])
            elif is_unary(s[0]):
                f, sub_string = Formula.parse_prefix(s[1:])
                if f != None:
                    return (Formula('~', f), sub_string)
            elif s[0] == '(':
                f, sub_string = Formula.parse_prefix(s[1:])
                if f == None or sub_string == '':
                    return (None, 'Wrong format of formula')
                elif not (is_binary(sub_string[0]) or is_binary(sub_string[:2]) or is_binary(sub_string[:3])):
                    return (None, 'Wrong format of formula')
                else:
                    if sub_string[0] == '-' and sub_string[1] in ['>', '&', '|']:
                        q, sub_string2 = Formula.parse_prefix(sub_string[2:])
                        if q == None or sub_string2 == '' or sub_string2[0] != ')':
                            return (None, 'Wrong format of formula')
                        else:
                            return (Formula(sub_string[:2], f, q), sub_string2[1:])
                    elif sub_string[0] == '<' and sub_string[1] == '-' and sub_string[2] == '>':
                        q, sub_string2 = Formula.parse_prefix(sub_string[3:])
                        if q == None or sub_string2 == '' or sub_string2[0] != ')':
                            return (None, 'Wrong format of formula')
                        else:
                            return (Formula(sub_string[:3], f, q), sub_string2[1:])
                    for op in ['|', '&', '+']:
                        if sub_string[0] == op:
                            q, sub_string2 = Formula.parse_prefix(sub_string[1:])
                            if (q == None or sub_string2 == '' or sub_string2[0] != ')'):
                                return (None, 'Wrong format of formula')
                            else:
                                return (Formula(op, f, q), sub_string2[1:])
            else:
                return (None, 'Wrong format of formula')


    @staticmethod
    def is_formula(s: str) -> bool:
        """Checks if the given string is a valid representation of a formula.

        Parameters:
            s: string to check.

        Returns:
            ``True`` if the given string is a valid standard string
            representation of a formula, ``False`` otherwise.
        """
        f, sub_string = Formula.parse_prefix(s)
        if (f != None and sub_string == ''):
            return True
        return False

    @staticmethod
    def parse(s: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            s: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        assert Formula.is_formula(s)
        f, sub_string = Formula.parse_prefix(s)
        return f

# Optional tasks for Chapter 1

    def polish(self) -> str:
        """Computes the polish notation representation of the current formula.

        Returns:
            The polish notation representation of the current formula.
        """
        # Optional Task 1.7

    @staticmethod
    def parse_polish(s: str) -> Formula:
        """Parses the given polish notation representation into a formula.

        Parameters:
            s: string to parse.

        Returns:
            A formula whose polish notation representation is the given string.
        """
        # Optional Task 1.8

# Tasks for Chapter 3

    def substitute_variables(
            self, substitution_map: Mapping[str, Formula]) -> Formula:
        """Substitutes in the current formula, each variable `v` that is a key
        in `substitution_map` with the formula `substitution_map[v]`.

        Parameters:
            substitution_map: the mapping defining the substitutions to be
                performed.

        Returns:
            The resulting formula.

        Examples:
            >>> Formula.parse('((p->p)|z)').substitute_variables(
            ...     {'p': Formula.parse('(q&r)')})
            (((q&r)->(q&r))|z)
        """
        for variable in substitution_map:
            assert is_variable(variable)
        if is_constant(self.root):
            return self
        if is_variable(self.root):
            if self.root in substitution_map.keys():
                return substitution_map[self.root]
            else:
                return self
        else:
            if is_unary(self.root):
                return Formula(self.root, self.first.substitute_variables(substitution_map))
            elif is_binary(self.root):
                return Formula(self.root, self.first.substitute_variables(substitution_map), self.second.substitute_variables(substitution_map))

    def substitute_operators(
            self, substitution_map: Mapping[str, Formula]) -> Formula:
        """Substitutes in the current formula, each constant or operator `op`
        that is a key in `substitution_map` with the formula
        `substitution_map[op]` applied to its (zero or one or two) operands,
        where the first operand is used for every occurrence of ``'p'`` in the
        formula and the second for every occurrence of ``'q'``.

        Parameters:
            substitution_map: the mapping defining the substitutions to be
                performed.

        Returns:
            The resulting formula.

        Examples:
            >>> Formula.parse('((x&y)&~z)').substitute_operators(
            ...     {'&': Formula.parse('~(~p|~q)')})
            ~(~~(~x|~y)|~~z)
        """
        for operator in substitution_map:
            assert is_binary(operator) or is_unary(operator) or \
                   is_constant(operator)
            assert substitution_map[operator].variables().issubset({'p', 'q'})
        if self.root in substitution_map:
            if is_unary(self.root):
                var_map = {}
                if (is_constant(self.first.root) and self.first.root not in substitution_map)\
                        or is_variable(self.first.root):
                    var_map['p'] = self.first
                else:
                    var_map['p'] = self.first.substitute_operators(substitution_map)
                return substitution_map[self.root].substitute_variables(var_map)
            elif is_binary(self.root):
                var_map = {}
                if (is_constant(self.first.root) and self.first.root not in substitution_map) or\
                        is_variable(self.first.root):
                    var_map['p'] = self.first
                else:
                    var_map['p'] = self.first.substitute_operators(substitution_map)
                if (is_constant(self.second.root) and self.second.root not in substitution_map) or\
                        is_variable(self.second.root):
                    var_map['q'] = self.second
                else:
                    var_map['q'] = self.second.substitute_operators(substitution_map)
                return substitution_map[self.root].substitute_variables(var_map)
            elif is_constant(self.root):
                return substitution_map[self.root]
        # check whether one or more of the elements in the formula is a constant that belongs to the dictionary
        elif (not self.root in substitution_map) and is_unary(self.root) and self.first.root in substitution_map:
            return Formula(self.root, substitution_map[self.first.root])
        elif (not self.root in substitution_map) and is_binary(self.root) and \
                (self.first.root in substitution_map or self.second.root in substitution_map):
            if self.first.root in substitution_map:
                if self.second.root in substitution_map:
                    return Formula(self.root, substitution_map[self.first.root], substitution_map[self.second.root])
                return Formula(self.root, substitution_map[self.first.root], self.second)
            elif self.second.root in substitution_map:
                return Formula(self.root, self.first, substitution_map[self.second.root])
        else:
            return self