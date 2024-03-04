#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import rich
from pydantic import BaseModel

from lucid_generate_data.utils.commands import Command, Perform, CommandRegistry, EntityQuery
from lucid_generate_data.utils.definitions import (
    ActionResult,
    ProgramTurn,
    RecommendedAction,
)
from lucid_generate_data.utils.entities import Entity
from lucid_generate_data.executor.rewrite_commands import RewriteConfirm, RewriteResume


def _var_name(index: int) -> str:
    return f"x{index}"


def _index(var_name: str) -> str:
    return int(var_name[1:])


def _normalise_args(
    args: list[ast.expr], keywords: list[ast.keyword], command_builder: Type[Command]
) -> dict[str, ast.expr]:
    keyword_args = {}
    if len(args):
        pos_arg_names = command_builder.positional_args()
        if len(args) > len(pos_arg_names):
            raise ValueError("Too many positional arguments passed")
        for arg, arg_name in zip(args, pos_arg_names):
            keyword_args[arg_name] = arg

    for keyword in keywords:
        assert keyword.arg is not None
        keyword_args[keyword.arg] = keyword.value

    return keyword_args


@dataclass
class ExecutionState:
    assignments: dict[str, Any] = field(default_factory=dict)
    app_context: dict[str, Any] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    current_recommendation: Optional[Tuple[str, RecommendedAction]] = None

    def get_incomplete_tasks(self) -> List[str]:
        return [k for k in self.assignments if k not in self.completed]

    def get_assignment(self, var_name: str, idx: Optional[int] = None) -> Command:
        var_obj = self.assignments.get(var_name)
        if var_obj is None:
            raise ValueError(f"{var_name} not found")
        if not isinstance(var_obj, Command) and not isinstance(var_obj, Entity):
            raise ValueError(f"{var_name} must be a Command or Entity (was {type(var_obj)})")

        if isinstance(var_obj, EntityQuery):
            return var_obj.get_entity(idx)

        return var_obj

    def flag_recommendation_followed(self) -> None:
        if self.current_recommendation is None:
            print("WARNING: A non-existent recommendation was followed")
            return None
        task = self.assignments[self.current_recommendation[0]]
        task.notify_action(self.current_recommendation[1].name)


class Call(BaseModel):
    var_name: str

    def _run(self, state: ExecutionState) -> ActionResult:
        raise NotImplementedError

    @property
    def index(self):
        return _index(self.var_name)

    def run(self, state: ExecutionState, follows_recommendation: bool) -> ActionResult:
        if follows_recommendation:
            state.flag_recommendation_followed()
        return self._run(state)

    def _perform_or_recommend(self, state: ExecutionState, command: Command) -> ActionResult:
        assert command is not None
        recommended_action = command.recommend_action()
        if recommended_action is not None:
            state.current_recommendation = (self.var_name, recommended_action)
            return ActionResult(recommended_action=recommended_action, index=self.index)
        else:
            state.current_recommendation = None

            result = ActionResult(result=command.perform(state.app_context), index=self.index)
            state.completed.add(self.var_name)
            return result


class CommandCall(Call):
    command: Command
    assigned: bool

    def _run(self, state: ExecutionState) -> ActionResult:
        if not self.assigned:
            state.assignments[self.var_name] = self.command
        return self._perform_or_recommend(state, self.command)


class PerformCall(Call):
    entity: Optional[Entity] = None

    def _run(self, state: ExecutionState) -> ActionResult:
        if self.entity is not None:
            # Provides access to entity x2 in x2 = perform(x1)
            state.assignments[self.var_name] = self.entity
        return ActionResult(index=self.index)


class AssignmentCall(Call):
    attribute_name: List[str]
    idx: List[Optional[int]]
    value: List[Any]

    def _run(self, state: ExecutionState) -> ActionResult:
        for att, ndx, val in zip(self.attribute_name, self.idx, self.value):
            command = state.get_assignment(self.var_name, ndx)
            setattr(command, att, val)

        return self._perform_or_recommend(state, command)


class AppendCall(Call):
    attribute_name: str
    value: Any

    def _run(self, state: ExecutionState) -> ActionResult:
        command = state.get_assignment(self.var_name)
        list_field = getattr(command, self.attribute_name, None)
        assert isinstance(list_field, list)
        list_field.append(self.value)

        return self._perform_or_recommend(state, command)


class Value(Call):
    value: Any

    def _run(self, state: ExecutionState) -> ActionResult:
        return ActionResult(result=self.value, index=self.index)


@dataclass
class ProgramExecutor:
    registry: CommandRegistry
    state: ExecutionState = ExecutionState()
    rewrites: Tuple[ast.NodeTransformer, ...] = (RewriteConfirm(), RewriteResume())

    def execute_program(self, turns: List[ProgramTurn]) -> None:
        for turn in turns:
            rich.print(f"\n[underline]Turn {turn.index}")
            print(f"expression: {turn.expression}")
            result = self.execute_turn(turn)
            rich.print(f"result: {result}")

    @staticmethod
    def _parse_program_turn(turn: ProgramTurn) -> Tuple[ast.Module, bool]:
        pattern = re.compile(
            r"""([^#\n]+)  # Anything on a single line that does not include '#'. This is m[1]
               \n?        # A possible newline
               (\#y)?     # A possible occurrence of the pattern #y. This is m[2]""",
            re.X,
        )
        assert (m := re.search(pattern, turn.expression)) is not None
        followed_hint = m[2] is not None
        return ast.parse(m[1]), followed_hint

    def rewrite_ast(self, turn_ast: ast.Module) -> ast.Module:
        for rewrite in self.rewrites:
            turn_ast = rewrite.visit(turn_ast)
        return turn_ast

    def execute_turn(self, turn: ProgramTurn) -> ActionResult:
        turn_ast, recommendation_followed = self._parse_program_turn(turn)
        turn_ast = self.rewrite_ast(turn_ast)
        call = self._get_call(turn.index, turn_ast)
        result = call.run(self.state, recommendation_followed)

        return result

    def _get_assignment_calls(
        self, attributes: List[ast.Attribute], values: List[ast.Constant]
    ) -> List[AssignmentCall]:
        grouped: Dict[str, List[Tuple[str, Any]]] = {}
        for att, val in zip(attributes, values):
            match att.value:
                case ast.Name():
                    var_name = att.value.id
                    idx = None
                    attr_name = att.attr
                case ast.Subscript():
                    var_name = att.value.value.id
                    idx = att.value.slice.value
                    attr_name = att.attr
                case _:
                    raise ValueError("Assignment query not valid")
            executed_value = self._execute_expr(val)
            grouped.setdefault(var_name, []).append((attr_name, idx, executed_value))

        return [
            AssignmentCall(
                var_name=k,
                idx=[g[1] for g in v],
                attribute_name=[g[0] for g in v],
                value=[g[2] for g in v],
            )
            for k, v in grouped.items()
        ]

    def _get_call(self, index: int, ast_module: ast.Module) -> Call:
        match ast_module.body:
            case [
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(value=name, attr=attribute_name), attr="append"
                        ),
                        args=args,
                    )
                )
            ]:
                # e.g. x0.entrees.append('fries')
                var_name = name.id  # type: ignore
                return AppendCall(
                    var_name=var_name,
                    attribute_name=attribute_name,
                    value=self._execute_expr(args[0]),
                )

            case [ast.Expr(value=ast.Call() as expr)]:
                # e.g. set_timer(duration='10 minutes')
                var_name = _var_name(index)
                value = self._execute_expr(expr)
                if isinstance(value, Perform):
                    return PerformCall(var_name=var_name, entity=value.args.entity)
                elif isinstance(value, Command):
                    return CommandCall(var_name=var_name, command=value, assigned=False)
                else:
                    return Value(var_name=var_name, value=value)

            case [ast.Expr(value=ast.Name() as expr)]:
                # e.g. 'x0'
                var_name = expr.id
                value = self._execute_expr(expr)
                if isinstance(value, Command):
                    return CommandCall(var_name=var_name, command=value, assigned=True)
                else:
                    return Value(var_name=var_name, value=value)

            case [ast.Assign(targets=[ast.Attribute() as attribute], value=value) as assign]:
                # e.g. x0.duration = '10 minutes'
                # but also x0[0].label = '10 minutes'
                return self._get_assignment_calls([attribute], [cast(ast.Constant, value)])[0]

            case [
                ast.Assign(targets=[ast.Tuple() as attribute_tuple], value=value_tuple) as assign
            ]:
                # e.g. x0.duration, x0.label = '10 minutes','noodles'
                values = cast(list[ast.Constant], cast(ast.Tuple, value_tuple).elts)
                attributes = [cast(ast.Attribute, x) for x in attribute_tuple.elts]
                assignments = self._get_assignment_calls(attributes, values)
                # It is possible for multiple tasks to be updated in a multi-assign statement
                # In this case we will just return the task with the highest number of updated
                # values
                return max(assignments, key=lambda a: len(a.value))

            case []:
                raise ValueError("No expressions found")
            case _:
                raise ValueError("Exactly one expression or assignment allowed")

    def _execute_expr(self, ast_expr: ast.expr) -> Any:
        """Executes a given action (or part of an action) represented as an abstract
         syntax tree (AST). Execution means that the expression tree is recursively
         traversed and  static assignment nodes are resolved to their value. A
         `python` object required to execute the current action is returned. The execution:

         1. Converts LLM actions to instances of corresponding `python` objects
            (eg `contact` -> `Contact()`, send_message -> `SendMessage()`)
         2. LLM-generated positional/keyword args are set as command attributes. If an arg
            is a variable previously assigned to, the executor recursively resolves
            it to its value (which can be a python object or simple type (str, int, None but
            also immutable container types such af frozenset/tuple)

        Args
            ast_expr: the abstract syntax tree of the action currently executed (or
             a sub-tree of said action)
        """
        match ast_expr:
            # matches a fcn call e.g., send_message(contents="Hi")
            case ast.Call(ast.Name() as func, args, keywords):

                command_name = func.id  # "send_message"
                command_builder = self.registry.commands.get(command_name)
                if command_builder is None:
                    raise ValueError(f"{command_name} is not registered")

                command = command_builder.build()  # instance of SendMessage

                # {'contents': Constant(value='Hi')}
                keyword_args = _normalise_args(args, keywords, command_builder)

                # e.g., set 'contents' to self._execute_expr(value=Constant(value='Hi'))
                # which recursively resolves to "Hi"
                for arg, value in keyword_args.items():
                    setattr(command, arg, self._execute_expr(value))

                return command
            # Constant(value='Hi') -> 'Hi'
            case ast.Constant(value):
                return value
            # we assigned a list to a variable/command arg, so we must resolve the variables
            # or expressions in the list (eg when the action is x1.recipients = [x5,x3] this
            # resolves [x3, x5] to a list containing the values of those vars (which can be
            # expressions)
            case ast.List(elts):
                return [self._execute_expr(elem) for elem in elts]
            # expression is a variable name (eg x3) -> return what is assigned to x3
            case ast.Name(id_):
                var = self.state.assignments.get(id_)
                # we must have assigned the variable to resolve it
                return var
            # x1.recipients = [x3] -> returns [x3], which would then be matched by the
            # ast.List(elts) case first and then by case ast.Name
            case ast.Attribute(value=ast.Name(id=id_), attr=attr):
                var = self.state.assignments.get(id_)
                if var is None:
                    raise ValueError(f"Variable {id_} not found")
                return getattr(var, attr)
            case ast.Attribute(
                value=ast.Subscript(value=ast.Name(id=id_), slice=ast.Constant(value=idx)),
                attr=attr,
            ):
                var = self.state.assignments.get(id_)
                if var is None:
                    raise ValueError(f"Variable {id_} not found")
                var = var[idx]
                return getattr(var, attr)
            case _:
                raise ValueError(f"Unsupported expression {ast_expr}")
