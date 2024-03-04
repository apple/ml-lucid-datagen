#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Optional

import docstring_parser
from pydantic import BaseModel, Field

from lucid_generate_data.utils.definitions import AppContext, RecommendedAction
from lucid_generate_data.utils.entities import Entity


def check_parsed_slots_vs_intent_json(slot_data_type: dict, intent_json):
    if slot_data_type is None:
        return False

    for slot, data_type in slot_data_type.items():
        if slot not in intent_json["args"]:
            print("Hallucinated slot name:", slot)
            return False
        if data_type != intent_json["args"][slot]["type"]:
            print("Invalid slot data type:", data_type, "for slot:", slot)
            return False
    return True


def parse_system_function_call(command: str):
    if "()" in command:
        return True, {}

    if "(" not in command or ")" not in command:
        print("LLM did not return a valid function: " + str(command))
        return False, None

    if '""' in command:
        print("Empty slot value")
        return False, None

    # Considering slot names and values inside the function call brackets
    command_args = command[command.find("(") + 1 : -1]

    # Remove str values
    brackets_open = False
    new_str = ""
    for char in command_args:
        if char == '"':
            new_str += char
            brackets_open = not brackets_open
        if not brackets_open and char != '"':
            new_str += char

    command_list = new_str.split(",")
    command_list = [x.strip() for x in command_list]
    command_list = [x.split("=") for x in command_list]

    slot_data_type_dict = {}
    for slot in command_list:
        if '"' in slot[1]:
            slot_data_type = "str"
        elif slot[1] == "True" or slot[1] == "False":
            slot_data_type = "bool"
        elif "." in slot[1]:
            slot_data_type = "float"
        else:
            slot_data_type = "int"

        slot_data_type_dict[slot[0]] = slot_data_type

    return True, slot_data_type_dict


def get_type_of_system_command(system_response: str):
    # TODO: At end, make sure all of these special types are being used. e.g. len

    if system_response.startswith("say()"):
        return "say_empty"

    elif system_response.startswith("say("):
        return "say_reference"

    elif system_response.startswith("perform("):
        return "perform"

    elif system_response.startswith("confirm("):
        return "confirm"

    elif system_response.startswith("len("):
        return "len"

    elif system_response.startswith("next("):
        return "next"

    elif system_response.startswith("append("):
        return "append"

    elif system_response.startswith("x") and ";" in system_response:
        return "multi_assignment"

    elif system_response.startswith("x"):
        return "single_assignment"

    elif "()" in system_response:
        return "intent_call_no_slots"

    else:
        assert "(" in system_response
        return "intent_call_with_slots"


def camel_to_snake_case(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class Command(BaseModel):
    entity: Optional[Entity] = None

    def recommend_action(self) -> Optional[RecommendedAction]:
        ...

    def perform(self, app_context: AppContext) -> Any:
        ...

    def notify_action(self, action_name: str) -> None:
        ...

    @classmethod
    def build(cls) -> Command:
        return cls()

    @classmethod
    def command_name(cls) -> str:
        return camel_to_snake_case(cls.__name__)

    @classmethod
    def positional_args(cls) -> list[str]:
        return []


class AutoCommand(Command):
    pass


class Hint(AutoCommand):
    """A recommandation coming from the app.

    Args:
        message: the hint to follow.
        ref: the reference to the task.
    """

    message: Optional[str] = None
    ref: Optional[str] = None

    @classmethod
    def positional_args(cls) -> list[str]:
        return ["message", "ref"]


class Perform(AutoCommand):
    """Perform an action, you must use the result to change an entity.

    Args:
        args: reference to task variable to perform
    """

    args: Optional[str] = None

    @classmethod
    def positional_args(cls) -> list[str]:
        return ["args"]


class Len(AutoCommand):
    """Return the lenght of the query results.

    Args:
        args: reference to query variable
    """

    args: Optional[str] = None

    @classmethod
    def positional_args(cls) -> list[str]:
        return ["args"]


class Next(AutoCommand):
    """Return the first entity of the query results.

    Args:
        args: reference to the entity
    """

    args: Optional[str] = None

    @classmethod
    def positional_args(cls) -> list[str]:
        return ["args"]


class EntityQuery(Command):
    entities: list[Entity] = Field(default_factory=list)

    def get_entity(self, idx: Optional[int]) -> Optional[Entity]:
        idx = idx or 0
        num_entities = len(self.entities)
        if num_entities > 1:
            if idx is None:
                raise ValueError("Multiple entities found")
            elif idx >= len(self.entities):
                raise ValueError("Index of entity out of bound")
            else:
                return self.entities[idx]
        elif num_entities == 0:
            return None
        else:
            return self.entities[0]


class Say(Command):
    """Call the response generation with the relevant variables.

    Args:
        args: the list of relevant variables to pass to the response generation system: x1, x2, etc.
    """

    args: Optional[str] = None

    @classmethod
    def positional_args(cls) -> list[str]:
        return ["args"]


class Contact(Command):
    """marks a name as a contactable person.

    Args:
        name:
    """

    name: str = None

    @classmethod
    def positional_args(cls) -> list[str]:
        return ["name"]


@dataclass
class CommandRegistry:
    commands: dict[str, Type[Command]] = field(default_factory=dict)

    def docs(self) -> str:
        def print_args(args: Dict[str, str], optionalise: bool = True) -> str:
            arg_list = []
            for arg_name, (arg_type, desc) in args.items():
                if arg_name in {"confirmed", "entity", "entities"}:
                    continue
                if optionalise and not arg_type.startswith("Optional["):
                    arg_type = f"Optional[{arg_type}]"
                arg_list.append(f"{arg_name}: {arg_type}, # {desc}")
            return "\n".join(arg_list)

        retval = []
        for k, v in self.commands.items():
            if issubclass(v, AutoCommand):
                continue
            doc = docstring_parser.parse(v.__doc__)
            annotations = inspect.get_annotations(v)

            param_str = print_args(
                {p.arg_name: (annotations[p.arg_name], p.description) for p in doc.params},
            )
            description = f" # {d}" if ((d := doc.short_description) is not None) else ""
            retval.append(f"{k}( # {description}\n{param_str}\n)")
        return retval


def build_registry() -> CommandRegistry:
    builders: list[Type[Command]] = [
        SendMessage,
        Say,
        Contact,
        OrderMeal,
        TimerQuery,
        SetTimer,
        CreateAlarm,
        Hint,
        Perform,
        Len,
    ]

    builder_dict = {builder.command_name(): builder for builder in builders}
    return CommandRegistry(commands=builder_dict)
