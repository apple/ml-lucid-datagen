#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import asyncio
import subprocess
from typing import Any, Sequence

from jinja2 import Environment
import rich
from rich.prompt import Prompt as TerminalPrompt

from lucid_generate_data.utils.completer import Completer, Prompt
from lucid_generate_data.utils.definitions import (
    ActionResult,
    Inform,
    ProgramTurn,
    RequestConfirmation,
    RequestDisambiguation,
    RequestValue,
    Turn,
    LucidTurn,
    UserTurn,
    AutoTurn,
    AutoTransientTurn,
)

LUCID_PREFIX = "lucid: "
USER_PREFIX = "user: "


def copy_to_clipboard(text: str) -> None:
    process = subprocess.Popen("pbcopy", universal_newlines=True, stdin=subprocess.PIPE)
    process.communicate(text)


def conversation_to_text(turns: Sequence[Turn]) -> str:
    return "\n".join(map(format_turn, turns))


def format_turn(turn: Turn) -> str:
    match turn:
        case UserTurn(query):
            return f"{USER_PREFIX}{query}"
        case ProgramTurn(index, expression):
            return f"{index} {expression.rstrip('#y')}"
        case AutoTurn(index, expression):
            return f"{index} {expression}"
        case AutoTransientTurn(index, expression):
            return f"{index} {expression}"
        case LucidTurn(response):
            return f"{LUCID_PREFIX}{response}"

    raise ValueError("Must be a Turn")


def max_turn_index(turns: Sequence[Turn]) -> int:
    def _get_turn_index(turn: Turn) -> int:
        match turn:
            case ProgramTurn(index):
                return index
            case AutoTurn(index):
                return index
            case AutoTransientTurn(index):
                return index
            case _:
                return -1

    return max([_get_turn_index(turn) for turn in turns] + [-1])


def _recommendation_turn(result: ActionResult, next_index: int) -> AutoTransientTurn:
    match result.recommended_action:
        case RequestValue(dialogue, name):
            return AutoTransientTurn(
                index=next_index,
                expression=f'hint("ask for value: {dialogue}", ref=x{result.index})',
            )
        case RequestDisambiguation(dialogue, name, options):
            return AutoTransientTurn(
                index=next_index,
                expression=f'hint("choose between values: {dialogue}", ref=x{result.index})',
            )
        case RequestConfirmation(dialogue, name):
            return AutoTransientTurn(
                index=next_index,
                expression=f'hint("please confirm: {dialogue}", ref=x{result.index})',
            )


def _result_turn(result: ActionResult, next_index: int) -> AutoTurn:
    match result.result:
        case Inform(dialogue):
            return AutoTurn(index=next_index, expression=f"perform(x{result.index})")


def add_user_turn(turns: list[Turn], scenario: list[UserTurn]) -> list[Turn]:
    if scenario:
        turns.append(scenario.pop())
    else:
        user_utterance = TerminalPrompt.ask("user")
        turns.append(UserTurn(user_utterance))
    return turns


def generate_response(
    turns: list[Turn],
    prompt_template: Environment,
    completer: Completer,
    revealed_values: list[Any],
    example_conversations=str,
):
    prefix = prompt_template.render(
        conversation_text=conversation_to_text(turns),
        example_conversations=example_conversations,
        lucid_prompt=LUCID_PREFIX,
        result=revealed_values,
    )
    prompt = Prompt(prefix=prefix, stop_texts=["user:", "\n"])

    completion = asyncio.run(completer.complete(prompt))

    lucid_turn = LucidTurn(completion.strip())
    turns.append(lucid_turn)
    rich.print(f"{LUCID_PREFIX}{lucid_turn.response}")
