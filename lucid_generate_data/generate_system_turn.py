#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from copy import deepcopy
from dataclasses import dataclass
from textwrap import dedent
from typing import List, Dict, Any
import json

import rich
from jinja2 import Environment
from rich.columns import Columns
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax

import os
from os import listdir
from os.path import isfile, join

from lucid_generate_data.utils.completer import OpenAiChatCompleter, Prompt

from lucid_generate_data.validate_with_tags import validation_from_tags

from lucid_generate_data.generate_str_slot_values import generate_slot_values
from lucid_generate_data.modelling_constants import STAGE_MODEL_LOOKUP
from lucid_generate_data.openai_call import (
    completer_with_llm_validation,
    completer_with_llm_cheating,
    completer_no_slot_values,
)
from lucid_generate_data.stage import StageExecutionException
from lucid_generate_data.utils.definitions import ActionResult, InformList, ProgramTurn, Turn

from lucid_generate_data.executor.demo import (
    _recommendation_turn,
    _result_turn,
    conversation_to_text,
    generate_response,
    max_turn_index,
)
from lucid_generate_data.executor.executor import ProgramExecutor
from lucid_generate_data.executor.prompt import prompt_template_nlg

environment = Environment()
response_prompt_template = prompt_template_nlg()

STOP_ON_ERROR = False
NO_TRIES_TO_AVOID_INVALID_COMMANDS = 10
VALIDATION_FOLDER = "lucid_generate_data/validation_issues/"
NUM_GENERATION_ATTEMPTS = 3

prompt_template = environment.from_string(
    dedent(
        """
    You are a smart AI assistant. Your task is to generate the next system command.

    Here are some examples of conversations to help you understand the format.

    {{all_examples}}

    Now it's your turn.  You must choose from one of the following commands:
    {% for definition in intent_definitions %}
    - {{definition}}
    {%- endfor %}
    {% if confirmation_required -%}
    - confirm - confirm an action, only used as `confirm(x)` where `x` is a task reference
    {% endif -%}
    - say - speak to the user. Always use this to ask the user for a value! Say can either reference a hint, a perform statement, or otherwise use say(). Never include a string inside say - use say() instead.

    {% if "find_" in " ".join(intent_definitions) %}
    If Lucid has just performed a query (an intent starting with 'find_'), and the user wants to perform an intent based on the query response, populate as many slots as possible based on the query response from Lucid.
    {%- endif %}

    If no relevant information has been provided by the user about an intent, slot-value, or confirmation, then reply with say(). You can only use say(x) when `x` refers to the last turn.

    When provided with a hint, you should follow it. NEVER predict hint yourself. If the user mentions a slot, but they don't actually give the value for the slot, you should reply with say(). E.g. if the user says 'How about [slot name]?'

    If asking the user for confirmation, Lucid must explicitly use the word 'confirm' or 'confirmation'.

    {% if not confirmation_required %}
    - Do NOT issue a confirm() command
    {% endif %}

    {%- if incomplete %} Bear in mind there are pending tasks that need to be completed:
    {%- for inc in incomplete %} {{ inc }}
    {%- if loop.revindex0==1 %} and {%- elif not loop.last %}, {%- endif -%}
    {% endfor -%}
    {% endif %}

    {{special_guidance}}

    Conversation:
    {{conversation_text}}
    {%- if result %}
    {{result}}
    {%- endif %}
    {%- if recommendation %}
    {{recommendation}}
    {%- endif %}

    {{next_index}}
    """
    ).lstrip()
)


@dataclass
class ResultTurn(Turn):
    result: str


def ref_last_hint_only(index: int, original_response_with_slots: str):
    if original_response_with_slots == "say()":
        return True, None

    if original_response_with_slots.startswith("say("):
        if original_response_with_slots == "say(x" + str(index) + ")":
            return True, None

    return (
        False,
        "Should reference say(x"
        + str(index)
        + ") but instead references: "
        + original_response_with_slots,
    )


def perform_validation(
    first_system_turn: bool,
    original_response: str,
    original_response_with_slots: str,
    prompt: Prompt,
    completer: OpenAiChatCompleter,
    input_turns: List[Turn],
    intent_definitions: List[str],
    conversation_rules: str,
    tags_extracted: list,
    last_turn,
) -> str:

    error_dict = {}
    error_checks = []

    error_dict.update(
        {
            "1st llm validation": completer_with_llm_validation(
                first_system_turn, prompt, completer, original_response, conversation_rules
            )
        }
    )

    error_dict.update(
        {
            "2nd llm validation": completer_with_llm_validation(
                first_system_turn, prompt, completer, original_response, conversation_rules
            )
        }
    )

    error_dict.update(
        {
            "cheating llm validation": completer_with_llm_cheating(
                first_system_turn, prompt, completer, conversation_rules, original_response
            )
        }
    )

    error_dict.update(
        {
            "tag validation": validation_from_tags(
                first_system_turn, original_response_with_slots, tags_extracted
            )
        }
    )

    if not first_system_turn:
        error_dict.update(
            {
                "only referencing last hint": ref_last_hint_only(
                    last_turn.index, original_response_with_slots
                )
            }
        )

    error_present = False
    for error_type, error_response in error_dict.items():
        if not error_response[0]:
            print("Error:", error_type, "with details:", error_response)
            error_present = True

    error_dict["all_turns"] = conversation_to_text(input_turns)
    error_dict["lucid_prediction"] = original_response_with_slots
    error_dict["conversation_rules"] = conversation_rules
    error_dict["intent_definitions"] = intent_definitions

    if error_present:
        file_name = save_validation_results(error_dict)
        if STOP_ON_ERROR:
            raise StageExecutionException(
                "Validation has identified an issue with this turn. Validation result saved in: "
                + file_name
            )
    else:
        file_name = None

    return file_name


def save_validation_results(error_dict: Dict[str, Any]) -> str:
    mypath = VALIDATION_FOLDER
    if not os.path.exists(VALIDATION_FOLDER):
        os.makedirs(VALIDATION_FOLDER)

    all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    if all_files:
        all_file_ints = [int(file[2:-5]) for file in all_files]
        if all_file_ints:
            max_error_number = max(all_file_ints)
        else:
            max_error_number = 0

        new_file_name = "e." + str(max_error_number + 1) + ".json"
    else:
        new_file_name = "e.0.json"

    with open(mypath + new_file_name, "w") as json_file:
        json.dump(error_dict, json_file, indent=4)

    return new_file_name


def _format_examples(list_of_examples: List[str]) -> str:
    full_str = ""

    for i, example in enumerate(list_of_examples):
        full_str += "<example " + str(i) + ">"
        full_str += example
        full_str += "</example " + str(i) + ">\n\n"

    return full_str


def generate_system_turn(
    input_turns: List[Turn],
    ssa_examples: List[str],
    executor: ProgramExecutor,
    intent_definitions: List[str],
    confirmation_required: bool,
    conversation_rules: str,
    tags_extracted: List[str],
    special_guidance: str,
) -> List[Turn]:
    # Make a copy to account for generation failure
    turns = deepcopy(input_turns)
    SHOW_PROMPT = True

    all_examples_str = _format_examples(ssa_examples)

    completer = OpenAiChatCompleter(
        model_name=STAGE_MODEL_LOOKUP["lucid_agent"]["model_name"],
        max_tokens=STAGE_MODEL_LOOKUP["lucid_agent"]["max_tokens"],
        temperature=STAGE_MODEL_LOOKUP["lucid_agent"]["temperature"],
    )

    program = [turn for turn in turns if isinstance(turn, ProgramTurn)]

    next_index = max_turn_index(turns) + 1

    last_value = ActionResult(index=next_index)
    for turn in program:
        last_value = executor.execute_turn(turn)

    num_system_turns = 0
    revealed_values = None
    while True:
        next_index = max_turn_index(turns) + 1
        if last_value.recommended_action is not None:
            new_turn = _recommendation_turn(last_value, next_index)
            last_value = executor.execute_turn(new_turn)
            turns.append(new_turn)
            continue

        if last_value.result is not None:
            if isinstance(last_value.result, InformList):
                revealed_values = last_value.result.items
            new_turn = _result_turn(last_value, next_index)
            last_value = executor.execute_turn(new_turn)
            turns.append(new_turn)
            continue

        incomplete = executor.state.get_incomplete_tasks()

        prefix = prompt_template.render(
            conversation_text=conversation_to_text(turns),
            next_index=next_index,
            incomplete=incomplete,
            intent_definitions=intent_definitions,
            confirmation_required=confirmation_required,
            all_examples=all_examples_str,
            special_guidance=special_guidance,
        )
        prompt = Prompt(prefix=prefix, stop_texts=["user:", "\n"])

        if SHOW_PROMPT:
            rich.print(Panel(escape("\n".join(prefix.split("\n")))))

        first_system_turn = num_system_turns == 0

        for i in range(NUM_GENERATION_ATTEMPTS):
            predicted_output_no_values = completer_no_slot_values(
                first_system_turn, prompt, completer, conversation_rules
            )

            if '"' in predicted_output_no_values:
                predicted_output = generate_slot_values(input_turns, predicted_output_no_values)
            else:
                predicted_output = predicted_output_no_values

            error_file = perform_validation(
                first_system_turn,
                predicted_output_no_values,
                predicted_output,
                prompt,
                completer,
                input_turns,
                intent_definitions,
                conversation_rules,
                tags_extracted,
                turns[-1],
            )
            program_turn = ProgramTurn(
                index=next_index, expression=predicted_output.strip(), errors=error_file
            )
            try:
                last_value = executor.execute_turn(program_turn)
                rich.print(
                    Columns([str(program_turn.index), Syntax(program_turn.expression, "python")])
                )
                if last_value.recommended_action is not None:
                    rich.print(
                        Columns(
                            [
                                str(next_index + 1),
                                Syntax(str(last_value.recommended_action), "python"),
                            ]
                        )
                    )
                break
            except Exception as e:
                if i == NUM_GENERATION_ATTEMPTS:
                    raise StageExecutionException(
                        f"Invalid SSA failed execution after {NUM_GENERATION_ATTEMPTS} attempts. On final attempt, failed with the following error: {e}"
                    )

        turns.append(program_turn)
        num_system_turns += 1
        if program_turn.expression.startswith("say"):
            generate_response(
                turns=turns,
                prompt_template=response_prompt_template,
                completer=completer,
                revealed_values=revealed_values,
                example_conversations=all_examples_str,
            )
            revealed_values = None
            break

        if num_system_turns > 3:
            print("Too many system turns, breaking")
            break
    return turns
