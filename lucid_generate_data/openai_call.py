#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import asyncio

from lucid_generate_data.modelling_constants import STAGE_MODEL_LOOKUP
from lucid_generate_data.utils.completer import OpenAiChatCompleter, Prompt


def _update_float_format(command: str):
    """
    We update float formats to make these consistent
    """

    number_started = False
    decimal_started = False
    last_char = ""

    new_command = ""
    number_str = ""

    for char in command:
        # We find numbers (and when these include decimals)
        if char.isdigit() and last_char == "=":
            number_started = True
        if number_started and char == ".":
            decimal_started = True

        # Checking no invalid values e.g. 4..0
        if number_started and char == ".":
            assert last_char.isdigit()

        # After the number is finished, we format it
        if not char.isdigit() and char != ".":
            if decimal_started:
                new_command += str(float(number_str))
            elif number_started:
                new_command += str(int(number_str))

            # We start again
            number_str = ""
            number_started = False
            decimal_started = False

        if not number_started:
            new_command += char
        else:
            number_str += char

        last_char = char

    if decimal_started:
        new_command += str(float(number_str))
    elif number_started:
        new_command += str(int(number_str))

    return new_command


def _format_values(first_system_turn: bool, answer: str) -> str:
    """
    We remove strs, sort multiple slot assignment, and format floats
    """

    # Removing str slot predictions
    updated_answer = ""
    brackets_opened = False
    for char in answer:
        if char == '"':
            brackets_opened = not brackets_opened
            updated_answer += char
        else:
            if not brackets_opened:
                updated_answer += char

    updated_answer = updated_answer.replace(" ", "")

    # We sort cases of mulitple assignment
    updated_answer = _order_args(updated_answer)

    # We update float formats to make these consistent
    updated_answer = _update_float_format(updated_answer)

    # We additional fix the following (potential) issue:
    # ..if the first system turn starts with say(, it must be say()
    if first_system_turn:
        if updated_answer.startswith("say("):
            updated_answer = "say()"

    return updated_answer


def _reorder_multiplie_assignment(command: str):
    """
    We sort cases of multiple assignment, preventing ordering disagreemnts
    """

    # We consider assignment after the first intent
    if command[-1] != ")":
        if "=" in command:
            # We parse the slots and values
            slots = []
            values = []

            in_progress = ""
            before_eql_sign = True
            str_quotes_open = False

            for char in command:
                if char == '"':
                    str_quotes_open = not str_quotes_open

                if (char == "," or char == "=") and not str_quotes_open:
                    if before_eql_sign:
                        slots.append(in_progress)
                    else:
                        values.append(in_progress)

                    in_progress = ""

                if char == "=" and not str_quotes_open:
                    before_eql_sign = False

                if (char != "," and char != "=") or str_quotes_open:
                    in_progress += char

            values.append(in_progress)

            values = [x.strip() for x in values]
            slots = [x.strip() for x in slots]
            assert len(values) == len(slots)

            # Now we re-order these....
            combined = [(slots[i], values[i]) for i in range(len(values))]
            combined = sorted(combined)
            values = []
            slots = []
            for x in combined:
                values.append(x[1])
                slots.append(x[0])

            # We now create a new str for the command
            new_command = ", ".join(slots) + " = " + ", ".join(values)
            return new_command

    return command


def _order_args(a1: str):
    """
    Sorting assignments
    """

    if "(" in a1 and a1[-1] == ")":
        start = a1[: a1.find("(") + 1]
        a1 = a1[a1.find("(") + 1 : -1]
        a1 = a1.split(",")
        a1 = [x.strip() for x in a1]
        a1 = sorted(a1)
        a1 = ", ".join(a1)
        a1 = start + a1 + ")"
    else:
        a1 = _reorder_multiplie_assignment(a1)

    a1 = a1.strip()

    return a1


def _resolve_differences(response_main: str, response_valid: str) -> str:
    """
    We state the disagreement between LLMs
    """

    if response_main == response_valid:
        return True, None
    else:
        return False, (
            "No agreement between three LLMs. \n Main LLM gives: "
            + response_main
            + "\n Validation LLM gives: "
            + response_valid
        )


def completer_with_llm_validation(
    first_system_turn: bool,
    prompt: Prompt,
    completer: OpenAiChatCompleter,
    response_main: str,
    conversation_rules,
) -> str:
    response_valid = asyncio.run(completer.complete(prompt, use_cache=False))

    assert response_valid is not None

    return _resolve_differences(response_main, _format_values(first_system_turn, response_valid))


def completer_no_slot_values(
    first_system_turn: bool, prompt: Prompt, completer: OpenAiChatCompleter, conversation_rules
) -> str:
    response_main = asyncio.run(completer.complete(prompt, use_cache=False))

    return _format_values(first_system_turn, response_main)


def completer_with_llm_cheating(
    first_system_turn: bool,
    prompt: Prompt,
    completer: OpenAiChatCompleter,
    conversation_rules: str,
    response_main: str,
) -> str:
    prompt.prefix = prompt.prefix.replace(
        "Conversation:",
        "To help you make the right prediction, here is some more information. The user is following these instructions throughout the conversation:\n"
        + conversation_rules
        + "\nHowever, you should only make predictions about what the user has explicitly said. Do not include slot names or slot values unless the user has explicitly mentioned these. \n\nConversation:",
    )

    response_cheating = asyncio.run(completer.complete(prompt, use_cache=False))
    assert response_cheating is not None

    if _format_values(first_system_turn, response_cheating) == response_main:
        return True, None
    else:
        return False, (
            "No match. LLM model gives: "
            + _format_values(first_system_turn, response_cheating)
            + " while original response is: "
            + response_main
        )

    return _format_values(first_system_turn, response_cheating)


def make_openai_call(stage_name: str, prompt: str) -> str:
    assert stage_name in STAGE_MODEL_LOOKUP

    model_dict = STAGE_MODEL_LOOKUP[stage_name]

    completer = OpenAiChatCompleter(
        model_name=model_dict["model_name"],
        max_tokens=model_dict["max_tokens"],
        temperature=model_dict["temperature"],
    )

    test_str = asyncio.run(completer.complete(Prompt(prefix=prompt), use_cache=False))

    return test_str
