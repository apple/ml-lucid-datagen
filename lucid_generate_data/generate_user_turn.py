#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import asyncio
from copy import deepcopy
from textwrap import dedent
from typing import List

import rich
from jinja2 import Environment
from rich.markup import escape
from rich.panel import Panel

from lucid_generate_data.modelling_constants import STAGE_MODEL_LOOKUP
from lucid_generate_data.utils.completer import OpenAiChatCompleter, Prompt
from lucid_generate_data.stage import StageExecutionException
from lucid_generate_data.utils.definitions import AppContext, LucidTurn, Turn, UserTurn

environment = Environment()

NUM_GENERATION_ATTEMPTS = 3

prompt_template = environment.from_string(
    dedent(
        """
    You are a smartphone user talking to a smart AI assistant (Lucid). Your conversation will centre around the commands that are supported by the AI assistant:

    {% for definition in intent_definitions %}
    - {{definition}}
    {%- endfor %}
    
    When the conversation is finished, dismiss the assistant by saying "end conversation".
    
    Here is an example conversation:
    {{example}}
    Now it's your turn. Your task is to generate the next user input:
    
    # Conversation rules:
    {{conversation_rules}}
    
    # Generated conversation:
    {{conversation_text}}
    user:
    """
    ).lstrip()
)


def format_dialogue(turns: List[Turn]) -> str:
    filtered_turns = []

    for turn in turns:
        if isinstance(turn, UserTurn):
            filtered_turns.append("user: " + turn.query)
        elif isinstance(turn, LucidTurn):
            filtered_turns.append("lucid: " + turn.response)

    conv_str = "\n".join(filtered_turns)

    return conv_str


def is_valid_user_turn(generated_text: str):
    return "lucid" not in generated_text.lower() and "say(" not in generated_text.lower()


def _format_examples(list_of_examples: List[str]):
    full_str = ""

    for i, example in enumerate(list_of_examples):
        full_str += "<example " + str(i) + ">"
        full_str += example
        full_str += "</example " + str(i) + ">\n\n"

    return full_str


def generate_user_turn(
    input_turns: List[Turn],
    examples: List[str],
    intent_definitions: List[str],
    conversation_rules: str,
    confirmation_required: bool,
    app_context: AppContext,
):
    examples_str = _format_examples(examples)

    turns = deepcopy(input_turns)
    SHOW_PROMPT = True

    completer = OpenAiChatCompleter(
        model_name=STAGE_MODEL_LOOKUP["user_agent"]["model_name"],
        max_tokens=STAGE_MODEL_LOOKUP["user_agent"]["max_tokens"],
        temperature=STAGE_MODEL_LOOKUP["user_agent"]["temperature"],
    )

    prefix = prompt_template.render(
        example=examples_str,
        conversation_text=format_dialogue(turns),
        intent_definitions=intent_definitions,
        conversation_rules=conversation_rules,
        confirmation_required=confirmation_required,
        app_context=app_context,
    )
    prompt = Prompt(prefix=prefix, stop_texts=["\n"])

    if SHOW_PROMPT:
        rich.print(Panel(escape("\n".join(prefix.split("\n")))))

    user_utterance = None
    for i in range(NUM_GENERATION_ATTEMPTS):
        # need to strip the user: prefix as this gets added back by demo.conversation_to_text
        generated_text = asyncio.run(completer.complete(prompt, use_cache=False))
        if generated_text.startswith("user:"):
            generated_text = generated_text[len("user:") :].strip()
        if is_valid_user_turn(generated_text):
            # We remove quotes, to prevent the model putting quotes around string slot values
            generated_text = generated_text.replace('"', "")
            user_utterance = generated_text
            break

    if user_utterance is None:
        raise StageExecutionException("Invalid user response (this was given as None)")

    turns.append(UserTurn(query=user_utterance, tags=[]))
    return turns
