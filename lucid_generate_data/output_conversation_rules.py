#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import Dict, Any, List
from lucid_generate_data.conversation_rule import ConversationRule


def prepare_text_for_rule(
    intent: Dict[str, Any],
    selected_rule: ConversationRule,
    args: List[str],
    query_info: Dict[str, Any],
) -> str:
    """
    Populate arguments in unhappy path templates
    """

    text = selected_rule.template
    if len(args) > 0:
        for arg_id, arg_chosen in enumerate(args):
            text_to_replace = "{arg" + str(arg_id + 1) + "}"
            assert text_to_replace in selected_rule.template, text_to_replace
            text = text.replace(text_to_replace, arg_chosen)

    if selected_rule.label == "query":
        text = text.replace("{query}", query_info[intent["command"]]["query"])
        text = text.replace("{rationale}", query_info[intent["command"]]["rationale"])

    return text


def output_conversation_rules_current_intent(
    command: str,
    next_command: str,
    intent: Dict[str, Any],
    rules_to_be_applied: list,
    unhappy_path_args: list,
    query_info: Dict[str, Any],
    tags_already_seen_for_intent,
) -> Dict[str, str]:
    """
    We generate rules that the conversation should follow, including the desired final command
    """

    if not intent:
        return "The user now wants to end the conversation"

    selected_text_rules = []
    next_id = 1

    # Add rule giving current intent
    text = f"The task the user wants to perform must be: {command}."
    if "confirmation_required" in intent:
        if (
            intent["confirmation_required"]
            and "CANCEL" not in [rule.tag for rule in rules_to_be_applied]
            and "DELAY_CONFIRMATION" not in [rule.tag for rule in rules_to_be_applied]
        ):
            text += " This intent requires confirmation, include [CONFIRM] at the start of the text when giving confirmation."
        elif intent["query_intent"]:
            text += " This information must all be given in a single turn."

        if intent["confirmation_required"] and "DELAY_CONFIRMATION" in tags_already_seen_for_intent:
            text += " The user has decided to go ahead with this intent after initially not giving their confirmation."

    selected_text_rules.append(f"{next_id}) {text}")

    # Add unhappy path rules
    for i, rule in enumerate(rules_to_be_applied):
        if rule.tag not in tags_already_seen_for_intent:
            next_id += 1
            if unhappy_path_args:
                args = unhappy_path_args[i]
            else:
                args = []
            text = prepare_text_for_rule(intent, rule, args, query_info)
            selected_text_rules.append(f"{next_id}) {text}")

    # If there is another intent to follow
    if next_command:
        next_id += 1
        text = (
            "After this command is achieved, the user should start with a second request for the following command: "
            + str(next_command)
        )
        selected_text_rules.append(f"{next_id}) {text}")

    # We allocate unhappy path rules to intents
    assert len(selected_text_rules) == next_id

    rules = "\n".join(selected_text_rules)
    return rules
