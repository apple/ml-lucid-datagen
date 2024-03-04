#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from textwrap import dedent
from typing import List

from jinja2 import Environment

from lucid_generate_data.openai_call import make_openai_call
from lucid_generate_data.utils.definitions import UserTurn
from lucid_generate_data.stage import StageExecutionException

environment = Environment()

NUM_GENERATION_ATTEMPTS = 3

prompt_template = environment.from_string(
    dedent(
        """
    You are trying to help a smart AI assistant (lucid) what a user wants. The smart AI needs help predicting what the slot values are that the user wants.

    Your job is to look at what the user said, and predict the most plausible slot value that accurately reflects what the user said. Every slot value must exactly match a subset of the user text (if it makes sense to do that). You only need to predict the string slot values (which are currently missing).

    If it doesn't make sense to exactly match a subset of the user text, then provide an answer as close to the original text as possible

    Each predicted slot value must be sub-strings of the user input.

    Examples:

    user: I want to order a cheese and bacon Dominos pizza
    lucid: order_dominos(pizza_type="")
    your suggestion: order_dominos(pizza_type="cheese and bacon")

    user: book an uber for 4 people from my home
    lucid: order_uber(no_people=4, pick_up_location="")
    your suggestion: order_uber(no_people=4, pick_up_location="home")

    user: I want delivery to be at midday, with delivered to 33 Carl Street
    lucid: x4.time, x4.delivery_address = "midday", "33 Carl Street"
    your suggestion: x4.time, x4.delivery_address = "midday", "33 Carl Street"

    user: Send it to my boss, John.
    lucid: x1.recipient = ""
    your suggestion: x1.recipient = "my boss, John"

    user: I want to play the that song [refering to 'Shape of you' mentioned earlier in the conversation]
    lucid: play_song(song="")
    your suggestion: play_song(song="Shape of you")

    user: Ask him if he is available for dinner
    lucid: x7.content=""
    your suggestion: x7.content="Are you available for dinner"

    user: {{user_utterance}}
    lucid: {{system_response}}
    your suggestion:
    """
    ).lstrip()
)


def check_str_responses_said_by_user(with_slots: str, user_string: str):
    in_slot_value = False
    slot_value = ""
    all_slots = []
    for char in with_slots:
        if char == '"':
            in_slot_value = not in_slot_value
            if not in_slot_value:
                all_slots.append(slot_value)
                slot_value = ""
        if in_slot_value:
            if char != '"':
                slot_value += char

    all_slots = [y for x in all_slots for y in x.split("|")]

    for value in all_slots:
        if value not in user_string:
            return False, "Slot: " + value + " is not in: " + user_string

    return True, None


def generate_slot_values(input_turns: list, system_response: str):
    """
    Populates string slot values
    """

    assert isinstance(input_turns[-1], UserTurn)

    user_utterance = input_turns[-1].query
    assert isinstance(user_utterance, str)

    prompt = prompt_template.render(
        user_utterance=user_utterance,
        system_response=system_response,
    )

    if user_utterance is None:
        raise StageExecutionException("Invalid slot values")

    with_slots = make_openai_call("get_slot_values", prompt=prompt)

    return with_slots
