#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from os import listdir
from os.path import isfile, join

STRING_REPLACE_CONVERSATION_INTERRUPTED = [
    "Got to go, let's finish this later",
    "Sorry got to dash",
    "Sorry I have to go",
    "Got to go",
    "We'll do this later",
    "Need to stop and do something else instead",
    "Sorry, I need to run",
    "I have to go, let's continue this later",
    "Gotta go, catch you later",
    "Sorry, I'll have to finish this later",
    "I need to go, let's pick this up later",
    "Apologies, I have to go",
    "Have to go, let's talk later",
    "I'm running late, we'll talk another time",
    "Sorry, I have to cut this short",
    "I need to stop and go, we'll continue later",
    "I have to go now, we'll finish this another time",
    "Sorry, something urgent came up, talk to you later",
    "I'll have to continue this conversation later",
    "Have to go now, we'll revisit this later",
    "I gotta run, we'll finish this conversation later",
    "Sorry, I need to step away, talk to you later",
    "I have to go, we'll wrap this up later",
    "I need to go, we'll finish this conversation at another time",
    "Sorry, I have to go now, we'll continue this later",
    "Apologies, I have to leave now",
    "I have to take care of something, we'll continue later",
    "Sorry, gotta cut this short, talk to you soon",
    "I need to step away for a bit, let's finish this later",
    "I'll have to pause our conversation for now",
    "I have to attend to something, we'll talk later",
    "I can't continue now, we'll pick this up later",
    "Sorry, I need to go, we'll resume our conversation later",
    "I have to leave for now, we'll continue this later",
    "Sorry, I need to wrap this up, we'll talk again later",
    "I need to go, catch you later",
    "I must go, let's finish this conversation later",
    "I have to sign off for now, we'll talk later",
    "Sorry, need to go, we'll continue later",
    "I have to pause this, we'll continue later",
    "I'm heading out, we'll finish this conversation later",
    "I'll have to stop here, we'll continue later",
    "I have to go now, we'll talk later",
    "I must step away, we'll continue later",
    "I have to go, we'll pick this up later",
    "Sorry, I've got to run, we'll catch up later",
    "I have to step out, let's continue this later",
    "I need to take off now, we'll talk later",
    "Sorry, something urgent has come up, we'll chat later",
    "I've got to go, we'll finish this later",
    "I need to go, we can continue this later",
    "I'm on a tight schedule, let's continue this later",
    "I have to go, we'll revisit this later",
    "I'll need to wrap this up, we'll talk later",
    "I have to leave, we'll pick this up later",
    "Sorry, I need to cut this short, we'll continue later",
    "I have to attend to something, let's finish this later",
    "I need to go now, we'll chat later",
    "I've got to head off, we'll continue later",
    "Something has come up, we'll talk later",
    "I have to step away, we'll continue this later",
    "I'm pressed for time, we'll continue later",
    "I need to go, let's pick this up later",
    "I'll have to continue this later",
]


def _get_last_hint_in_conversation(turn, last_hint_in_conversation):
    """
    Get last hint
    """
    if turn["author"] == "AutoTransientTurn":
        if "ask for value" in turn["expression"]:
            hint = turn["expression"][: turn["expression"].rfind(" ") - 1]
            slot = hint[hint.rfind(" ") + 1 : -1]
            last_hint_in_conversation = slot

    if turn["author"] == "System":
        if "say(" not in turn["expression"]:
            last_hint_in_conversation = "NONE"

    return last_hint_in_conversation


def _check_answer_other_slot(turn, last_hint_in_conversation):
    """
    Check if HINT is being used
    """
    if "say(" not in turn["expression"]:
        if last_hint_in_conversation not in turn["expression"]:
            if last_hint_in_conversation != "NONE" and "=" in turn["expression"]:
                return True

    return False


def _find_slots_now_populated(turn):
    """
    Identify slots that have been populated
    """
    slots_now_populated = {}

    all_intents = find_all_intents_and_query_intents()

    for intent in all_intents:
        if intent + "(" in turn["expression"]:
            split_of_slots = turn["expression"][turn["expression"].find("(") + 1 : -1].split(",")
            slots_populated = [x[: (x.find("="))].strip() for x in split_of_slots]
            slots_populated = [x for x in slots_populated if x != ""]
            index = turn["index"]

            slots_now_populated[str(index)] = slots_populated

    for index in range(999):
        if "x" + str(index) + "." in turn["expression"]:
            remaining_str = turn["expression"]

            while "x" + str(index) + "." in remaining_str:
                # Look after x3. type expression to find the slot name...
                start_text = remaining_str.find("x" + str(index) + ".") + 2 + len(str(index))
                remaining_str = remaining_str[start_text:]

                # We try looking at the first slot name (we have slotname=...., so we look up to '=')
                if "=" in remaining_str:
                    end_text = remaining_str.find("=")
                else:
                    end_text = len(remaining_str)

                # Check the slot name
                command = remaining_str[:end_text].strip()

                # We create a list of the slots populated in this turn
                if command != "":
                    if str(index) not in slots_now_populated:
                        slots_now_populated[str(index)] = [command]
                    else:
                        slots_now_populated[str(index)].append(command)
                # We continue to iterate through the expression for the turn
                remaining_str = remaining_str[end_text:]

    return slots_now_populated


def find_all_intents_and_query_intents():
    # We find all our intents (minus held out intents)
    mypath = "lucid_v1.0/toolbox_intents"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # We also find our heldout intents
    mypath = "lucid_v1.0/toolbox_intents_heldout"
    files_heldout = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # We outout a list of the intent names
    all_files = files + files_heldout
    all_intents = [x.replace(".json", "") for x in all_files]

    return all_intents


def _check_correction(new_slots_populated, slots_already_populated):
    """
    We identify corrections
    """
    for intent_index in new_slots_populated:
        if intent_index in slots_already_populated:
            for slot in new_slots_populated[intent_index]:
                if slot in slots_already_populated[intent_index]:
                    return True

    return False


def add_select_system_tags(conversation):
    """
    We add in our unhappy paths tags. Some are provided by the user tags, while others are calculated
    """

    no_system_turn = -1
    slots_already_populated = {}
    last_hint_in_conversation = "NONE"

    turn_tags = []

    # The first turn can not include any of the derived scenario tags
    answer_other_slot = False
    correction = False

    for turn in conversation["turns"]:
        # We find tags at a user turn level, resetting at each user turn
        if turn["author"] == "User":
            turn_tags.append(
                {"derived_answer_other_slot": answer_other_slot, "derived_correction": correction}
            )

            answer_other_slot = False
            correction = False

        # We identify when unhappy paths are present in system turns
        # .. (the turns preceeding the user turn when the tags are added)
        if turn["author"] == "System":
            if _check_answer_other_slot(turn, last_hint_in_conversation):
                answer_other_slot = True

            # Checking for corrections
            new_slots_populated = _find_slots_now_populated(turn)
            if _check_correction(new_slots_populated, slots_already_populated):
                correction = True

            # We now combine together the two dictionaries
            for intent_index, slot_names in new_slots_populated.items():
                if intent_index in slots_already_populated:
                    slots_already_populated[intent_index] += new_slots_populated[intent_index]
                else:
                    slots_already_populated[intent_index] = new_slots_populated[intent_index]

        # Get last hint
        last_hint_in_conversation = _get_last_hint_in_conversation(turn, last_hint_in_conversation)

    # We add in these calculated unhappy path tags
    user_turn = 1
    for turn_no in range(len(conversation["turns"])):
        if conversation["turns"][turn_no]["author"] == "User":
            if ("end conversation" not in conversation["turns"][turn_no]["query"].lower()) and (
                "conversation interrupted" not in conversation["turns"][turn_no]["query"].lower()
            ):
                for key, value in turn_tags[user_turn].items():
                    if value:
                        conversation["turns"][turn_no]["tags"][0].append(key)

                user_turn += 1

    return conversation
