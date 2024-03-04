#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import random
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Any
from utils_compile_data import add_select_system_tags, STRING_REPLACE_CONVERSATION_INTERRUPTED


ERROR_FOLDER = "lucid_generate_data/validation_issues"
PATH_CONVERSATIONS = "lucid_generate_data/saved_conversations"
FIX_ERRORS = True
SPLITS = {"pop": ["train", "dev", "test"], "weights": [0.8, 0.1, 0.1]}
MIN_TURNS_TO_RETRIEVE_CONVO = 10
UNHAPPY_PATHS_BEFORE_SAY = [
    "EARLY_END",
    "IRRELEVANT",
    "CANCEL",
    "SARCASTIC",
    "DELAY_CONFIRMATION",
    "OVERHEARD",
]


def find_conversations() -> List[str]:
    """
    Find the location of all saved conversation files
    """
    all_files = []

    path = PATH_CONVERSATIONS
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for file in files:
        json_file = path + "/" + file

        all_files.append(json_file)

    return all_files


def extract_conversation(json_file: str) -> Dict[str, Any]:
    """
    Load a conversation from file location
    """
    with open(json_file, "r") as f:
        conversation = json.load(f)

    return conversation


def update_interrupted_lines(exp: str) -> str:
    """
    We provide a more natural language ending to conversations that did not complete
    """
    if "conversation interrupted" in exp.lower():
        assert exp == "conversation interrupted", exp
        new_str = random.choice(STRING_REPLACE_CONVERSATION_INTERRUPTED)
        new_str += ". End conversation."

    else:
        new_str = exp

    return new_str


def store_conversation(conversation: Dict[str, Any], split: str, i: int) -> Dict[str, Any]:
    """
    We reformat the conversation
    """

    saved_conversation = {"turns": [], "split": split, "_id": "LUCID_" + str(i)}

    tags = []

    for turn in conversation["turns"][3:]:
        if turn["author"] == "User":
            tags = turn["tags"]
            tags = tags[0]

            saved_conversation["turns"].append(
                {"author": "User", "query": update_interrupted_lines(turn["query"])}
            )

        elif turn["author"] == "System":
            saved_conversation["turns"].append(
                {
                    "author": "System",
                    "expression": turn["expression"],
                    "index": turn["index"],
                    "unhappy_paths": tags,
                }
            )

        elif turn["author"] == "Response":
            saved_conversation["turns"].append({"author": "Response", "text": turn["response"]})

        elif turn["author"] == "AutoTransientTurn":
            saved_conversation["turns"].append(
                {
                    "author": "Signal",
                    "dialog": turn["expression"],
                    "index": turn["index"],
                }
            )

        elif turn["author"] == "AutoTurn":
            saved_conversation["turns"].append(
                {
                    "author": "Signal",
                    "dialog": turn["expression"],
                    "index": turn["index"],
                }
            )

    return saved_conversation


def get_all_tags(conversation: Dict[str, Any]) -> List[str]:
    """
    Extract any unhappy paths in the conversation
    """

    all_tags = []

    for turn in conversation["turns"]:
        if turn["author"] == "User":
            all_tags += turn["tags"][0]

    all_tags = list(set(all_tags))
    return all_tags


def identify_predictions_of_hint(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """
    System labels should never predict a hint
    """
    for turn in conversation["turns"]:
        if turn["author"] == "System":
            if "hint" in turn["expression"].lower():
                turn["turn_errors"].append("post_processing_hint_prediction")

    return conversation


def identify_empty_str_predictions(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """
    string slots should never be empty
    """

    for turn in conversation["turns"]:
        if turn["author"] == "System":
            if '=""' in turn["expression"]:
                turn["turn_errors"].append("post_processing_empty_str")
            elif "=''" in turn["expression"]:
                turn["turn_errors"].append("post_processing_empty_str")

    return conversation


def identify_intents_from_list(intent_list: List[str], turn: Dict[str, Any]) -> List[str]:
    """
    Identify the intents present in a turn
    """

    intents_present = []

    for intent in intent_list:
        if intent in turn["expression"]:
            intents_present.append(intent)

    return intents_present


def get_heldout_intents() -> List[str]:
    """
    We get a list of only our heldout intents
    """

    # We also find our heldout intents
    mypath = "lucid_v1.0/toolbox_intents_heldout"
    all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    all_intents = [x.replace(".json", "") for x in all_files]

    return all_intents


def find_all_intents_and_query_intents() -> List[str]:
    """
    Get all the intents in our toolbox
    """

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


def find_all_conversation_intents(conversation: Dict[str, Any]) -> List[str]:
    """
    Find all intents in a conversation
    """

    all_intents = find_all_intents_and_query_intents()

    intents_included = []
    query_intents_included = []

    for turn in conversation["turns"]:
        if turn["author"] == "System":
            intents_included += identify_intents_from_list(all_intents, turn)

    intents_included = list(set(intents_included))

    return intents_included


def append_errors(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract any validation errors found in the conversation
    """
    turn_errors = []

    for turn in conversation["turns"]:
        turn_errors = []
        if turn["author"] == "System":
            if turn["errors"]:
                with open(ERROR_FOLDER + "/" + turn["errors"], "r") as f:
                    error_dict = json.load(f)

                for error_layer in ERROR_LAYERS:
                    if error_dict[error_layer][0]:
                        turn_errors.append(error_layer)
            turn["turn_errors"] = turn_errors

    return conversation


def search_for_checkpoint(turn: Dict[str, Any], checkpoint_reached: bool) -> bool:
    """
    Finding completed parts of the conversation before an error
    """

    # We search for the next checkpoint when we can save the conversation
    if turn["author"] == "System":
        if turn["expression"].startswith("find_"):
            checkpoint_reached = True

    elif turn["author"] == "AutoTurn":
        if turn["expression"].startswith("perform("):
            checkpoint_reached = True

    return checkpoint_reached


def check_corrections_intended(conversation: Dict[str, Any]) -> bool:
    """
    We only want slot corrections when this was intentional
    """

    filter_conversation = False
    for turn in conversation["turns"][2:]:
        if turn["author"] == "User":
            if ("CORRECTION" not in turn["tags"][0]) and "derived_correction" in turn["tags"][0]:
                filter_conversation = True

    return filter_conversation


def truncate_to_avoid_errors(conversation: Dict[str, Any]) -> (Dict[str, Any], bool):
    """
    We truncate conversations that contain errors
    """

    checkpoint_reached = False
    errors_present = False
    index_of_last_good_turn = 0
    user_turns_before_error = 0

    # Once a checkpoint is reached, we can 'save' all turns up to the next User turn
    for turn_idx, turn in enumerate(conversation["turns"]):
        if turn["author"] == "System":
            if turn["turn_errors"]:
                errors_present = True

        # We search for the next checkpoint when we can save the conversation
        if not errors_present:
            checkpoint_reached = search_for_checkpoint(turn, checkpoint_reached)

        # We want to know the last User turn when no errors were present
        if turn["author"] == "User" and not errors_present:
            user_turns_before_error = turn_idx

        # We find the user turn after this checkpoint (up to here is our last good turn)
        if checkpoint_reached and turn["author"] == "User" and not errors_present:
            index_of_last_good_turn = turn_idx

            # We reset, ready for the next checkpoint
            checkpoint_reached = False

    # If we cannot retrieve conversions that have reached a checkpoint, we try another strategy
    if (
        index_of_last_good_turn == 0
        and errors_present
        and user_turns_before_error > MIN_TURNS_TO_RETRIEVE_CONVO
    ):
        assert conversation["turns"][user_turns_before_error]["author"] == "User"
        conversation["turns"] = conversation["turns"][: user_turns_before_error + 1]
        conversation["turns"][-1]["query"] = "conversation interrupted"
        conversation["turns"][-1]["tags"] = [[]]

    elif index_of_last_good_turn == 0:
        conversation["turns"] = []

    elif errors_present:
        assert conversation["turns"][index_of_last_good_turn]["author"] == "User"
        conversation["turns"] = conversation["turns"][: index_of_last_good_turn + 1]
        conversation["turns"][-1]["query"] = "end conversation"
        conversation["turns"][-1]["tags"] = [[]]

    return conversation, errors_present


def check_say_after_specific_unhappy_paths(conversation: Dict[str, Any]) -> bool:
    """
    Validation to make sure specific unhappy paths are followed by say()
    """

    filter_example = False
    for turn in conversation["turns"]:
        if turn["author"] == "System":
            for tag in UNHAPPY_PATHS_BEFORE_SAY:
                if "expression" in turn:
                    if tag.lower() in turn["unhappy_paths"]:
                        if turn["expression"] != "say()":
                            filter_example = True

    return filter_example


def get_split(intents_present: List[str], test_intents: List[str]) -> str:
    """
    Assign a conversation to train/dev/test/test_ood
    """
    split = ""
    contains_id = False
    contains_ood = False
    for intent in intents_present:
        if intent in test_intents:
            contains_ood = True
        if intent not in test_intents:
            contains_id = True
    if contains_id and contains_ood:
        split = "exclude"
    if contains_ood:
        split = "test_ood"
    else:
        split = random.choices(SPLITS["pop"], weights=SPLITS["weights"])[0]
    return split


if __name__ == "__main__":
    random.seed(42)
    conversation_files = find_conversations()

    total_system_predictions = {}
    total_intents_present = {"train": {}, "dev": {}, "test": {}, "test_ood": {}, "exclude": {}}
    full_conversations = []
    valid_conversation_idx = 0

    # We reformat conversations, saving appropriate conversations
    for i, conversation_file in enumerate(conversation_files):

        conversation = extract_conversation(conversation_file)
        conversation = append_errors(conversation)

        # We need to check that the conversation does not contain a question by the user, not followed by a query intent
        # An analysis of the data has shown this is an area where we see data quality issues
        conversation = identify_predictions_of_hint(conversation)
        conversation = identify_empty_str_predictions(conversation)

        # We remove the part of the conversation that contains errors
        conversation, error_present = truncate_to_avoid_errors(conversation)

        # We now consider the truncated conversation without errors
        if not conversation["turns"]:
            continue

        elif error_present:
            if not FIX_ERRORS:
                continue

        conversation = add_select_system_tags(conversation)
        intents_present = find_all_conversation_intents(conversation)

        # We have different intents for our OOD test set
        test_intents = get_heldout_intents()
        conversation["split"] = get_split(intents_present, test_intents)

        # We save conversations that pass two post-processing checks
        filter_from_correction_post_processing = check_corrections_intended(conversation)
        if conversation["turns"] and not filter_from_correction_post_processing:
            saved_conv = store_conversation(
                conversation, conversation["split"], valid_conversation_idx
            )

            filter_from_say_post_processing = check_say_after_specific_unhappy_paths(saved_conv)

            if not filter_from_say_post_processing:
                full_conversations.append(saved_conv)
                valid_conversation_idx += 1

    # We remove duplicates
    all_turns_checking_for_duplicates = []
    conversations_no_duplicates = []
    for conv in full_conversations:
        if conv["turns"] not in all_turns_checking_for_duplicates:
            if conv["split"] != "exclude":
                conversations_no_duplicates.append(conv)
                all_turns_checking_for_duplicates.append(conv["turns"])

    print("Total observations:", len(conversations_no_duplicates))

    with open("LUCID_data.json", "w") as json_file:
        json.dump(conversations_no_duplicates, json_file, indent=4)
