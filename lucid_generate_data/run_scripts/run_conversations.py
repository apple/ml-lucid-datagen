#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import logging
import os
from os import listdir
from os.path import isfile, join

from lucid_generate_data.run_scripts.constants import INTENT_PATH
from lucid_generate_data.execute import execute, load_config
from lucid_generate_data.utils.definitions import (
    ProgramTurn,
    UserTurn,
    AutoTransientTurn,
    AutoTurn,
    LucidTurn,
)

CONVS_PER_INTENT = 1
MAX_INTENTS_IN_CONVERSATION = 1
UNHAPPY_PATHS = ["start_multi_slot"]


def save_conversation(filename: str, conv: dict):
    with open("lucid_generate_data/saved_conversations/" + f"{filename}.json", "w") as jsonfile:
        json.dump(conv, jsonfile)


def get_list_of_turns(trace):
    all_turns = []
    for turn in trace["turns_with_hints"]:
        turn_dict = {}
        if isinstance(turn, UserTurn):
            speaker = "User"
        elif isinstance(turn, ProgramTurn):
            speaker = "System"
        elif isinstance(turn, LucidTurn):
            speaker = "Response"
        elif isinstance(turn, AutoTransientTurn):
            speaker = "AutoTransientTurn"
        elif isinstance(turn, AutoTurn):
            speaker = "AutoTurn"
        else:
            raise ValueError("Invalid speaker type")

        turn_dict["author"] = speaker

        all_attributes = [attribute for attribute in dir(turn) if not attribute.startswith("__")]
        for att in all_attributes:
            att_value = getattr(turn, att)
            turn_dict[att] = att_value

        all_turns.append(turn_dict)

    return all_turns


if __name__ == "__main__":
    config_path = "lucid_generate_data/configs/run_with_created_intents.yaml"
    stages = load_config(config_path)

    num_conversations = 0

    all_intents = []

    mypath = INTENT_PATH + "/"

    file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in file_list:
        if file[-5:] == ".json":
            with open(mypath + file, "r") as json_file:
                intent = json.load(json_file)
            all_intents.append(intent)

    for intent in all_intents:
        assert intent["confirmation_required"]

        for _ in range(0, CONVS_PER_INTENT):
            trace = {
                "max_intents": MAX_INTENTS_IN_CONVERSATION,
                "rules_to_be_applied": UNHAPPY_PATHS,
                "primary_intent_json": intent,
            }
            try:
                execute(stages, trace)
                output_dict = {"turns": get_list_of_turns(trace)}
                output_dict["dialogue_id"] = str(num_conversations)
                output_dict["unhappy_path"] = "None"

                save_conversation("conversation_" + str(num_conversations), output_dict)
                print("Saved conversation:", num_conversations)

                num_conversations += 1

            except Exception as e:
                logging.error(e)
