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


INTENTS_IN_DOMAIN = 5

INTENTS = {
    "messages_and_emails": [
        "Delete an email",
        "Move a message to the junk folder",
    ]
}


def save_intent(filename: str, conv: dict):
    with open(INTENT_PATH + "/" + f"{filename}.json", "w") as jsonfile:
        json.dump(conv, jsonfile)


if __name__ == "__main__":
    config_path = "lucid_generate_data/configs/create_intents_from_description.yaml"
    stages = load_config(config_path)

    existing_intent_descriptions = []

    mypath = INTENT_PATH
    file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in file_list:
        if file[-5:] == ".json":
            existing_intent_descriptions.append(file[:-5])

    for domain in INTENTS:
        for desc in INTENTS[domain]:
            trace = {
                "existing_intent_descriptions": existing_intent_descriptions,
                "domain": domain,
                "intent_description": desc,
            }
            try:
                execute(stages, trace)

                save_intent(trace["intent_name"], trace["intent"])
                print("Saved intent:", trace["intent_name"])
                existing_intent_descriptions.append(trace["intent_name"])

            except Exception as e:
                logging.error(e)
