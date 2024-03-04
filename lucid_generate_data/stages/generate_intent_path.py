#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import random
from typing import Any, Dict, List, Union
from os import listdir
from os.path import isfile, join
import copy

from jinja2 import Environment

from lucid_generate_data.run_scripts.constants import INTENT_PATH
from lucid_generate_data.openai_call import make_openai_call
from lucid_generate_data.stage import Stage, StageExecutionException

INCLUDE_ALL_INTENTS_IN_SAME_DOMAIN = True
PROB_FIRST_INTENT_QUERY = 0.25
INTENTS_TO_INCLUDE = 10
QUERY_DATE_OF_EXP = [
    "yesterday",
    "most recent",
    "last time",
    "last week",
    "last one",
    "the one before last",
    "last one",
    "last month",
    "Monday",
    "Tuesday",
    "the weekend",
    "last weekend",
]


class GenerateRequestPath(Stage):
    def __init__(self) -> None:
        self.prompt = """

Your task is to generate an abstract representation of a coherent conversation between a human and a virtual assistant, in the form of a series of function calls representing different intents.  You will be shown the allowed function calls in advance and you must decide which to include and how to order them. After generating the sequence of function calls, you must justify why this is a sensible sequence of intents.

The intents must be related to make a coherent story, with the slot values in the intents relating to each other. The slots in one intent should make a reference to the intents that have already happened.

There are two outputs. Output 1), which lists the intents in order, and Output 2), which gives a justification.

Example 1:

The conversation must use the intents ['send_message', 'book_flight', 'book_nail_appointment', 'book_ride', 'order_flowers', 'buy_shopping']. The conversation must first start with 'book_restaurant'. 

Output 1) book_restaurant, book_ride, send_message

Output 2) The user books a restaurant, before ordering an uber for the restaurant for later. Finally, the user sends a text to their friend to let them know the table is booked.

Example 2:

Output 1) send_message, book_ride, set_reminder, 

Output 2) The user sends a message to tell their sister their present is in the post. They then book an uber to the shops to buy the present. They set a reminder for 10 minutes to wait for the uber to arrive.

Example 3:

Output 1) find_song, play_song

Output 2) The user can't remember who Paranoid Android is by, so they ask Lucid first before they play the song

Now it's your turn. Include both Output 1) and Output 2)

The conversation must use the intents {{sample_of_intents}}. The conversation must first start with {{primary_intent}}. Use at most {{max_intents}} intents.

Output 1)
        """.strip()
        self.jinga_template = Environment().from_string(self.prompt)

    def remove_duplicated_commands(
        self, command_list: List[str], confirmation_required: List[bool]
    ) -> (List[str], List[bool]):
        """
        We remove commands that have been repeated in the conversation
        """
        # We remove duplicates, and require confirmation unless the task was abandoned
        prev_command = "N/A"
        unique_commands = []
        unique_confirmation_required = []

        for i, command in enumerate(command_list):
            if command != prev_command:
                unique_commands.append(command)

                # Did the last command require confirmation?
                if i > 0:
                    unique_confirmation_required.append(confirmation)

                conf_required = confirmation_required[i]

            confirmation = confirmation_required[i]
            prev_command = command

        # We always confirm the last command
        unique_confirmation_required.append(True)

        return unique_commands, unique_confirmation_required

    def get_conversation_substring(self, prompt_str: str) -> str:
        # We only want to look at the conversation transcript
        str_for_next_section = "\n"
        str_for_end_section = "Output 2)"

        if str_for_next_section not in prompt_str:
            raise StageExecutionException(
                "There is no new line to separate intent path from reasons"
            )
        if str_for_end_section not in prompt_str:
            raise StageExecutionException(
                "There is no reason provided with the intent path 'Output 2)' not present"
            )

        prompt_str_start = prompt_str[: prompt_str.find(str_for_next_section)].strip()
        prompt_str_end = prompt_str[
            prompt_str.find(str_for_end_section) + len(str_for_end_section) :
        ].strip()

        return prompt_str_start, prompt_str_end

    def get_command_list(self, prompt_str: str) -> List[str]:
        command_list = prompt_str.strip().replace(",", "").split(" ")
        command_list = [command for command in command_list if command != ""]

        return command_list

    def extract_command_list(
        self, prompt_str: str, sample_of_intents: List[str], intent_query_lookup
    ) -> (List[str], List[bool]):
        """
        We extract a list of commands from the conversation str
        """
        # We extract our command list
        prompt_str, reasoning = self.get_conversation_substring(prompt_str)
        command_list = self.get_command_list(prompt_str)

        is_query_list = []
        corresponding_intent = []
        updated_command_list = []

        # We create a list of intents for the conversation, extracted from the LLM response
        for i in range(len(command_list)):
            command = command_list[i]

            # No commands should be hallucinated
            if command not in sample_of_intents:
                raise StageExecutionException(
                    "Command: " + command + " generated that is not in sample provided"
                )
            # Some of the intents chosen may be query intents
            query_intent = command.startswith("find_")

            # We do not allow queries when we have already performed the corresponding transactional intent
            if query_intent:
                if intent_query_lookup[command] in updated_command_list:
                    continue

            # We add the intent to our intent path
            is_query_list.append(query_intent)
            updated_command_list.append(command)

            # We identify the corresponding transactional intents for each query intent
            if query_intent:
                corresponding_intent.append(intent_query_lookup[command])
            else:
                corresponding_intent.append(None)

        return updated_command_list, is_query_list, corresponding_intent, reasoning

    def get_all_intents(self) -> List[Dict[str, Any]]:
        """
        Return a list of all intent JSONs
        """
        all_intents = []
        mypath = INTENT_PATH + "/"

        file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for file in file_list:
            if file[-5:] == ".json":
                with open(mypath + file, "r") as json_file:
                    intent = json.load(json_file)
                all_intents.append(intent)

        return all_intents

    def get_sample_of_intents(
        self, all_intents: List[Dict[str, Any]], primary_intent: str, primary_domain: str
    ) -> (List[str], List[str]):
        # Sample a subset of these intents
        no_intents = min(INTENTS_TO_INCLUDE, len(list(all_intents)))
        sample_of_intents = random.sample(all_intents, no_intents)

        # We can also include all intents from the primary domain
        if INCLUDE_ALL_INTENTS_IN_SAME_DOMAIN:
            domain_intents = [x for x in all_intents if (x["domain"] == primary_domain)]
            sample_of_intents += domain_intents

        # For each intent, we also also include its corresponding query intent
        intent_to_query_lookup = {}

        sample_intent_names = []
        for intent_json in sample_of_intents:
            entity_name = intent_json["entity_name"]
            sample_intent_names.append(f"find_{entity_name}")
            sample_intent_names.append(intent_json["command"])
            intent_to_query_lookup[f"find_{entity_name}"] = intent_json["command"]

        # We make sure the primary intent is included
        sample_intent_names = [primary_intent] + sample_intent_names

        # Unique list of intents including primary intent
        sample_intent_names = list(set(sample_intent_names))

        return sample_intent_names, intent_to_query_lookup

    def get_query_version(self, app_intent):
        query_intent = copy.deepcopy(app_intent)
        query_intent["query_intent"] = True
        query_name = "find_" + query_intent["entity_name"]
        query_intent["command"] = query_name

        # There are some differences between the query intent and the corresponding transactional intent
        for slot_name, slot_dict in app_intent["args"].items():
            # We avoid slots that will clash with date_of_entity
            if "date" in slot_name or "time" in slot_name or "day" in slot_name:
                del query_intent["args"][slot_name]
            else:
                query_intent["args"][slot_name]["optional"] = True

        # We introduce a new slot called date_of_[entity_name], that has a standard set of initial values
        entity_name = query_intent["entity_name"]
        query_intent["args"].update(
            {
                f"date_of_{entity_name}": {
                    "type": "str",
                    "values": QUERY_DATE_OF_EXP,
                    "optional": False,
                }
            }
        )

        # Query intents do not require confirmation
        query_intent["confirmation_required"] = False
        query_intent["description"] = f"Query intent for finding existing {entity_name}"

        return query_intent

    def update_query_intents(self, command_list, corresponding_intent, is_query_list):
        intents_json = []

        # Intent JSONs for query intents are derived from their corresponding transactional intent
        for i, intent in enumerate(command_list):
            # We find the file path of the intent JSON
            intent_path = INTENT_PATH

            if is_query_list[i]:
                path = intent_path + "/" + corresponding_intent[i] + ".json"
            else:
                path = intent_path + "/" + intent + ".json"

            with open(path, "r") as json_file:
                intent_json = json.load(json_file)

            # We update the query intent
            if is_query_list[i]:
                query_version = self.get_query_version(intent_json)
                intents_json.append(query_version)

            # For other intents, we use the JSON intent
            else:
                intent_json["query_intent"] = False
                intents_json.append(intent_json)

        return intents_json

    def create_intent_path(self, primary_intent, primary_domain, max_intents):
        # We get all possible intents that could be included
        all_intents = self.get_all_intents()

        # Unique list of intents including primary intent
        sample_of_intents, intent_query_lookup = self.get_sample_of_intents(
            all_intents, primary_intent, primary_domain
        )

        # We generate an example conversation containing our intent path
        full_prompt = self.jinga_template.render(
            max_intents=max_intents,
            primary_intent=primary_intent,
            sample_of_intents=sample_of_intents,
        )

        path_in_str = ""
        while "Output 2)" not in path_in_str:
            path_in_str = make_openai_call("find_intent_path", prompt=full_prompt)

            # Extract our intent path from this conversation
        (command_list, is_query_list, corresponding_intent, reasoning) = self.extract_command_list(
            path_in_str, sample_of_intents, intent_query_lookup
        )

        return command_list, is_query_list, corresponding_intent, reasoning

    def __call__(
        self, max_intents: int, primary_intent_json: Dict[str, Any]
    ) -> Dict[str, List[Union[str, bool]]]:
        # We decide if the first intent is a query or not
        if random.random() < PROB_FIRST_INTENT_QUERY:
            primary_intent_json = self.get_query_version(primary_intent_json)

        primary_intent = primary_intent_json["command"]
        primary_domain = primary_intent_json["domain"]

        # We create a prompt for an LLM to choose the intent path,
        # .. providing possible choices of intents
        (command_list, is_query_list, corresponding_intent, reasoning) = self.create_intent_path(
            primary_intent, primary_domain, max_intents
        )

        # We make sure the first intent in the intent path is the primary intent specified
        if len(command_list) == 0:
            raise StageExecutionException("No commands provided")

        if command_list[0] != primary_intent:
            raise StageExecutionException("Conversation should start with primary intent")

        # We update the intent JSONs where these are queries
        intents_json = self.update_query_intents(command_list, corresponding_intent, is_query_list)

        # We truncate the intent path is necessary
        if max_intents < len(command_list):
            intents_json = intents_json[:max_intents]

        return {"intents": intents_json, "intent_path_reasoning": reasoning}
