#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
from typing import Any, Dict

from jinja2 import Environment

from lucid_generate_data.openai_call import make_openai_call
from lucid_generate_data.stage import Stage

REPEATED_CALLS = 15


class GenerateAppIntentValues(Stage):
    def __init__(self) -> None:
        self.prompt = """
The task is to generate an additional field in mobile intent definitions in JSON format.

The current intent definitions describe commands that can be executed by an AI virtual assistant which can help with a range of different tasks. For each command, the JSON describes the arguments needed to support that command, including the their type, whether they are optional, and if any disambiguation is required.

Your task is to take an existing mobile intent JSON, and generate the same JSON, but including possible values for each argument.

Every argument must have a list of possible values - leaving an empty list is not an option. If there are an unlimited number of possible values, please provide a selection of possible, plausible values that could be use for that argument.

Here is an example of an intent that allows the user to get a ride with Uber
```
{
  "command": "get_ride",
  "args": {
    "destination": {
      "type": "str",
      "optional": false,
      "disambiguation": true
    },
    "number_of_passengers": {
      "type": "int",
      "optional": false
    },
    "service_type": {
      "type": "str",
      "optional": false
    },
    "origin": {
      "type": "str",
      "optional": true,
      "disambiguation": true
    },
    "luggage": {
      "type": "bool",
      "optional": true
    },
    "booking_time": {
      "type": "str",
      "optional": true
    }
  },
  "confirmation_required": true,
  "description": "get a ride with Uber",
  "entity_name": "rides", 
  "domain": "transportation"
}

We now provide the same intent, but with possible values for each argument:

{
  "command": "get_ride",
  "args": {
    "destination": {
      "type": "str",
      "values": [
        "the pool", "school", "the gym", "the office", "back home", "the airport", "Dan's house"
        ],
      "optional": false,
      "disambiguation": true
    },
    "number_of_passengers": {
      "type": "int",
      "values": [
        0, 1, 2, 3, 4, 5, 6, 7, 8
        ],
      "optional": false
    },
    "service_type": {
      "type": "str",
      "values": [
        "regular", "UberX", "UberXL", "UberSUV", "UberBLACK"
        ],
      "optional": false
    },
    "origin": {
      "type": "str",
      "values": [
        "home", "the office", "the park", "the office", "the airport"
        ],
      "optional": true,
      "disambiguation": true
    },
    "luggage": {
      "type": "bool",
      "values": [
        false, true
        ],
      "optional": true
    },
    "booking_time": {
      "type": "str",
      "values": [
        "5:30pm", "2pm", "7pm this evening", "4pm", "6:30am", "8am", "midnight"
        ],
      "optional": true
    }
  },
  "confirmation_required": true,
  "description": "get a ride with Uber",
  "entity_name": "rides",
  "domain": "transportation"
}


Now generate values for the following JSON intent description:
{{intent_no_values}}

Generate an intent JSON for the intent:

        """.strip()
        self.jinga_template = Environment().from_string(self.prompt)

    def __call__(self, intent_no_values: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:  # type: ignore
        full_prompt = self.jinga_template.render(intent_no_values=intent_no_values)

        intent_with_values = None

        for i in range(REPEATED_CALLS):
            intent_str = make_openai_call("intent_value_creation", prompt=full_prompt)
            try:
                intent = json.loads(intent_str)
                valid_values, reason = self.validate(intent, intent_no_values)
                if reason:
                    print("Generation failed due to:", reason)
            except Exception as e:
                error = e
            else:
                if valid_values:
                    if intent_with_values:
                        intent_with_values = self.combine_jsons(intent_with_values, intent)
                    else:
                        intent_with_values = intent
            print("Cycle:", i, "complete")

        print("Intent with values:", intent_with_values)

        return {"intent": intent_with_values}

    def combine_jsons(self, json1, json2):
        for slot_name, slot_dict in json1["args"].items():
            json1["args"][slot_name]["values"] += json2["args"][slot_name]["values"]
            json1["args"][slot_name]["values"] = list(set(json1["args"][slot_name]["values"]))

        return json1

    def validate(self, intent: Dict[str, Any], intent_no_values: Dict[str, Any]) -> None:
        for arg in intent["args"]:
            # We want to compare before / after adding the values
            if arg not in intent_no_values["args"]:
                return False, "Generating values has changed JSON intent definitions"

            if "values" not in intent["args"][arg]:
                return False, "No values were generated for intents"

            intent_no_values["args"][arg]["values"] = intent["args"][arg]["values"]

            # Checking the type of the values
            if not isinstance(intent["args"][arg]["values"], list):
                return False, "Values should be a list"

            # Check there are values for each arg
            if intent["args"][arg]["values"] == []:
                return False, "Values have not been provided for each argument"

        # Check no changes to the rest of the JSON
        if intent_no_values != intent:
            return False, "Generating values has changed JSON intent definitions"

        return True, None
