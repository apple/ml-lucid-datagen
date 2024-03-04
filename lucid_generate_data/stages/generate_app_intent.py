#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
from typing import Any, Dict, Optional, List

from jinja2 import Environment

from lucid_generate_data.openai_call import make_openai_call
from lucid_generate_data.stage import Stage, StageExecutionException


class GenerateAppIntent(Stage):
    def __init__(self) -> None:
        self.prompt = """
The task is to generate mobile app intent definitions in JSON format. The following rules must be followed strictly:

1. The JSON file should include both a command as well as arguments of the command.
2. More important arguments must be generated first.
3. Mandatory arguments must have field "optional" set to false. These arguments must be provided by the user before the command can be executed.
4. Arguments that support value disambiguation must have field "disambiguation" set to true. These arguments usually have an open set of string values, which the user often provides a partial match.
5. Any items contained within "args" must include the fields "type" and "optional".
6. The entity_name should describe the entities created by the intent (e.g. get_ride creates rides, or set_alarm creates alarms)

Here is an example of an app intent that allows the user to get a ride with Uber
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
  "entity_name": "rides"
}

The intent must not have any of the following names (as existing intents use these names):
{{existing_intent_descriptions_str}}

Now given a new app with the following description:
{{intent_description}}

This new app is part of the domain: {{domain}}

This app should require confirmation, so confirmation_required should be set to true.

Generate an app intent JSON for the app:

        """.strip()
        self.jinga_template = Environment().from_string(self.prompt)

    def __call__(
        self,
        domain: str,
        intent_description: str,
        existing_intent_descriptions: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        We generate a JSON for our intent based on the description
        """

        # We want to avoid using the same intent name for differen intents
        if existing_intent_descriptions:
            existing_intent_descriptions_str = ", ".join(existing_intent_descriptions)
        else:
            existing_intent_descriptions_str = "N/A"

        # We create the intent JSONS using an LLM
        full_prompt = self.jinga_template.render(
            intent_description=intent_description,
            existing_intent_descriptions_str=existing_intent_descriptions_str,
            domain=domain,
        )

        intent_str = make_openai_call("create_intent_json", prompt=full_prompt)

        # We perform validation checks on the intent JSON generated
        try:
            # Check the JSON can be loaded
            intent = json.loads(intent_str)
        except:
            raise StageExecutionException("Invalid App Intent Json.")
        else:
            # Additional validation
            self.validate(intent)

            if intent["command"] in existing_intent_descriptions:
                raise StageExecutionException(
                    intent["command"]
                    + " is an existing intent, that can be found in the current intent list: "
                    + str(existing_intent_descriptions)
                )

            intent["domain"] = domain

            print("\n\nThe following intent has been generated:\n\n", intent)
            return {"intent_no_values": intent, "intent_name": intent["command"]}

    def validate(self, intent: Dict[str, Dict[str, Any]]) -> None:
        # Check at least one slot has been generated
        if len(intent["args"]) == 0:
            raise StageExecutionException("Invalid App Intent with no slots.")

        # Check there is at least one required slot
        all_slots_optional = True
        for arg, arg_dict in intent["args"].items():
            if not arg_dict["optional"]:
                all_slots_optional = False

        if all_slots_optional:
            raise StageExecutionException("Each intent requires at least one required slot")
