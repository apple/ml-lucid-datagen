#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import Any, Dict, List, Optional

from jinja2 import Environment

from lucid_generate_data.openai_call import make_openai_call
from lucid_generate_data.stage import Stage


class GenerateAppIntentDescription(Stage):
    def __init__(self) -> None:
        self.prompt = """
The task is to produce a single sentence to describe an intent that a virtual AI assistant may execute from a phone.

Examples of descriptions include
1. "Making a phone call"
2. "Sending an email"
3. "Booking a ride with an UBER"

The following intents already exist, so the generated description should be different from these: 
{{existing_intent_descriptions_str}}

The intent must not be a query, or a request for information. E.g. "Sending an email" is OK, but "check my emails" is not.

The app intent should belong to the domain: {{domain}}

Output a string describing a new app intent:

        """.strip()
        self.jinga_template = Environment().from_string(self.prompt)

    def __call__(
        self, domain: str, existing_intent_descriptions: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:  # type: ignore
        # We create a str of existing intent descriptions
        if existing_intent_descriptions:
            existing_intent_descriptions_str = ", ".join(existing_intent_descriptions)
        else:
            existing_intent_descriptions_str = "N/A"

        # We generate another description
        full_prompt = self.jinga_template.render(
            domain=domain, existing_intent_descriptions_str=existing_intent_descriptions_str
        )
        intent_str = make_openai_call("create_intent_descriptions", prompt=full_prompt)

        print("\n\nThe generated description is:\n\n", intent_str)

        return {"intent_description": intent_str}
