#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

LIST_OF_TAGS_POSSIBLE = [
    "IRRELEVANT",
    "START_MULTI",
    "CANCEL",
    "CONFIRM",
    "QUERY",
    "CORRECTION",
    "WRONG_SLOT",
    "ALL_IN_ONE",
    "OVERHEARD",
    "SARCASTIC",
    "CONFIRM_EARLY",
    "TURN_CORRECTION",
    "EARLY_END",
    "DELAY_CONFIRMATION",
]


def validation_from_tags(first_system_command: bool, predicted_response, tags_extracted):
    if "[EARLY_END]" in tags_extracted:
        if predicted_response == "say()":
            return True, "Response cut off early is identified as irrelevant as expected"
        else:
            return (
                False,
                "Expected [EARLY_END] response to get say() prediction, instead got: "
                + str(predicted_response),
            )

    if "[IRRELEVANT]" in tags_extracted:
        if predicted_response == "say()":
            return True, "Response is irrelevant as expected"
        else:
            return (
                False,
                "Expected [IRRELEVANT] response to get say() prediction, instead got: "
                + str(predicted_response),
            )

    if "[ALL_IN_ONE]" in tags_extracted and first_system_command:
        if "(" in predicted_response in predicted_response:
            return True, "New intent given"
        else:
            return False, "Expected [ALL_IN_ONE] to start a new intent, instead got: " + str(
                predicted_response
            )

    if "[START_MULTI]" in tags_extracted and first_system_command:
        if "(" in predicted_response and "," in predicted_response:
            return True, "Intent given with more than one slot"
        else:
            return (
                False,
                "Expected [START_MULTI] to set values for >1 slot for new intent, instead got: "
                + str(predicted_response),
            )

    if "[CANCEL]" in tags_extracted:
        if predicted_response == "say()":
            return True, "Cancellation leads to say() prediction"
        else:
            return False, "Expected [CANCEL] to get say() prediction, instead got: " + str(
                predicted_response
            )

    if "[CONFIRM]" in tags_extracted and first_system_command:
        if predicted_response.startswith("confirm("):
            return True, "User confirmation being given with Confirm"
        else:
            return False, "Expected [CONFIRM] to get confirm() prediction, instead got: " + str(
                predicted_response
            )

    if "[SARCASTIC]" in tags_extracted:
        if predicted_response == "say()":
            return True, "Response is sarcastic, and identified as irrelevant as expected"
        else:
            return (
                False,
                "Expected [SARCASTIC] response to get say() prediction, instead got: "
                + str(predicted_response),
            )

    if "[CANCEL]" in tags_extracted:
        if predicted_response == "say()":
            return True, "Response for cancelling a task is correctly returning say()"
        else:
            return False, "Expected [CANCEL] response to get say() prediction, instead got: " + str(
                predicted_response
            )

    if "[DELAY_CONFIRMATION]" in tags_extracted:
        if predicted_response == "say()":
            return True, "Response for delaying confirmation of a task is correctly returning say()"
        else:
            return (
                False,
                "Expected [DELAY_CONFIRMATION] response to get say() prediction, instead got: "
                + str(predicted_response),
            )

    if "[OVERHEARD]" in tags_extracted:
        if predicted_response == "say()":
            return (
                True,
                "Response is over-heard from an irrelevant conversation, and has been identified as irrelevant as expected",
            )
        else:
            return (
                False,
                "Expected [OVERHEARD] response to get say() prediction, instead got: "
                + str(predicted_response),
            )

    return True, None
