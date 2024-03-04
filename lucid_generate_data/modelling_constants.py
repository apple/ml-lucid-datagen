#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

STAGE_MODEL_LOOKUP = {
    "create_intent_descriptions": {
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 600,
        "temperature": 0.7,
    },  # Stage 0 - unused
    "create_intent_json": {
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 600,
        "temperature": 0.7,
    },  # Stage 1 (creating intents)
    "intent_value_creation": {
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 800,
        "temperature": 0.7,
    },  # Stage 2 (creating intents)
    "find_intent_path": {
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 600,
        "temperature": 0.7,
    },  # Stage 3 (conversation planner)
    "make_slot_values_more_realistic": {
        "model_name": "gpt-4",
        "max_tokens": 300,
        "temperature": 0.7,
    },  # Stage 4 (conversation planner)
    "get_context_for_future_command": {
        "model_name": "gpt-4",
        "max_tokens": 100,
        "temperature": 0.7,
    },  # Stage 5 (conversation planner)
    "get_likely_slot_value_given_context": {
        "model_name": "gpt-4",
        "max_tokens": 100,
        "temperature": 0.7,
    },  # Stage 6 (conversation planner)
    "replanning_for_coherence": {
        "model_name": "gpt-4",
        "max_tokens": 300,
        "temperature": 0.7,
    },  # Stage 7 (conversation planner)
    "get_query_values": {
        "model_name": "gpt-4",
        "max_tokens": 300,
        "temperature": 0.7,
    },  # Stage 8 (conversation planner)
    "user_agent": {
        "model_name": "gpt-4",
        "max_tokens": 200,
        "temperature": 0.7,
    },  # Stage 9 (generating conversations)
    "lucid_agent": {
        "model_name": "gpt-4",
        "max_tokens": 200,
        "temperature": 0.7,
    },  # Stage 10 (generating conversations), and Stages 12 and 13 (conversation validation)
    "get_slot_values": {
        "model_name": "gpt-4",
        "max_tokens": 200,
        "temperature": 0.7,
    },  # Stage 11 (generating conversations)
}
