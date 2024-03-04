#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import Any, Dict, List, Type, Optional, Tuple, Union

from lucid_generate_data.validate_with_tags import LIST_OF_TAGS_POSSIBLE
from lucid_generate_data.generate_system_turn import generate_system_turn
from lucid_generate_data.generate_user_turn import generate_user_turn
from lucid_generate_data.stage import StageExecutionException, Stage

from lucid_generate_data.output_conversation_rules import output_conversation_rules_current_intent
from lucid_generate_data.utils.definitions import (
    Turn,
    UserTurn,
    ProgramTurn,
    AutoTurn,
    LucidTurn,
    AutoTurn,
    AutoTransientTurn,
)
from lucid_generate_data.executor.executor import ExecutionState, ProgramExecutor
from lucid_generate_data.utils.commands import Command, CommandRegistry, Hint, Perform, Say
from lucid_generate_data.code_gen import (
    AppIntent,
    create_entity_from_intent,
    build_app_context,
    entity_to_query_class,
    entity_to_query_def,
    intent_to_class,
    intent_to_func_def,
)

PRINT_TURNS = False


def build_custom_registry(custom_classes: list[Type[Command]]) -> CommandRegistry:
    builders: list[Type[Command]] = [Say, Hint, Perform] + custom_classes
    builder_dict = {builder.command_name(): builder for builder in builders}
    return CommandRegistry(commands=builder_dict)


class SSAConversation(Stage):
    def __init__(self):
        self.max_trial = 5
        self.max_conversation_len = 50

    def create_executor(
        self, intents: List[Dict[str, Any]], query_info: Dict[str, Any]
    ) -> Tuple[ProgramExecutor, List[str], Dict[str, Any]]:
        """
        We create our SSA executor
        """
        all_intent_definitions = []
        list_of_commands_for_registry = []
        app_context = {}

        # We register each intent in the conversation plan in the executor
        for intent in intents:
            if not intent["query_intent"]:
                intent_definition = intent_to_func_def(intent)
                if intent_definition in all_intent_definitions:
                    continue
                all_intent_definitions.append(intent_definition)
                list_of_commands_for_registry.append(intent_to_class(intent))

            # We check if we need to also register queries included via our unhappy paths
            if intent["command"] in query_info:
                app_entity = create_entity_from_intent(intent)
                all_intent_definitions.append(entity_to_query_def(app_entity))
                list_of_commands_for_registry.append(entity_to_query_class(app_entity))
                app_context.update(
                    build_app_context(app_entity, query_info[intent["command"]]["entity"])
                )

        # Create executor
        registry = build_custom_registry(list_of_commands_for_registry)
        executor = ProgramExecutor(registry=registry, state=ExecutionState(app_context=app_context))

        return executor, all_intent_definitions, app_context

    def extract_tags_from_last_user_turn(
        self, turns: List[Union[ProgramTurn, LucidTurn, UserTurn, AutoTurn, AutoTransientTurn]]
    ) -> Tuple[
        List[str], List[Union[ProgramTurn, LucidTurn, UserTurn, AutoTurn, AutoTransientTurn]]
    ]:
        all_tags = []
        for tag in LIST_OF_TAGS_POSSIBLE:
            if "[" + tag + "]" in turns[-1].query:
                all_tags.append(tag)
                turns[-1].query = turns[-1].query.replace("[" + tag + "]", "").strip()

        return all_tags, turns

    def create_conversation_rules(
        self,
        full_intent_list: List[str],
        intent_number: int,
        intents: List[Dict[str, Any]],
        rules_to_be_applied_by_intent: List[list],
        unhappy_path_args: List[List[List[str]]],
        query_info: Dict[str, Any],
        tags_already_seen_for_intent: List[str],
    ) -> str:
        # The conversation rules need to look ahead one intent
        if len(full_intent_list) > intent_number + 1:
            next_intent = full_intent_list[intent_number + 1]
        else:
            next_intent = None

        if len(full_intent_list) <= intent_number:
            intent = None
            full_intent = None
            unhappy_path = None
            rules = None
        else:
            intent = intents[intent_number]
            full_intent = full_intent_list[intent_number]
            unhappy_path = unhappy_path_args[intent_number]
            rules = rules_to_be_applied_by_intent[intent_number]

        conversation_rules = output_conversation_rules_current_intent(
            full_intent,
            next_intent,
            intent,
            rules,
            unhappy_path,
            query_info,
            tags_already_seen_for_intent,
        )

        return conversation_rules

    def get_last_system_turns(
        self, turns: List[Union[ProgramTurn, LucidTurn, UserTurn, AutoTurn, AutoTransientTurn]]
    ) -> Tuple[ProgramTurn, AutoTurn]:
        recent_system_turns = []
        recent_auto_turns = []

        for turn in reversed(turns):
            if isinstance(turn, ProgramTurn):
                if not turn.expression.startswith("say("):
                    recent_system_turns.append(turn)
            elif isinstance(turn, AutoTurn):
                recent_auto_turns.append(turn)
            elif isinstance(turn, UserTurn):
                break

        return recent_system_turns, recent_auto_turns

    def query_performed_in_system_turns(
        self, turns: List[Union[ProgramTurn, LucidTurn, UserTurn, AutoTurn, AutoTransientTurn]]
    ) -> bool:
        query_performed = False

        for turn in turns:
            if turn.expression.startswith("find_"):
                query_performed = True

        return query_performed

    def special_guidance_from_tags(self, tags: List[str]):
        special_guidance = ""

        if "IRRELEVANT" in tags or "OVERHEARD" in tags:
            special_guidance += "If the user is saying something irrelevant, reply with say()"

        if "START_MULTI" in tags:
            special_guidance += (
                "You should include all of the slot values provided by the user this turn"
            )

        if "CANCEL" in tags or "DELAY_CONFIRMATION" in tags:
            special_guidance += (
                "You should response with say() if the user declines to confirm an intent"
            )

        if "ALL_IN_ONE" in tags:
            special_guidance += (
                "You should include all of the slot values provided by the user this turn"
            )

        if "SARASTIC" in tags:
            special_guidance += "If the user is being sarcastic, and not giving a plausible slot value, reply with say()"

        if "CONFIRM_EARLY" in tags:
            special_guidance += "The user is not allowed to confirm in the same turn as giving new slot values. If this is the case, do not confirm the task for the user this turn"

        if "EARLY_END" in tags:
            special_guidance += "If the user utterance doesn't make sense, and may have been cut off early, reply with say()"

        return special_guidance

    def update_position_in_converesation(
        self,
        turns: List[Union[ProgramTurn, LucidTurn, UserTurn, AutoTurn, AutoTransientTurn]],
        intent_number: int,
        tags_already_seen_for_intent: List[str],
        tags: List[str],
    ):
        # Looking at the system and auto_turns since the last user turn
        # .. helps us understand when to update the conversation rules
        # .. e.g. when we're incrementing the intent number
        recent_system_turns, recent_auto_turns = self.get_last_system_turns(turns)

        # When an intent is performed, we consider the next intent
        if recent_auto_turns:
            if recent_auto_turns[-1].expression.startswith("perform("):
                intent_number += 1
                tags_already_seen_for_intent = []
                tags = []

        # When an intent is cancelled, we also consider the next intent
        elif "CANCEL" in tags:
            intent_number += 1
            tags_already_seen_for_intent = []
            tags = []

        # We check if a query intent was used. We need to know if this
        # .. was an intent in its own right, or was being applied as
        # .. part of an unhappy path
        elif recent_system_turns:
            query_performed = self.query_performed_in_system_turns(recent_system_turns)

            if query_performed:
                rules_for_intent = [x.label for x in rules_to_be_applied_by_intent[intent_number]]

                # Handling the case where the query is an unhappy path
                if "query" in rules_for_intent:
                    tags_already_seen_for_intent.append("query")

                # Handling the case the query is its own intent
                else:
                    intent_number += 1
                    tags_already_seen_for_intent = []
                    tags = []

        return intent_number, tags_already_seen_for_intent, tags

    def __call__(  # type: ignore
        self,
        intents: List[Dict[str, Any]],
        full_intent_list: List[str],
        examples: List[str],
        ssa_examples: List[str],
        rules_to_be_applied_by_intent: List[List[str]],
        unhappy_path_args: List[List[str]],
        query_info: Dict[str, Dict[str, Any]],
        query_entity: Optional[str] = None,
    ) -> List[Turn]:
        intent_number = 0
        tags_already_seen_for_intent = []
        tags = []

        # We create rules for the conversation based on our plan
        conversation_rules = self.create_conversation_rules(
            full_intent_list,
            intent_number,
            intents,
            rules_to_be_applied_by_intent,
            unhappy_path_args,
            query_info,
            tags_already_seen_for_intent,
        )

        executor, all_intent_definitions, app_context = self.create_executor(intents, query_info)
        turns: List[Turn] = [
            UserTurn(query="hey lucid", tags=[[]]),
        ]

        try:
            num_turns = 1
            confirmation_required = intents[0]["confirmation_required"]

            # We generate system and user turns until the user ends the conversation
            # .. or we have reached the maximum number of turns
            while (
                "end conversation" not in turns[-1].query.lower()
                and num_turns <= self.max_conversation_len
            ):
                # Some unhappy paths require extra guidance in the LLM prompt
                special_guidance = self.special_guidance_from_tags(tags)

                turns = generate_system_turn(
                    input_turns=turns,
                    ssa_examples=ssa_examples[intent_number],
                    executor=executor,
                    intent_definitions=all_intent_definitions,
                    confirmation_required=confirmation_required,
                    conversation_rules=conversation_rules,
                    tags_extracted=tags,
                    special_guidance=special_guidance,
                )

                # We see what intents and unhappy paths are already completed
                (
                    intent_number,
                    tags_already_seen_for_intent,
                    tags,
                ) = self.update_position_in_converesation(
                    turns, intent_number, tags_already_seen_for_intent, tags
                )

                # We perform any updates to our conversation rules
                # .. e.g. if we've moved to the next intent, or if we have
                # .. already done an unhappy path
                conversation_rules = self.create_conversation_rules(
                    full_intent_list,
                    intent_number,
                    intents,
                    rules_to_be_applied_by_intent,
                    unhappy_path_args,
                    query_info,
                    tags_already_seen_for_intent,
                )

                # We update the examples shown in Lucid's prompt
                if len(full_intent_list) <= intent_number:
                    shown_examples = ["user: end conversation"]
                else:
                    shown_examples = examples[intent_number]

                # Generate the user turn
                turns = generate_user_turn(
                    input_turns=turns,
                    examples=shown_examples,
                    intent_definitions=all_intent_definitions,
                    conversation_rules=conversation_rules,
                    confirmation_required=confirmation_required,
                    app_context=app_context,
                )

                # We extract the unhappy path tags from the user,
                # .. allowing us to map unhappy paths to individiual turns
                tags, turns = self.extract_tags_from_last_user_turn(turns)
                turns[-1].tags.append(tags)
                tags_already_seen_for_intent += tags

                print(f"user: {turns[-1].query}")
                if PRINT_TURNS:
                    print(turns)

                # We limit how many turns per conversation (stops infinite user/system loops)
                num_turns += 1

        except Exception as e:
            raise StageExecutionException(
                f"Invalid SSA failed execution (generate_conservation.py main call): {e}"
            )

        return {"turns_with_hints": turns}
