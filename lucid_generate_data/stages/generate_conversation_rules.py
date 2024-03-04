#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import random
from random import Random
from typing import Any, Dict, List, Tuple
import copy

from jinja2 import Environment

from lucid_generate_data.conversation_rule import (
    ConversationRule,
    conversation_rules,
    default_example,
    default_ssa_examples,
)
from lucid_generate_data.openai_call import make_openai_call
from lucid_generate_data.stage import Stage, StageExecutionException
from lucid_generate_data.code_gen import intent_to_func_def
from lucid_generate_data.utils.commands import (
    parse_system_function_call,
    check_parsed_slots_vs_intent_json,
)

NO_RETRIES = 5


class GenerateConversationRules(Stage):
    def __init__(self) -> None:
        self.rules: List[ConversationRule] = conversation_rules
        self.rng = Random()

    def check_function_valid_types(self, command: str, intent: Dict[str, Any]) -> bool:
        is_valid_json, parsed_slot_data_types = parse_system_function_call(command)

        is_correct_slot_data_types = check_parsed_slots_vs_intent_json(
            parsed_slot_data_types, intent
        )

        if is_valid_json and is_correct_slot_data_types:
            return True

        return False

    def check_all_rules_valid(self, args: List[str], rules_to_be_applied: List[Any]) -> None:
        """
        Which rules have enough (valid) arguments to support them
        """
        valid_rules = []

        if rules_to_be_applied:
            for rule in self.rules:
                if rule.label in rules_to_be_applied:
                    if rule.args_required > len(args):
                        raise StageExecutionException("Not enough args for suggested rule")

    def create_single_unhappy_path(
        self,
        remaining_args: List[str],
        selected_examples: List[str],
        all_ssa_examples: List[str],
        selected_rule: ConversationRule,
    ) -> (list, list, bool, list, str):
        """
        Adding an unhappy path to our conversation rules
        """
        selected_args_for_rule = []

        if selected_rule.args_required > len(remaining_args):
            raise StageExecutionException(
                "Not enough args for suggested rule after other rules applied"
            )

        for arg_id in range(1, selected_rule.args_required + 1):
            arg_chosen = remaining_args.pop(self.rng.choice(range(0, len(remaining_args))))
            selected_args_for_rule.append(arg_chosen)

        # The selected example is provided for the final rule selected
        selected_examples.append(selected_rule.example)
        all_ssa_examples.append(selected_rule.ssa_example)

        return (remaining_args, selected_examples, all_ssa_examples, selected_args_for_rule)

    def create_unhappy_paths(
        self, args: List[str], rules_to_be_applied: List[str]
    ) -> Tuple[List[str], str, bool]:
        """
        We create our unhappy path rules for the conversation
        """
        # We populate the templates with arguments that are contained in our desired final command
        selected_examples = copy.deepcopy(default_example)
        all_ssa_examples = copy.deepcopy(default_ssa_examples)
        unhappy_path_labels = []
        selected_args = []

        remaining_args = args
        # We create all the rules for our unhappy path, adding these to our prompt
        if rules_to_be_applied:
            for rule_text in [x.label for x in rules_to_be_applied]:
                for rule in self.rules:
                    if rule_text == rule.label:
                        (
                            remaining_args,
                            selected_examples,
                            all_ssa_examples,
                            selected_args_for_rule,
                        ) = self.create_single_unhappy_path(
                            remaining_args, selected_examples, all_ssa_examples, rule
                        )
                        selected_args.append(selected_args_for_rule)

        return selected_args, selected_examples, all_ssa_examples

    def get_query_entity_and_rationale(
        self,
        original_intent: Dict[str, Any],
        syntax: str,
        command: str,
        entity_name: str,
        all_args: List[str],
        conversation_full_list: List[str],
    ) -> (str, str):
        all_args_str = ", ".join(all_args)
        syntax_all_str = syntax

        prompt = """
        A user is talking to a virtual assistant called Lucid. The user wants Lucid to do the following query:
        {{command}}
        
        You need to think of a suitable {{entity_name}} entity that should be returned by the query {{command}} when it happens in the conversation. You also need to guess a rationale for the query.

        Example 1:
        query command: find_email(to="danny")
        conversation: find_email(to="danny"), send_email(to="danny", cc="danny@work_email.com")
        rationale for query: The user wants to find out what Danny's work email is, so they check the CC field in the last email
        returned entity: email(to="danny", cc="danny@work_email.com", subject="countdown")

        Example 2:
        query command: find_sent_flowers(to="mum")
        conversation: find_sent_flowers(to="mum"), send_message(to="Josh", content="We should send mum some more flowers, just like the roses we last got her")
        rationale for query: The user wants to find out what flowers they last sent to their mum
        returned entity: sent_flowers(to="mum", type="roses", message="happy mothers day")

        Now it's your turn.
        Your answer must contain 1) the rationale for query, 2) the returned entity - in the form {{entity_name}}(...). The returned entity can have args: {{all_args_str}}

        You must use the following data types:
        {{syntax_all_str}}

        query command: {{command}}
        conversation: {{conversation_full}}
        rationale for query: 
        """
        jinga_template = Environment().from_string(prompt)

        conversation_full = ", ".join(conversation_full_list)
        full_prompt = jinga_template.render(
            command=command,
            entity_name=entity_name,
            conversation_full=conversation_full,
            all_args_str=all_args_str,
            syntax_all_str=syntax_all_str,
        )

        no_retries = 0

        # We use an LLM to give us the query rationale and entity
        while no_retries <= NO_RETRIES:
            query_information = make_openai_call("get_query_values", prompt=full_prompt)
            query_info = query_information.split("\n")[:3]
            rationale = query_info[0].replace("rationale for query: ", "").strip()

            if "returned entity: " not in query_info[1]:
                no_retries += 1
                print("Retry (query generation pt2):", no_retries)
            else:
                entity = query_info[1].replace("returned entity: ", "").strip()
                query_intent = copy.deepcopy(original_intent)
                query_intent["args"]["date_of_" + entity_name] = {"type": "str"}

                # We check the data types of each generated slot value
                if self.check_function_valid_types(entity, query_intent):
                    break
                else:
                    no_retries += 1
                    print("Retry (query generation pt2 - type checking):", no_retries)

        assert no_retries <= NO_RETRIES

        return rationale.strip(), entity.strip()

    def get_query_intent_entity_and_rationale(
        self,
        original_intent: Dict[str, Any],
        syntax: str,
        command: str,
        entity_name: str,
        all_args: List[str],
    ) -> (str, str):
        query_command_name = "find_" + entity_name

        all_args_str = ", ".join(all_args)
        syntax_all_str = syntax

        prompt = """
        A user is talking to a virtual assistant called Lucid. The goal of the user is for Lucid to perform the following command:
        {{command}}

        At some point in the conversation, the user will raise a query. Your task is to suggest a plausible query that they might raise.

        When finding {{entity_name}} entities with {{query_command_name}}, you can refer to any of the slots in the schema below, in addition to the 'date_of_{{entity_name}}' slot (str), which says how many days ago the entity was created. 

        Slots:
        {{syntax_all_str}} 

        command at end is the command the user ends up with after performing the query, so the query should help the user reach this command.

        Examples:
        command at end: send_email(to="danny", subject="next Monday", cc="margaret")
        rationale for query: Find out who I CC'd in my email to danny last week
        query: find_email(to="danny", date_of_email="last week")
        returned entity: email(to="danny", date_of_email="last week", cc="margaret")
        
        command at end: send_flowers(to="mum", subject="happy mothers day", size="large")
        rationale for query: Check how big the most recent flowers were that I sent to my mum
        query: find_sent_flowers(to="mum", date_of_sent_flowers="most recent")
        returned entity: sent_flowers(to="mum", date_of_sent_flowers="most recent", size="large")

        Now it's your turn. Create an query using {{query_command_name}} to find {{entity_name}} entities.

        Your answer must contain 1) the rationale for query, 2) query and 3) returned entity:

        command: {{command}}
        rationale for query: 
        """
        jinga_template = Environment().from_string(prompt)
        full_prompt = jinga_template.render(
            command=command,
            entity_name=entity_name,
            all_args_str=all_args_str,
            query_command_name=query_command_name,
            syntax_all_str=syntax_all_str,
        )

        no_retries = 0

        # We use an LLM to give us the query intent, rationale and entity
        while no_retries <= NO_RETRIES:
            query_information = make_openai_call("get_query_values", prompt=full_prompt)
            query_info = query_information.split("\n")[:3]
            rationale = query_info[0].replace("rationale for query: ", "").strip()

            if "query: " not in query_info[1] or "returned entity: " not in query_info[2]:
                no_retries += 1
                print("Retry (query generation):", no_retries)
            else:
                # We check that the slot data types are correct
                query = query_info[1].replace("query: ", "").strip()
                entity = query_info[2].replace("returned entity: ", "").strip()

                query_intent = copy.deepcopy(original_intent)
                query_intent["args"]["date_of_" + entity_name] = {"type": "str"}

                if self.check_function_valid_types(
                    query, query_intent
                ) and self.check_function_valid_types(entity, query_intent):
                    break
                else:
                    no_retries += 1
                    print("Retry (query generation type checking):", no_retries)

        assert no_retries <= NO_RETRIES

        return rationale.strip(), query.strip(), entity.strip()

    def allocate_number_rules_to_intents(
        self, total_no_rules: int, total_no_intents: int
    ) -> List[int]:
        unhappy_paths_per_intent = []
        rules_remaining = total_no_rules

        # Find out how
        for i in range(total_no_intents):
            intents_remaining = total_no_intents - i
            rules_for_intent = self.get_number_rules_per_intent(intents_remaining, rules_remaining)
            unhappy_paths_per_intent.append(rules_for_intent)
            rules_remaining -= rules_for_intent

        if rules_remaining != 0:
            raise StageExecutionException("Rules not all allocated to intents")

        return unhappy_paths_per_intent

    def get_number_rules_per_intent(self, intents_remaining: int, rules_remaining: int) -> int:
        remaining_rules_to_be_applied = []

        # We decide how many unhappy paths should be included for the intent
        unhappy_rules_per_intent = rules_remaining / intents_remaining

        rules = int(unhappy_rules_per_intent)

        prob_another = unhappy_rules_per_intent - rules
        if prob_another > 0:
            if random.random() < prob_another:
                rules += 1

        return rules

    def apportion_rules_to_intents(
        self, unhappy_paths_per_intent: List[int], rules_to_be_applied: List[Any]
    ) -> List[List[Any]]:
        rules_to_be_applied_by_intent = []

        for intent_unhappy_paths in unhappy_paths_per_intent:
            rules_for_intent = []

            for new_rule in range(intent_unhappy_paths):
                rules_for_intent.append(rules_to_be_applied.pop(0))
            rules_to_be_applied_by_intent.append(rules_for_intent)

        return rules_to_be_applied_by_intent

    def get_unhappy_path_args_and_examples_by_intent(
        self,
        intents: List[Dict[str, Any]],
        used_intent_args: List[List[str]],
        rules_to_be_applied_by_intent: List[List[Any]],
    ) -> (List[List[str]], List[List[str]], List[List[str]]):
        all_unhappy_path_args = []
        all_selected_examples = []
        all_ssa_examples = []

        for intent_number in range(len(intents)):
            # We check our unhappy paths have enough args for the intent
            self.check_all_rules_valid(
                used_intent_args[intent_number], rules_to_be_applied_by_intent[intent_number]
            )

            # The unhappy paths inform our choice of prompt examples for each intent
            # ... we also find the slots corresponding to each unhappy path
            unhappy_path_args_, selected_examples_, ssa_examples_ = self.create_unhappy_paths(
                used_intent_args[intent_number], rules_to_be_applied_by_intent[intent_number]
            )

            all_unhappy_path_args.append(unhappy_path_args_)
            all_selected_examples.append(selected_examples_)
            all_ssa_examples.append(ssa_examples_)

        return all_unhappy_path_args, all_selected_examples, all_ssa_examples

    def get_all_query_info(self, intents, rules_by_intent, intent_list, used_intent_args):
        all_query_info = {}

        # For each intent we find any queries, their corresponding entities
        # .. and a rationale for each query
        for intent_number in range(len(intent_list)):
            intent_args = list(intents[intent_number]["args"])

            intent_name = intents[intent_number]["command"]

            # We find the slot names and their data types for the intent
            # ... queries and query entities should conform to these
            syntax = intent_to_func_def(intents[intent_number])
            syntax = syntax[syntax.find("(") + 1 : -1]

            # We first consider queries are unhappy paths
            # .. that is, they are used to help a user perform another intent
            if "query" in [
                x.label for x in rules_by_intent[intent_number]
            ] and not intent_name.startswith("find_"):
                rationale, query, entity = self.get_query_intent_entity_and_rationale(
                    intents[intent_number],
                    syntax,
                    intent_list[intent_number],
                    intents[intent_number]["entity_name"],
                    intent_args,
                )

                query_info = {
                    intent_name: {"query": query, "entity": entity, "rationale": rationale}
                }

                all_query_info.update(query_info)

            # Alternatively, the intent itself may be a query intent
            elif intent_name.startswith("find_"):
                rationale, entity = self.get_query_entity_and_rationale(
                    intents[intent_number],
                    syntax,
                    intent_list[intent_number],
                    intents[intent_number]["entity_name"],
                    intent_args,
                    intent_list,
                )

                # We already have the query intent
                query = intent_list[intent_number].strip()

                query_info = {
                    intent_name: {"query": query, "entity": entity, "rationale": rationale}
                }
                all_query_info.update(query_info)

        return all_query_info

    def remove_queries_for_query_intents(
        self, rules_to_be_applied_by_intent: List[List[Any]], intents: List[Dict[str, Any]]
    ) -> List[List[Any]]:
        # We do not have query unhappy paths for query intents
        rules_by_intent_no_double_query = []

        for i, rules_intent in enumerate(rules_to_be_applied_by_intent):
            new_rules_intent = []
            for rule in rules_intent:
                intent_is_query = intents[i]["command"].startswith("find_")

                if not intent_is_query:
                    new_rules_intent.append(rule)

            rules_by_intent_no_double_query.append(new_rules_intent)

        return rules_by_intent_no_double_query

    def assign_rules_to_intent(
        self,
        unhappy_paths_per_intent: List[int],
        rules_to_be_applied: List[str],
        intents: List[Any],
    ):
        # We extract a list of conversation rules based on the rule labels
        rules_to_be_applied_after_lookup = []

        for rule_text in rules_to_be_applied:
            for rule in self.rules:
                if rule_text == rule.label:
                    rules_to_be_applied_after_lookup.append(rule)

        # We apportion these rules to intents
        rules_to_be_applied_by_intent = self.apportion_rules_to_intents(
            unhappy_paths_per_intent, rules_to_be_applied_after_lookup
        )

        # We do not have query unhappy paths for query intents
        rules_to_be_applied_by_intent = self.remove_queries_for_query_intents(
            rules_to_be_applied_by_intent, intents
        )

        return rules_to_be_applied_by_intent

    def validation(
        self,
        full_intent_list: List[str],
        rules_to_be_applied_by_intent: List[List[Any]],
        intents: List[Dict[str, Any]],
        all_unhappy_path_args: List[List[List[str]]],
        all_selected_examples: List[List[str]],
        all_ssa_examples: List[List[str]],
    ) -> None:
        # Error handling
        if len(rules_to_be_applied_by_intent) != len(full_intent_list):
            raise StageExecutionException(
                "Rules by intents list does not have the correct number of intents"
            )
        if len(intents) != len(full_intent_list):
            raise StageExecutionException(
                "List of intents does not have the correct number of intents"
            )
        if len(all_unhappy_path_args) != len(full_intent_list):
            raise StageExecutionException(
                "List of unhappy path args does not have the correct number of intents"
            )
        if len(all_selected_examples) != len(full_intent_list):
            raise StageExecutionException(
                "List of examples does not have the correct number of intents"
            )
        if len(all_ssa_examples) != len(full_intent_list):
            raise StageExecutionException(
                "List of ssa examples does not have the correct number of intents"
            )

    def print_conversation_plan(
        self,
        full_intent_list: List[str],
        rules_to_be_applied_by_intent: List[List[Any]],
        all_query_info: Dict[str, Dict[str, str]],
        all_unhappy_path_args: List[List[List[str]]],
    ) -> None:
        print("\n\n --- --- --- --- --- Conversation plan --- --- --- --- --- \n")
        print("\n  In total, there are:", len(full_intent_list), "intents:\n")
        for i, intent in enumerate(full_intent_list):
            print("  Intent " + str(i + 1) + ") " + intent)
        print("\n\n --- --- --- --- \n")

        print("\n  The following intents also have unhappy paths:\n")
        for i, intent in enumerate(full_intent_list):
            if rules_to_be_applied_by_intent[i]:
                rule_labels = [x.label for x in rules_to_be_applied_by_intent[i]]
                print("  Intent " + str(i + 1) + ") ")
                print("     - We have apply the rules:", rule_labels)
                print(
                    "     - Each rule is applied to these corresponding args:",
                    all_unhappy_path_args[i],
                )
        print("\n\n --- --- --- --- \n")

        print("\n  We have the following queries supporting the conversation:\n")
        for intent, query_dict in all_query_info.items():
            print("For intent:", intent, "we have:", query_dict)

        print("\n\n --- --- --- --- --- End of plan --- --- --- --- --- \n\n")

    def __call__(
        self,
        intent_path_reasoning: str,
        intents: List[Dict[str, Any]],  # List of remaining intents
        full_intent_list: List[str],  # List of intents with all slot values
        used_intent_args: List[List[str]],  # Slots used for each intent,
        rules_to_be_applied: List[str],
    ) -> Dict[str, str]:  # type: ignore
        """
        We generate rules that the conversation should follow, including the desired final command
        """
        # We allocate unhappy path rules to intents
        unhappy_paths_per_intent = self.allocate_number_rules_to_intents(
            len(rules_to_be_applied), len(intents)
        )

        # Check all unhappy paths rules are valid
        for rule in rules_to_be_applied:
            rule_exists = False
            for conv_rule in self.rules:
                if conv_rule.label == rule:
                    rule_exists = True
            if not rule_exists:
                raise StageExecutionException("Rule: " + rule + ", does not exist")

        # We provide a list of conversation rules for each intent
        rules_to_be_applied_by_intent = self.assign_rules_to_intent(
            unhappy_paths_per_intent, rules_to_be_applied, intents
        )

        # We find examples for the lucid/user prompts based on the unhappy path
        # .. and find the slots the unhappy paths are applied to)
        (
            all_unhappy_path_args,
            all_selected_examples,
            all_ssa_examples,
        ) = self.get_unhappy_path_args_and_examples_by_intent(
            intents, used_intent_args, rules_to_be_applied_by_intent
        )

        # We find queries, the corresponding entities they return, and a rationale for each query
        all_query_info = self.get_all_query_info(
            intents, rules_to_be_applied_by_intent, full_intent_list, used_intent_args
        )

        # Validation checks
        self.validation(
            full_intent_list,
            rules_to_be_applied_by_intent,
            intents,
            all_unhappy_path_args,
            all_selected_examples,
            all_ssa_examples,
        )

        # Printing an output of the conversation plan
        self.print_conversation_plan(
            full_intent_list, rules_to_be_applied_by_intent, all_query_info, all_unhappy_path_args
        )

        return {
            "rules_to_be_applied_by_intent": rules_to_be_applied_by_intent,
            "unhappy_path_args": all_unhappy_path_args,
            "examples": all_selected_examples,
            "ssa_examples": all_ssa_examples,
            "query_info": all_query_info,
        }
