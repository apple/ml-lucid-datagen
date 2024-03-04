#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from lucid_generate_data.stage import Stage
from typing import Any, Dict, List, Optional, Tuple
from random import Random
from jinja2 import Environment
from lucid_generate_data.openai_call import make_openai_call
from lucid_generate_data.utils.commands import (
    parse_system_function_call,
    check_parsed_slots_vs_intent_json,
)

from lucid_generate_data.code_gen import intent_to_func_def

NO_RETRIES = 6


class GenerateFullIntents(Stage):
    def __init__(self) -> None:
        self.rng = Random()
        self.prob_optional_arg = 0.2

    def _prompt_getting_context_for_future_cmd(self):
        prompt = """
        The task is to find a plausible reason why a user of an AI virtual assistant could make a new request as a direct consequence of previous requests.

        The reason for the new request should: 
        1) Reference what happened in the previous requests
        2) Be plausible real world scenario
        3) Be creative. You can reference new things/people/places, as long as there is a connection to the previous requests.
        
        You should give a 1-2 sentence reason for the new request after 'Plausible reason for new request'. This must reference the old request.

        ```
        Example 1:

        Previous requests (most recent at end):
        order_ride(destination="the office", num_passengers=2, origin="home")

        New request:
        send_message

        Plausible reason for new request:
        I'm on the way to the office with our son Josh. I need to text his Dad to let them know.

        Example 2:

        Previous requests (most recent at end):
        phone_call(recipient="Dad") phone_call(recipient="Mum")

        New request:
        order_shopping

        Plausible reason for new request:
        Mum just told me on the phone she broke the microwave again, so I'll order another one now

        Example 3:

        Previous requests (most recent at end):
        book_uber(destination="Westfield") send_message("Really sorry I forgot about your birthday, I'll buy you a present today")

        New request:
        set_reminder

        Plausible reason for new request:
        Setting a reminder for 10 minutes when the Uber should arrive

        Now your turn:
        
        Previous requests (most recent at end):
        {{old_requests}}

        New request:
        {{new_request}}

        Plausible reason for new request:
        """.strip()

        jinga_template = Environment().from_string(prompt)

        return jinga_template

    def _get_likely_value_given_context(
        self,
        arg: str,
        command_context: str,
        past_intent_commands: List[str],
        var_type: str,
        next_intent: str,
    ) -> str:
        previous_commands = " ".join(past_intent_commands)

        prompt = f"""
        There is a user interacting with a virtual AI assistant.  Your task is to predict sensible argument values for the command that the user wants. These argument values must align to the type specified.
    
        You must provide a sensible argument value (based on the context).

        *** Example 1:
        
        Background context: I need to text Josh's mum to let him know he's not well today, and so he is coming to the office with me instead of going to school
        Previous commands: book_ride(destination="office", origin="home", no_passengers=2)
        Command: send_message
        Argument: recipient
        Type: str
        Value: "Josh's mum"

        *** Example 2:

        Background context: I need to set a reminder to find my glasses before I go to the park
        Previous commands: send_text(recipient="Lisa", content="Dont forget about the park trip") book_ride(destination="park", time="5pm")
        Command: set_reminder
        Argument: time
        Type: str
        Value: "4:45pm"

        *** Example 3: 

        Background context: I need to book a table for everyone in the taxi
        Previous commands: book_uber(destination="St James Square", no_passengers=5)
        Command: book_table
        Argument: no_guests
        Type: int
        Value: 5

        *** Now it's your turn:

        Background context: {command_context}
        Previous commands: {previous_commands}
        Command: {next_intent}
        Argument: {arg}
        Type: {var_type}
        Value:
        """.strip()

        jinga_template = Environment().from_string(prompt)
        full_prompt = jinga_template.render(
            command_context=command_context,
            previous_commands=previous_commands,
            next_intent=next_intent,
            var_type=var_type,
            arg=arg,
        )

        val = make_openai_call("get_likely_slot_value_given_context", prompt=full_prompt).strip()

        return val

    def get_context_for_next_intent(self, past_intent_commands: List[str], new_request: str) -> str:
        """
        We generate a context for doing the new request after the old (current) request
        """

        past_intent_commands = " ".join(past_intent_commands)
        jinga_template = self._prompt_getting_context_for_future_cmd()
        full_prompt = jinga_template.render(
            past_intent_commands=past_intent_commands, new_request=new_request
        )
        command_context = make_openai_call("get_context_for_future_command", prompt=full_prompt)

        return command_context

    def check_function_valid_types(self, command, intent):
        is_valid_json, parsed_slot_data_types = parse_system_function_call(command)

        is_correct_slot_data_types = check_parsed_slots_vs_intent_json(
            parsed_slot_data_types, intent
        )

        if is_valid_json and is_correct_slot_data_types:
            return True

        return False

    def find_slot_vals_using_prev_intents(
        self,
        past_intent_commands: List[str],
        next_intent: dict,
    ) -> (list, list):
        """
        We add a conversation rule to deal with task switching (specifying the next task)
        """

        # We generate a context for why the user wants to do the next intent, considering
        # .. the intents performed so far. This will inform the slot values of the intent.
        command_context = self.get_context_for_next_intent(past_intent_commands, next_intent)

        # We generate slots and values for the inten
        no_retries = 0
        while no_retries <= NO_RETRIES:
            (
                future_command,
                future_command_args,
            ) = self.generate_full_intent(next_intent, command_context, past_intent_commands)

            if self.check_function_valid_types(future_command, next_intent):
                break
            else:
                no_retries += 1
                print("Retry (predicing slots for next intent):", no_retries)

        assert no_retries <= NO_RETRIES

        # We get the command (with slot values), along with the slots

        return (future_command, future_command_args, command_context)

    def _populate_value(
        self,
        arg: str,
        values: List[str],
        past_intent_commands: List[str],
        var_type: str,
        next_intent: str,
        command_context: Optional[str],
    ) -> str:
        """
        We give values for the slot (the conversation should end with this slot value)
        """

        # We use an LLM to generate inital slot values given previous intents
        if command_context is not None:
            val = self._get_likely_value_given_context(
                arg, command_context, past_intent_commands, var_type, next_intent
            )

        # .. except in the case of our first intent, where we sample slot values from
        # .. its intent JSON
        else:
            val = self.rng.choice(values)

        return val

    def generate_full_intent(
        self,
        intent_json: Dict[str, Dict[str, Any]],
        command_context: Optional[str] = None,
        past_intent_commands: Optional[List[str]] = None,
    ) -> Tuple[str, List[str], List[str]]:
        """
        We generate the desired final command to be predicted at the end of the conversation
        """
        next_intent = intent_json["command"]
        used_slots: List[str] = []  # Args to be used by the next intent
        set_slot_values = []  # Slots and slot values

        # We need to decide which arguments will be present in this desired final command
        for slot_name, slot_dict in intent_json["args"].items():
            # All required args must be included
            if not slot_dict["optional"]:
                used_slots.append(slot_name)
                val = self._populate_value(
                    slot_name,
                    slot_dict["values"],
                    past_intent_commands,
                    slot_dict["type"],
                    next_intent,
                    command_context,
                )

                set_slot_values.append(f"{slot_name}={val}")

            # We also consider some optional args
            else:
                if self.rng.random() <= self.prob_optional_arg:
                    used_slots.append(slot_name)
                    val = self.rng.choice(slot_dict["values"])
                    if isinstance(val, str):
                        val = '"' + val + '"'

                    val = self._populate_value(
                        slot_name,
                        slot_dict["values"],
                        past_intent_commands,
                        slot_dict["type"],
                        next_intent,
                        command_context,
                    )

                    set_slot_values.append(f"{slot_name}={val}")

        next_intent += "(" + ", ".join(set_slot_values) + ")"

        return next_intent, used_slots

    def make_slot_values_more_realistic(
        self, syntax, command, args, is_query, command_context=None
    ):
        if is_query:
            prompt = """
            A user is interacting with an AI virtual assistant called Lucid, and the user wants Lucid to perform the following command:
            {{command}}

            Your job is to make this command more coherent and realistic. This can involve rephrasing, or completely changing the slot values. The slot values must be consistent with each other, so the command makes sense. The slot values should also reflect how a user would naturally express these.

            As the intent is a query intent, to make it more realistic, reduce the number of slots mentioned to at most 1 or 2. Only provide slots that a user is likely to ask about when trying when doing a query.

            Example 1:
            Old command: find_restaurant(date="Saturday", cuisine="Asian", location="Cambridge", disabled_access=True, rating=2)
            New command: find_restaurant(cuisine="Japanese", location="Cambridge", rating=4)

            Example 2:
            Old command: find_email(subject="Hi", recipient="Family", content="Happy birtyday!", date_of_email="Last week")
            New command: find_email(recipient="Mum", date_of_email="Last week")

            Example 3:
            Old command: find_song(band="radiohead", song="wannabe", date_of_song="1998")
            New command: find_song(song="paranoid android")

            Example 4:
            Old command: find_timer(label="cooking", date_of_timer="today", duration="2 hours")
            New command: find_timer(date_of_timer="today")

            You can change the meaning of the slot values, as long as they sound natural, and the command is realistic and coherent.

            You cannot change the data type of any slots, change the slot names, or add new slots. E.g. int slots must remain int slots.

            {{extra_info}}

            The slots for this command must have the following data types:
            {{syntax}}

            Your turn:
            Old command: {{command}}
            New command:
            """.strip()

        else:
            prompt = """
            A user is interacting with an AI virtual assistant called Lucid, and the user wants Lucid to perform the following command:
            {{command}}

            Your job is to make this command more coherent and realistic. This can involve rephrasing, or completely changing the slot values. The slot values must be consistent with each other, so the command makes sense. The slot values should also reflect how a user would naturally express these.

            Example 1:
            Old command: order_meal(food_type="breakfast", restaurant="McDonalds", quantity=10, time="8pm")
            New command: order_meal(food_type="breakfast", restaurant="McDonalds", quantity=2, time="7:30am")

            Example 2:
            Old command: send_email(subject="Hi", content="Hi Dad, looking forward to seeing you tomorrow!", recipient="Mum", urgent=True)
            New command: send_email(subject="Hi", content="Hi Dad, looking forward to seeing you tomorrow!", recipient="Dad", urgent=False)

            Example 3:
            Old command: play_song(band="radiohead", song="wannabe")
            New command: play_song(band="radiohead", song="paranoid android")

            Example 4:
            Old command: book_uber(location="home", date="13/04/2021", no_passengers=5)
            New command: book_uber(location="home", date="On the 13th", no_passengers=2)

            You can change the meaning of the slot values, as long as they sound natural, and the command is realistic and coherent.

            You cannot change the data type of any slots, change the slot names, or add new slots. E.g. int slots must remain int slots.

            {{extra_info}}

            The slots for this command must have the following data types:
            {{syntax}}

            Your turn:
            Old command: {{command}}
            New command:
            """.strip()

        # We make the slot values more realistic (considering all previous intents)
        if command_context:
            extra_info = command_context
        else:
            extra_info = ""

        jinga_template = Environment().from_string(prompt)
        full_prompt = jinga_template.render(extra_info=extra_info, command=command, syntax=syntax)
        new_command = make_openai_call(
            "make_slot_values_more_realistic", prompt=full_prompt
        ).strip()

        return new_command

    def validate_updated_command(self, command, intent_json):
        # We make sure the command is written as a function
        if "(" not in command or ")" not in command:
            print("Realistic commands LLM did not return a valid function: " + str(command))
            return False

        # We make sure the slot names are correct and we check the type of each slot value
        else:
            command_args = command[command.find("(") + 1 : -1]
            command_list = command_args.split(",")
            command_list = [x.strip() for x in command_list]
            command_list = [x.split("=") for x in command_list]

            for arg in command_list:
                # Checking the slot name
                if arg[0] not in intent_json["args"]:
                    print(
                        "Realistic commands LLM generated invalid slot name: "
                        + arg[0]
                        + " from response: "
                        + str(command)
                        + " and args: "
                        + str(intent_json["args"])
                    )
                    return False

                # Checking the data type for each slot value
                else:
                    if '"' in arg[1]:
                        if intent_json["args"][arg[0]]["type"] != "str":
                            print("Expected str, but slot value is: " + str(arg[1]))
                            return False
                    elif arg[1] == "True" or arg[1] == "False":
                        if intent_json["args"][arg[0]]["type"] != "bool":
                            print("Expected bool, but slot value is: " + str(arg[1]))
                            return False
                    elif "." in arg[1]:
                        if intent_json["args"][arg[0]]["type"] != "float":
                            print("Expected float, but slot value is: " + str(arg[1]))
                            return False
                    else:
                        if intent_json["args"][arg[0]]["type"] != "int":
                            print("Expecting integer, but got slot value: " + str(arg[1]))
                            return False
            return True

    def better_links_between_in_intents(self, syntax_all, full_intent_list):
        full_intent_str = ", ".join(full_intent_list)
        syntax_all_str = ", ".join(syntax_all)

        prompt = """
            A user is interacting with an AI virtual assistant called Lucid. Over the conversation, the user requests the following intents:
            {{full_intent_str}}

            Your job is to make sure these commands from a coherent conversation. For example, the slot values in each intent should relate to each other (reflecting a realistic conversation). You may not change the data type of any slots. You also cannot add any new slots, or change the slot names.

            Example 1:
            Old commands: find_notes(title="Birthday gift ideas", date_of_notes="last month"), create_reminder(title="Client meeting", date="this morning", time="9 o\'clock")
            New commands: find_notes(title="Birthday gift ideas", date_of_notes="last month"), create_reminder(title="Buying birthday gift for Anna", date="today", time="11 o\'clock")

            Example 2:
            Old commands: set_reminder(duration="30 minutes") send_message(recipient="Mum", content="Don't forget my birthday!")
            New commands: set_reminder(duration="30 minutes") send_message(recipient="Mum", content="I'll leave in 30 minutes when I've finished packing, see you soon!")

            Example 3:
            Old commands: create_reminder(title="Pick up Groceries", date="10th of May", time="half past two in the afternoon") write_note(title="To-Do List for Birthday Party", reminder="Yes please")
            New commands: create_reminder(title="Pick up Groceries for the Birthday Party", date="10th of May", time="half past two in the afternoon"), write_note(title="To-Do List for the Birthday Party", reminder="Buy goodie bags, prepare games, bake cake and tidy flat")

            You must use the following data types:
            {{syntax_all_str}}

            Your turn:
            Old commands: {{full_intent_str}}
            New commands:
        """

        # We update the predicted slot values for the entire conversation
        jinga_template = Environment().from_string(prompt)
        full_prompt = jinga_template.render(
            full_intent_str=full_intent_str, syntax_all_str=syntax_all_str
        )

        new_intent_str = make_openai_call("replanning_for_coherence", prompt=full_prompt).strip()

        return full_intent_list

    def generate_more_coherent_conversation(self, all_intents_def, full_intent_list, intents):
        no_retries = 0

        # We regenerate the slot values for every intent in the conversation
        while no_retries <= NO_RETRIES:
            full_intent_list = self.better_links_between_in_intents(
                all_intents_def, full_intent_list
            )

            exists_issue = False

            # We again validate the data types for each proposed slot value
            for command, intent in zip(full_intent_list, intents):
                if not self.check_function_valid_types(command, intent):
                    exists_issue = True

            if not exists_issue:
                break
            else:
                no_retries += 1
                print("Retry (step 3):", no_retries)

        assert no_retries <= NO_RETRIES

        return full_intent_list

    def get_intent_values_other_intents(
        self, full_intent_list, used_intent_args, all_intents_def, intents
    ):
        no_retries = 0

        # We now consider all the remaining conversation intents
        for intent in intents[1:]:
            syntax = intent_to_func_def(intent)
            all_intents_def.append(syntax)

            while no_retries <= NO_RETRIES:
                # We initially predict slot values based all the intents up to that point
                command_raw, args, command_context = self.find_slot_vals_using_prev_intents(
                    full_intent_list, intent
                )

                # We then try to make the slot values for that specific intent more realistic
                command = self.make_slot_values_more_realistic(
                    syntax, command_raw, args, command_raw.startswith("find_"), command_context
                )

                # We validate the data types of the proposed slot values
                if self.check_function_valid_types(command, intent):
                    full_intent_list.append(command)
                    used_intent_args.append(args)
                    break
                else:
                    no_retries += 1
                    print("Retry (step 2):", no_retries)

        return full_intent_list, used_intent_args, all_intents_def

    def get_intent_value_first_intent(self, intents):
        full_intent_list = []
        used_intent_args = []
        all_intents_def = []

        # For the first intent, we initially sample values for each slot from the intent JSON
        command_raw, args = self.generate_full_intent(intents[0])
        syntax = intent_to_func_def(intents[0])

        no_retries = 0

        # We ask an LLM to update these slot values, making them more realistic
        while no_retries <= NO_RETRIES:
            command = self.make_slot_values_more_realistic(
                syntax, command_raw, args, command_raw.startswith("find_")
            )

            # We validate the data types of the proposed slot values
            if self.check_function_valid_types(command, intents[0]):
                break
            else:
                no_retries += 1
                print("Retry (step 1):", no_retries)

        full_intent_list.append(command)
        used_intent_args.append(args)
        all_intents_def.append(syntax)

        return full_intent_list, used_intent_args, all_intents_def

    def __call__(
        self,
        intents: List[Dict[str, Any]],  # List of remaining intents
    ) -> Dict[str, str]:  # type: ignore
        """
        We generate rules that the conversation should follow, including the desired final command
        """

        # We generate the slots and slot values used for the first intent
        full_intent_list, used_intent_args, all_intents_def = self.get_intent_value_first_intent(
            intents
        )

        # We now consider any other intents in the conversation
        full_intent_list, used_intent_args, all_intents_def = self.get_intent_values_other_intents(
            full_intent_list, used_intent_args, all_intents_def, intents
        )

        # We now ask an LLM to update the slot values for every intent in the conversation
        # ... with the aim of creating a more reastlic, coherent conversation across intents
        full_intent_list = self.generate_more_coherent_conversation(
            all_intents_def, full_intent_list, intents
        )

        print("Created the plan for the intents in the conversation")
        return {"full_intent_list": full_intent_list, "used_intent_args": used_intent_args}
