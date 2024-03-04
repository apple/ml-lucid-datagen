#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import datasets
import json
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm


def load_lucid(tokenizer, all_intents, include_tools=False, oracle_bool=True):
    with open("../lucid_v1.0/LUCID_data.json", "r") as json_file:
        lucid = json.load(json_file)

    retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")
    if not oracle_bool:
        embeddings_dict = get_embeddings_for_all_intents(retrieval_model, all_intents)
    else:
        embeddings_dict = None

    lucid = _process_lucid(
        lucid, tokenizer, all_intents, include_tools, oracle_bool, retrieval_model, embeddings_dict
    )
    lucid = datasets.Dataset.from_dict(lucid)

    return lucid


def get_embeddings_for_all_intents(retrieval_model, all_intents):
    all_intents_dict = {}

    for intent in all_intents:
        intent_str = intent.replace("_", " ")
        intent_embedding = retrieval_model.encode(intent_str)
        all_intents_dict[intent] = intent_embedding

    return all_intents_dict


def _process_lucid(
    lucid, tokenizer, all_intents, include_tools, oracle_bool, retrieval_model, embeddings_dict
):
    all_context = []
    all_target = []
    all_split = []

    background_tup, example_tup = _get_prompt_templates(tokenizer)

    for i, conv in enumerate(lucid):
        context = ""

        for turn in conv["turns"]:
            if turn["author"] == "System":
                context += "\n" + str(turn["index"]) + " "

                if include_tools:
                    tool_prompt = _get_all_tool_prompts(
                        context,
                        turn["expression"],
                        all_intents,
                        oracle_bool,
                        retrieval_model,
                        embeddings_dict,
                    )
                    tool_tup = (tool_prompt, get_tok_len(tool_prompt, tokenizer))
                else:
                    tool_prompt = ""
                    tool_tup = ("", 0)

                context_tup = _truncate_context_and_incl_tools(context, tool_tup, tokenizer)
                prompt = _combine_prompts(
                    background_tup, example_tup, context_tup, tokenizer.model_max_length
                )

                all_context.append(prompt)
                all_target.append(turn["expression"])
                all_split.append(conv["split"])

                context += turn["expression"]

            if turn["author"] == "User":
                context += "\nuser: " + turn["query"]

            if turn["author"] == "Signal":
                context += "\n" + str(turn["index"]) + " " + turn["dialog"]

            if turn["author"] == "Response":
                context += "\nresponse: " + turn["text"]

    return {"context": all_context, "target": all_target, "split": all_split}


def _combine_prompts(background_tup, example_tup, context_tup, max_len):
    if background_tup[1] + example_tup[1] + context_tup[1] < max_len:
        prompt = background_tup[0] + example_tup[0] + context_tup[0]

    elif background_tup[1] + context_tup[1] < max_len:
        prompt = background_tup[0] + context_tup[0]

    else:
        prompt = context_tup[0]

    return prompt


def _truncate_context_and_incl_tools(context, tools_tup, tokenizer):
    max_len = tokenizer.model_max_length

    context_len = get_tok_len(context, tokenizer)

    context = context[context.find("\n") + 1 :]

    while context_len >= max_len - tools_tup[1] - 10:
        context = context[context.find("\n") + 1 :]
        context_len = get_tok_len(context, tokenizer)

    context = "\n" + tools_tup[0] + "\n\n" + context
    context_len = get_tok_len(context, tokenizer)

    return (context, context_len)


def _get_prompt_templates(tokenizer):
    background_prompt = """You are a smart AI assistant who is responsible for writing system commands to describe what the user has asked for. Your job is to write the next system command based on the latest user turn, considering the conversation so far.\n"""

    example_prompt = """
Here is an example conversation:
user: When I last ordered flowers for my mum, what size did I get her?
1 find_sent_flowers(recipients="mum", date_of_sent_flowers="last ordered")
2 perform(x1)
3 say(x2)
response: You bought your mum large flowers on 13th April, with the message "happy mothers day"
user: fantastic! same again please
4 send_flowers(recipients="Mum", size="large", content="Happy mothers day")
5 Hint("Ask to confirm", ref=x4)
6 say(x5)
response: Are you ready to confirm sending the flowers?
user: Do it
7 confirm(x6)
8 perform(x7)
9 say(x8)
response: Flowers ordered

Now it's your turn:
"""

    max_len = tokenizer.model_max_length

    background_len = get_tok_len(background_prompt, tokenizer)
    example_len = get_tok_len(example_prompt, tokenizer)

    return ((background_prompt, background_len), (example_prompt, example_len))


def _get_all_tool_prompts(
    context, expression, all_intents, oracle_bool, retrieval_model, embeddings_dict
):
    intents_with_index = []

    # We find which intents are already present in the conversation
    for line in context.split("\n"):
        for intent in all_intents:
            if intent + "(" in line:
                idx = line[: line.find(" ")]
                intents_with_index.append((intent, idx))

    intents_performed = []
    # Now we check if any of these intents have already been performed
    for line in context.split("\n"):
        for intent, idx in intents_with_index:
            if "perform(x" + str(idx) + ")" in line:
                intents_performed.append(intent)

    intents = [x[0] for x in intents_with_index if x[0] not in intents_performed]

    # Finally, we get a tool for the current system turn.
    # .. This is through retrieval or using an oracle
    intent_for_last_user_utterance = retrieve_latest_tool(
        context, expression, all_intents, oracle_bool, retrieval_model, embeddings_dict
    )

    if intent_for_last_user_utterance:
        if intent_for_last_user_utterance not in intents:
            intents.append(intent_for_last_user_utterance)

    tools_prompt = get_tools_for_intents(intents)

    return tools_prompt


def retrieve_latest_tool(
    context, expression, all_intents, oracle_bool, retrieval_model, embeddings_dict
):
    if oracle_bool:
        for intent in all_intents:
            if intent + "(" in expression:
                return intent

        return None
    else:
        score_per_intent = {}
        last_user_mention = context.rfind("user:")
        last_user_utterance = context[last_user_mention : context.rfind("\n")].replace("user:", "")
        user_embedding = retrieval_model.encode(last_user_utterance)
        best_sim_score = 0

        for intent in all_intents:
            intent_embedding = embeddings_dict[intent]
            cos_sim = dot(intent_embedding, user_embedding) / (
                norm(user_embedding) * norm(intent_embedding)
            )

            if cos_sim > best_sim_score:
                best_intent = intent
                best_sim_score = cos_sim

        return best_intent


def get_tools_for_intents(intents):
    OOD_INTENTS = [
        "add_payment_card",
        "create_direct_debit",
        "find_bills",
        "find_direct_debits",
        "find_invoices",
        "find_payment_cards",
        "find_transactions",
        "pay_bill",
        "send_invoice",
        "transfer_money",
    ]

    tools = []

    for intent in intents:
        if intent in OOD_INTENTS:
            loc = "../lucid_v1.0/toolbox_intents_heldout"
        else:
            loc = "../lucid_v1.0/toolbox_intents"
        with open(loc + "/" + intent + ".json", "r") as json_file:
            intent_json = json.load(json_file)

        intent_str = intent_json["command"] + "("

        # We don't need information from the toolbox about optional slots
        for arg, arg_dict in intent_json["args"].items():
            if intent_str[-1] != "(":
                intent_str += ", "
            intent_str += arg + ": " + arg_dict["type"]

        intent_str += ")"
        tools.append(intent_str)

    if len(tools) > 1:
        tools_text = "Information about the following tools may help: " + str(tools)
    elif len(tools) == 1:
        tools_text = "Information about the following tool may help: " + str(tools[0])
    else:
        tools_text = ""

    return tools_text


def get_tok_len(string, tokenizer):
    return len(tokenizer.encode_plus(string)["input_ids"])
