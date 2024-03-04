#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from fuzzywuzzy import fuzz
import numpy as np


def compute_metrics_with_extra(tokenizer, all_intents):
    def compute_metrics(eval_predictions):

        context_enc = eval_predictions.inputs
        preds_enc = eval_predictions.predictions[0]
        labels_enc = eval_predictions.label_ids

        special_tokens = tokenizer.all_special_tokens

        intent_acc, intent_acc_turn, intent_acc_conv = get_active_intent_acc(
            context_enc, preds_enc, labels_enc, special_tokens, all_intents, tokenizer
        )
        exact_match, exact_match_turn, exact_match_conv = find_exact_match(
            context_enc, preds_enc, labels_enc, special_tokens, tokenizer
        )
        fuzzy_slot_acc, fuzzy_slot_turn, fuzzy_slot_conv = find_fuzzy_slots_joint_acc(
            context_enc, preds_enc, labels_enc, special_tokens, all_intents, tokenizer
        )
        fuzzy_goal_joint_acc = find_fuzzy_goal_joint_acc(
            context_enc, preds_enc, labels_enc, special_tokens, all_intents, tokenizer
        )

        print(
            {
                "intent_acc": intent_acc,
                "joint_intent_acc_conv": intent_acc_conv,
                "exact_match": exact_match,
                "exact_match_turn": exact_match_turn,
                "exact_match_conv": exact_match_conv,
                "fuzzy_slot_acc": fuzzy_slot_acc,
                "joint_fuzzy_slot_conv": fuzzy_slot_conv,
                "fuzzy_goal_joint_acc": fuzzy_goal_joint_acc,
            }
        )

        return {
            "intent_acc": intent_acc,
            "joint_intent_acc_conv": intent_acc_conv,
            "exact_match": exact_match,
            "exact_match_turn": exact_match_turn,
            "exact_match_conv": exact_match_conv,
            "fuzzy_slot_acc": fuzzy_slot_acc,
            "joint_fuzzy_slot_conv": fuzzy_slot_conv,
            "fuzzy_goal_joint_acc": fuzzy_goal_joint_acc,
        }

    return compute_metrics


def eos_fix(str1, str2):

    if str1.startswitih(str2):
        str1 = str2

    return str1


def decode_from_list(ls, model_preds, tokenizer):

    if model_preds:
        t = torch.argmax(torch.tensor(ls), dim=1)
    else:
        t = torch.tensor([x if x != -100 else 0 for x in ls])

    output = tokenizer.decode(t, skip_special_tokens=False)
    output = output[: output.find("</s>")]
    return output


def compile_with_product(compiled_totals, total_to_compile, reset_after_compile=True):

    if total_to_compile:
        compiled_totals.append(np.product(total_to_compile))

    if reset_after_compile:
        total_to_compile = []

    return compiled_totals, total_to_compile


def find_fuzzy_slots_joint_acc(
    contexts_enc, preds_enc, labels_enc, special_tokens, all_intents, tokenizer
):

    score_all_turns = []
    in_turn = []
    in_conv = []
    in_turn_total = []
    in_conv_total = []

    for example in range(len(preds_enc)):

        pred = (
            compile_and_rm_special_tokenss(preds_enc[example], special_tokens, True, tokenizer)
            .strip()
            .lower()
        )
        label = (
            compile_and_rm_special_tokenss(labels_enc[example], special_tokens, False, tokenizer)
            .strip()
            .lower()
        )
        context = compile_and_rm_special_tokenss(
            contexts_enc[example], special_tokens, False, tokenizer
        ).strip()
        context = strip_examples(context)

        current_sys_turn_no = int(context[context.rfind("\n") + 1 :].strip())

        if current_sys_turn_no == 1:

            in_turn, in_turn_total = compile_with_product(in_turn, in_turn_total)
            in_conv, in_conv_total = compile_with_product(in_conv, in_conv_total)

            in_turn_total = []
            in_conv_total = []

        preds_slots = parsed_cmd_into_slots(pred, current_sys_turn_no, all_intents, context)
        label_slots = parsed_cmd_into_slots(label, current_sys_turn_no, all_intents, context)

        if preds_slots or label_slots:
            if preds_slots and label_slots:

                score = perform_fuzzy_matching(preds_slots, label_slots)

                score_all_turns.append(score)
                in_turn_total.append(score)
                in_conv_total.append(score)
            else:
                score_all_turns.append(0)
                in_turn_total.append(0)
                in_conv_total.append(0)

        # Reset indicates for new turn/conversation
        if label.startswith("say("):
            in_turn, in_turn_total = compile_with_product(in_turn, in_turn_total)

    in_turn, in_turn_total = compile_with_product(in_turn, in_turn_total)
    in_conv, in_conv_total = compile_with_product(in_conv, in_conv_total)

    return np.mean(score_all_turns), np.mean(in_turn), np.mean(in_conv)


def find_fuzzy_goal_joint_acc(contexts, preds, labels, special_tokens, all_intents, tokenizer):

    score_all_turns = []
    preds_conversation_slots = {}
    label_conversation_slots = {}

    for example in range(len(preds)):

        pred = (
            compile_and_rm_special_tokenss(preds[example], special_tokens, True, tokenizer)
            .strip()
            .lower()
        )
        label = (
            compile_and_rm_special_tokenss(labels[example], special_tokens, False, tokenizer)
            .strip()
            .lower()
        )
        context = compile_and_rm_special_tokenss(
            contexts[example], special_tokens, False, tokenizer
        ).strip()
        context = strip_examples(context)

        current_sys_turn_no = int(context[context.rfind("\n") + 1 :].strip())

        if current_sys_turn_no == 1:
            preds_conversation_slots = {}
            label_conversation_slots = {}

        preds_slots = parsed_cmd_into_slots(pred, current_sys_turn_no, all_intents, context)
        label_slots = parsed_cmd_into_slots(label, current_sys_turn_no, all_intents, context)

        if preds_slots:
            preds_conversation_slots.update(preds_slots)
        if label_slots:
            label_conversation_slots.update(label_slots)

        intent_preds_slots = get_slots_for_active_intent(
            pred, preds_conversation_slots, context, current_sys_turn_no, all_intents
        )
        intent_label_slots = get_slots_for_active_intent(
            label, label_conversation_slots, context, current_sys_turn_no, all_intents
        )

        if preds_slots or label_slots:
            if preds_slots and label_slots:
                score = perform_fuzzy_matching(intent_preds_slots, intent_label_slots)
                score_all_turns.append(score)
            else:
                score_all_turns.append(0)

    return np.mean(score_all_turns)


# TODO Note: we assume that non-parsable lines are the same as not giving any slot values


def get_slots_for_active_intent(command, dict_of_slots, context, current_sys_turn_no, all_intents):

    current_intent = get_intent_from_cmd(command, context, current_sys_turn_no, all_intents)

    if not current_intent:
        return dict_of_slots

    reduced_dict_of_slots = {}
    for slot, value in dict_of_slots.items():
        if slot.startswith(current_intent):
            reduced_dict_of_slots[slot] = value

    return reduced_dict_of_slots


def perform_fuzzy_matching(preds, labels):

    if len(preds) != len(labels):
        return 0

    vals = []

    all_keys = list(set(list(preds.keys()) + list(labels.keys())))
    for slot in all_keys:
        if slot in preds.keys() and slot in labels.keys():
            vals.append(fuzzy_string_match(preds[slot], labels[slot]))
        else:
            vals.append(0)

    return np.product(vals)


def fuzzy_string_match(str_ref, str_hyp):
    """Returns fuzzy string similarity score in range [0.0, 1.0]."""

    if str_ref[0] == '"' and str_ref[-1] == '"':
        str_ref = str_ref[1:-1]
    if str_hyp[0] == '"' and str_hyp[-1] == '"':
        str_hyp = str_hyp[1:-1]

    # The higher the score, the higher the similarity between the two strings.
    return fuzz.token_sort_ratio(str_ref, str_hyp) / 100.0


def parsed_cmd_into_slots(command, current_sys_turn_no, all_intents, context):

    for intent in all_intents:

        if intent + "(" in command:
            return parse_slots_from_intent_call(command, intent, current_sys_turn_no)

    for idx in range(current_sys_turn_no):
        if command.strip().startswith("x" + str(idx) + "."):
            return parse_slots_from_assignent(command, idx, context, all_intents)

    return None


def parse_slots_from_assignent(command, idx, context, all_intents):

    command = command.strip()
    slots = command[: command.find("=")]

    slots_list = [x.strip() for x in slots.split(",")]
    slots_list = update_slots_with_original_intent(slots_list, context, all_intents)

    values = command[command.find("=") + 1 :]

    values_split_points = get_comma_split_points(values)

    values_list = [
        values[i + 1 : j].strip().lower()
        for (i, j) in zip([-1] + values_split_points, values_split_points + [len(command)])
    ]

    if len(slots_list) != len(values_list):
        return None

    else:
        return_dict = {}
        for i in range(len(slots_list)):
            return_dict[slots_list[i]] = values_list[i]
        return return_dict


def update_slots_with_original_intent(slots, context, all_intents):

    new_slots_list = []

    for slot in slots:
        idx = slot[1 : slot.find(".")]
        intent = find_intent_from_context(context, idx, all_intents)
        if intent:
            new_slots_list.append(intent + "." + slot)
        else:
            new_slots_list.append("UNKNOWN_SLOT")

    return new_slots_list


def parse_slots_from_intent_call(command, intent, current_sys_turn_no):

    intent = command[: command.find("(")]
    command = command[command.find("(") + 1 : -1]

    if command.strip() == "":
        return None
    split_points = get_comma_split_points(command)
    assignments = [
        command[i + 1 : j].strip().lower()
        for (i, j) in zip([-1] + split_points, split_points + [len(command)])
    ]

    return_dict = {}

    # We label an intent with its turn idx, in case of having two of the same intents in a conversation
    for x in assignments:
        return_dict["turn" + str(current_sys_turn_no) + "." + intent + "." + x[: x.find("=")]] = x[
            x.find("=") + 1 :
        ]

    return return_dict


def get_comma_split_points(command):

    quotes_opened = False
    comma_splits = []

    for idx, char in enumerate(command):

        if char == '"':
            quotes_opened = not quotes_opened
        if not quotes_opened and char == ",":
            comma_splits.append(idx)

    return comma_splits


def find_exact_match(contexts, preds, labels, special_tokens, tokenizer):

    total = []
    in_turn = []
    in_conv = []
    in_turn_total = []
    in_conv_total = []

    for example in range(len(preds)):
        context = compile_and_rm_special_tokenss(
            contexts[example], special_tokens, False, tokenizer
        ).strip()
        context = strip_examples(context)
        pred = (
            compile_and_rm_special_tokenss(preds[example], special_tokens, True, tokenizer)
            .strip()
            .lower()
        )
        label = (
            compile_and_rm_special_tokenss(labels[example], special_tokens, False, tokenizer)
            .strip()
            .lower()
        )

        # Storing totals for conversations/turns
        current_sys_turn_no = int(context[context.rfind("\n") + 1 :].strip())

        if current_sys_turn_no == 1:

            in_turn, in_turn_total = compile_with_product(in_turn, in_turn_total)
            in_conv, in_conv_total = compile_with_product(in_conv, in_conv_total)

        if pred == label:
            total.append(1)
            in_turn_total.append(1)
            in_conv_total.append(1)
        else:
            total.append(0)
            in_turn_total.append(0)
            in_conv_total.append(0)

        # Reset indicates for new turn/conversation
        if label.startswith("say("):
            in_turn, in_turn_total = compile_with_product(in_turn, in_turn_total)

    in_turn, in_turn_total = compile_with_product(in_turn, in_turn_total)
    in_conv, in_conv_total = compile_with_product(in_conv, in_conv_total)

    return np.mean(total), np.mean(in_turn), np.mean(in_conv)


def get_active_intent_acc(contexts, preds, labels, special_tokens, all_intents, tokenizer):

    total = []
    in_turn = []
    in_conv = []
    in_turn_total = []
    in_conv_total = []

    for example in range(len(preds)):
        context = compile_and_rm_special_tokenss(
            contexts[example], special_tokens, False, tokenizer
        ).strip()
        context = strip_examples(context)
        pred = compile_and_rm_special_tokenss(
            preds[example], special_tokens, True, tokenizer
        ).strip()
        label = compile_and_rm_special_tokenss(
            labels[example], special_tokens, False, tokenizer
        ).strip()

        # Storing totals for conversations/turns
        current_sys_turn_no = int(context[context.rfind("\n") + 1 :].strip())

        if current_sys_turn_no == 1:

            in_turn, in_turn_total = compile_with_product(in_turn, in_turn_total)
            in_conv, in_conv_total = compile_with_product(in_conv, in_conv_total)

        pred_intent = get_intent_from_cmd(pred, context, current_sys_turn_no, all_intents)
        label_intent = get_intent_from_cmd(label, context, current_sys_turn_no, all_intents)

        if pred_intent or label_intent:
            if pred_intent == label_intent:
                total.append(1)
                in_turn_total.append(1)
                in_conv_total.append(1)
            else:
                total.append(0)
                in_turn_total.append(0)
                in_conv_total.append(0)

        # Reset indicates for new turn/conversation
        if label.startswith("say("):
            in_turn, in_turn_total = compile_with_product(in_turn, in_turn_total)

    in_turn, in_turn_total = compile_with_product(in_turn, in_turn_total)
    in_conv, in_conv_total = compile_with_product(in_conv, in_conv_total)

    return np.mean(total), np.mean(in_turn), np.mean(in_conv)


def get_intent_from_cmd(command, context, current_sys_turn_no, all_intents):

    for intent in all_intents:
        if command.startswith(intent + "("):
            return "turn" + str(current_sys_turn_no) + "." + intent

    intent = search_for_assignment_intent(command, context, current_sys_turn_no, all_intents)
    if intent:
        return intent

    return None


def search_for_assignment_intent(command, context, current_sys_turn_no, all_intents):

    for idx in range(current_sys_turn_no):
        if command.startswith("x" + str(idx) + "."):
            return find_intent_from_context(context, idx, all_intents)


def find_intent_from_context(context, idx, all_intents):

    for line in context.split("\n"):
        line = line.strip()
        if line.startswith(str(idx) + " "):
            for intent in all_intents:
                if line[line.find(" ") + 1 :].startswith(intent + "("):
                    return "turn" + str(idx) + "." + intent

    return None


def compile_and_rm_special_tokenss(tok_list, special_tokens, model_preds, tokenizer):

    command = decode_from_list(tok_list, model_preds, tokenizer)

    for tok in special_tokens:
        if tok != "\n":
            command = command.replace(tok, "")

    command = command.replace("<extra_id_0>", "")

    return command


def strip_examples(command):

    if "Now it's your turn:" in command:
        command = command[command.find("Now it's your turn:") :].replace("Now it's your turn:", "")

    return command
