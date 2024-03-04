#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from dataclasses import dataclass


@dataclass
class ConversationRule:
    label: str
    template: str
    args_required: int
    disambig_required: bool
    cancel_request: bool
    example: str
    ssa_example: str
    tag: str


QUERY = ConversationRule(
    label="query",
    template="The user wants to: {rationale}. To do this they need to perform the following query: {query}. When doing the query, include [QUERY] at the beginning of the text.",
    args_required=0,
    disambig_required=False,
    cancel_request=False,
    tag="QUERY",
    example="""
# Conversation rules:
1) The user must check for existing sent_flowers entities, asking for find_sent_flowers(recipients="Mum"). When doing the query, include [QUERY] at the beginning of the text.
2) The task the user wants to perform must be: send_flowers(recipients="Mum", size="large", content="Happy mothers day")

# Generated conversation:
user: Send flowers to my mum
lucid: How big should the flowers be?
user: [QUERY] How big were the last flowers I bought for mum?
lucid: Last time you bought flowers for mum was on 13th April, when you bought large flowers with the message "happy mothers day"
user: Large again then
lucid: what should they say?
user: The same message as last time!
lucid: Ready to order the flowers?
user: Yes please!
lucid: Flowers ordered. Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: When I last ordered flowers for my mum, what size did I get her?
2 find_sent_flowers(recipients="mum", date_of_sent_flowers="last ordered")
3 perform(x2)
4 say(x3)
lucid: You bought your mum large flowers on 13th April, with the message "happy mothers day"
user: fantastic! same again please
5 send_flowers(recipients="Mum", size="large", content="Happy mothers day")
6 Hint("Ask to confirm", ref=x5)
7 say(x6)
lucid: Are you ready to confirm sending the flowers?
user: Do it
8 confirm(x5)
9 perform(x5)
10 say(x9)
lucid: Flowers ordered
""",
)

DELAY_CONFIRMATION = ConversationRule(
    label="delay_confirmation",
    template="If asked for confirmation, the user decides not to confirm the task. When giving this answer, include [DELAY_CONFIRMATION] at the beginning of the text.",
    args_required=0,
    disambig_required=False,
    cancel_request=False,
    tag="DELAY_CONFIRMATION",
    example="""
# Conversation rules:
1) If asked for confirmation, the user decides not to confirm the task. When giving this answer, include [DELAY_CONFIRMATION] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="hey!!!")

# Generated conversation:
user: I need to send a message
lucid: Who would you like to send the message to?
user: Please send the message to Tom
lucid: Added Tom as a recipient. What would you like to say in the message?
user: Just say hey!!!
lucid: Message content is "hey!!!", do you want to send it?
user: [DELAY_CONFIRMATION] actually no, hold off sending it for now, I'll check if he's on Slack
lucid: message not sent
user: OK he's not on Slack, ready to send it now
lucid: message sent. Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: set a new alarm for 8:30, and label it swimming get-up
2 create_alarm(time="8:30", label="swimming get-up")
3 Hint("Ask to confirm", ref=x2)
4 say(x3)
lucid: Ready to confirm setting the alarm?
user: Hold off for now, I need to double check the time of my swimming lesson
5 say()
lucid: Alarm has not been set
user: Yep all good, set it now.
6 confirm(x2)
7 perform(x2)
8 say(x7)
lucid: Alarm set
user: end conversation
""",
)

CORRECTION = ConversationRule(
    label="correction",
    template="The user must give an initial value of {arg1}, then later correct themselves and change the value of {arg1}. When giving the correction, include [CORRECTION] at the beginning of the text.",
    args_required=1,
    disambig_required=False,
    cancel_request=False,
    tag="CORRECTION",
    example="""
# Conversation rules:
1) The user must give an initial value of content, then later correct themselves and change the value of content. When giving the correction, include [CORRECTION] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="how are you")

# Generated conversation:
user: Send message to Tom please
lucid: What do you want to say in the message?
user: make the content hey!!!
lucid: Message content is "hey!!!", do you want to send the message?
user: [CORRECTION] change the content to how are you in the message
lucid: Changed message content to "how are you". Are you ready to send?
user: yep
lucid: Message sent! Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: set new timer for 5 minutes
2 set_timer(duration="5 minutes")
3 Hint("Ask to confirm", ref=x2)
4 say(x3)
lucid: Would you like to confirm setting a timer for 5 minutes?
user: change it to 6 minutes
5 x2.duration = "6 minutes"
6 Hint("Ask to confirm", ref=x2)
7 say(x6)
lucid: Timer duration updated, do you want to confirm setting a timer for 6 minutes?
user: Yes please
8 confirm(x2)
9 perform(x2)
10 say(x9)
lucid: Timer set
""",
)


TURN_CORRECTION = ConversationRule(
    label="turn_correction",
    template="When giving a value for {arg1}, the user should start giving the wrong answer, before correcting themselves in the same sentence. When giving this early confirmation, include [TURN_CORRECTION] at the beginning of the text.",
    args_required=1,
    disambig_required=False,
    cancel_request=False,
    tag="TURN_CORRECTION",
    example="""
# Conversation rules:
1) When giving a value for recipients, the user should start giving the wrong answer, before correcting themselves in the same sentence. When giving this early confirmation, include [TURN_CORRECTION] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="how are you")

# Generated conversation:
user: Send message
lucid: who do you want to send the message to
user: [TURN_CORRECTION] to Dad, no to Tom
lucid: What do you want to say in the message?
user: make the content hey!!!
lucid: Message content is "hey!!!", do you want to send the message?
user: change the content to how are you in the message
lucid: Changed message content to "how are you". Are you ready to send?
user: yep
lucid: Message sent! Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: set new timer for 5 minutes
2 set_timer(duration="5 minutes")
3 Hint("Ask to confirm", ref=x2)
4 say(x3)
lucid: Would you like to confirm setting a timer for 5 minutes?
user: change it to 6 minutes
5 x2.duration = "6 minutes"
6 Hint("Ask to confirm", ref=x2)
7 say(x6)
lucid: Timer duration updated, do you want to confirm setting a timer for 6 minutes?
user: Yes please
8 confirm(x2)
9 perform(x2)
10 say(x9)
lucid: Timer set
""",
)

WRONG_SLOT = ConversationRule(
    label="wrong_slot",
    template="When Lucid asks about {arg1} the first time, the user should instead respond about {arg2} and not mention {arg1}. When doing this, include [WRONG_SLOT] at the beginning of the text.",
    args_required=2,
    disambig_required=False,
    cancel_request=False,
    tag="WRONG_SLOT",
    example="""
# Conversation rules:
1) When Lucid asks about recipients the first time, the user should instead respond about content and not mention recipients. When doing this, include [WRONG_SLOT] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="hey!!!")

# Generated conversation:
user: Send message for me please
lucid: Who do you want to send it to?
user: [WRONG_SLOT] Put hey!!! in the message
lucid: Sure, and who do you want to send to?
user: Send it to Tom
lucid: Message to Tom says "hey!!!". Do you want to send it?
user: yep
lucid: Your message has been sent! Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: set a new alarm
2 create_alarm()
3 Hint("ask for value: time", ref=x2)
4 say(x3)
lucid: When do you want the alarm set for?
user: Call it run alarm
5 x2.label = "run alarm"
6 Hint("ask for value: time", ref=x2)
7 say(x6)
lucid: What time do you want to set the alarm for?
user: 7:00 AM
8 x2.time = "7:00 AM"
9 Hint("Ask to confirm", ref=x2)
10 say(x9)
lucid: Are you ready to confirm you want to set the alarm?
user: Yes please
11 confirm(x2)
12 perform(x2)
13 say(x12)
lucid: Alarm set
""",
)

EARLY_END = ConversationRule(
    label="early_end",
    template="When Lucid asks about {arg1} the first time, the user should be cut off too early (giving an incoherent reply). The cut off must not be during a word. When giving the incoherent answer include [EARLY_END] at the beginning of the text.",
    args_required=1,
    disambig_required=False,
    cancel_request=False,
    tag="EARLY_END",
    example="""
# Conversation rules:
1) When Lucid asks about {arg1} the first time, the user should be cut off too early (giving an incoherent reply). The cut off must not be during a word. When giving the incoherent answer include [EARLY_END] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="hey!!!")

# Generated conversation:
user: Send message for me please
lucid: Who do you want to send it to?
user: [EARLY_END] send the email to
lucid: Sorry I missed that, can you say that again?
user: Send the email to Tom!
lucid: Message to Tom says "hey!!!". Do you want to send it?
user: yep
lucid: Your message has been sent! Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: set a new alarm
2 create_alarm()
3 Hint("ask for value: time", ref=x2)
4 say(x3)
lucid: When do you want the alarm set for?
user: Set it for
5 say()
lucid: Sorry I couldn't understand that. What time do you want to set the alarm for?
user: 7:00 AM
6 x2.time = "7:00 AM"
7 Hint("Ask to confirm", ref=x2)
8 say(x7)
lucid: Ready to confirm setting the alarm?
user: Yes please
9 confirm(x2)
10 perform(x2)
11 say(x10)
lucid: Alarm set
""",
)

IRRELEVANT_ANSWER = ConversationRule(
    label="irrelevant_answer",
    template="When Lucid asks about {arg1} the first time, give a statement that is an irrelevant, impossible response. When giving the irrelevant answer include [IRRELEVANT] at the beginning of the text.",
    args_required=1,
    disambig_required=False,
    cancel_request=False,
    tag="IRRELEVANT",
    example="""
# Conversation rules:
1) When Lucid asks about recipients for the first time, give an irrelevant response. This must not be a valid, possible answer to recipients. When giving the irrelevant answer include [IRRELEVANT] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="hey!!!")

# Generated conversation:
user: I need to send a message
lucid: Who would you like to send the message to?
user: [IRRELEVANT] Oh no, battery running out again
lucid: That's not a valid recipient - who would you like to send the message to?
user: Please send the message to Tom
lucid: Added Tom as a recipient. What would you like to say in the message?
user: Just say hey!!!
lucid: Message content is "hey!!!". Ready to send it?
user: yep
lucid: Your message to Tom has been sent. Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: set a new alarm
2 create_alarm()
3 Hint("ask for value: time", ref=x2)
4 say(x3)
lucid: When do you want the alarm set for?
user: Hamish, get down from there
5 say()
lucid: What time do you want to set the alarm for?
user: 7:00 AM
6 x2.time = "7:00 AM"
7 Hint("Ask to confirm", ref=x2)
8 say(x7)
lucid: Ready to confirm setting the alarm?
user: Yes please
9 confirm(x2)
10 perform(x2)
11 say(x10)
lucid: Alarm set
""",
)

OVERHEARD_ANSWER = ConversationRule(
    label="overheard_answer",
    template="When Lucid asks about {arg1}, don't answer, but instead say something to someone who is in the same room (e.g. a pet or a child). When giving this overheard answer include [OVERHEARD] at the beginning of the text.",
    args_required=1,
    disambig_required=False,
    cancel_request=False,
    tag="OVERHEARD",
    example="""
# Conversation rules:
1) When Lucid asks about recipients, don't answer, but instead say something to someone who is in the same room (e.g. a pet or a child). When giving this overheard answer include [OVERHEARD] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="hey!!!")

# Generated conversation:
user: I need to send a message
lucid: Who would you like to send the message to?
user: [OVERHEARD] Not again Hamish! Off the counter now!
lucid: That's not a valid recipient - who would you like to send the message to?
user: Sorry, my cat was misbehaving. Please send the message to Tom
lucid: Added Tom as a recipient. What would you like to say in the message?
user: Just say hey!!!
lucid: Message content is "hey!!!". Ready to send it?
user: yep
lucid: Your message to Tom has been sent. Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: set a new alarm
2 create_alarm()
3 Hint("ask for value: time", ref=x2)
4 say(x3)
lucid: When do you want the alarm set for?
user: Hamish, get down from there
5 say()
lucid: What time do you want to set the alarm for?
user: 7:00 AM
6 x2.time = "7:00 AM"
7 Hint("Ask to confirm", ref=x2)
8 say(x7)
lucid: Ready to confirm setting the alarm?
user: Yes please
9 confirm(x2)
10 perform(x2)
11 say(x10)
lucid: Alarm set
""",
)

SARCASTIC_ANSWER = ConversationRule(
    label="sarcastic_answer",
    template="When Lucid asks about {arg1} the first time, say something sarcastic and totally impossible to Lucid. When giving this sarcastic answer include [SARCASTIC] at the beginning of the text.",
    args_required=1,
    disambig_required=False,
    cancel_request=False,
    tag="SARCASTIC",
    example="""
# Conversation rules:
1) When Lucid asks about recipients the first time, say something sarcastic and totally impossible to Lucid. When giving this sarcastic answer include [SARCASTIC] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="hey!!!")

# Generated conversation:
user: I need to send a message
lucid: Who would you like to send the message to?
user: [SARCASTIC] Abraham Lincoln of course, and make sure to CC Winston Churchill and Nelson Mandela!
lucid: That's not a valid recipient - who would you like to send the message to?
user: Obviously, send the message to Tom
lucid: Added Tom as a recipient. What would you like to say in the message?
user: Just say hey!!!
lucid: Message content is "hey!!!". Ready to send it?
user: yep
lucid: Your message to Tom has been sent. Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: set a new alarm
2 create_alarm()
3 Hint("ask for value: time", ref=x2)
4 say(x3)
lucid: When do you want the alarm set for?
user: How about 2 minutes past never o'clock?
5 say()
lucid: Very funny. What time do you want to set the alarm for?
user: 7:00 AM
6 x2.time = "7:00 AM"
7 Hint("Ask to confirm", ref=x2)
8 say(x7)
lucid: Ready to confirm setting the alarm?
user: Yes please
9 confirm(x2)
10 perform(x2)
11 say(x10)
lucid: Alarm set
""",
)

START_WITH_SLOT_VALUES = ConversationRule(
    label="start_multi_slot",
    template="The first message from the user should provide values for both {arg1} and {arg2} together. When giving this answer, include [START_MULTI] at the beginning of the text.",
    args_required=2,
    disambig_required=False,
    cancel_request=False,
    tag="START_MULTI",
    example="""
# Conversation rules:
1) The first message from the user should provide values for both recipients and content together. When giving this answer, include [START_MULTI] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="im going to be late")

# Generated conversation:
user: [START_MULTI] I need to send a message saying im going to be late to Tom
lucid: Your message to Tom says "im going to be late". Do you want to send the message?
user: yep
lucid: message sent! Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: set a new alarm for 8:30, and label it swimming get-up
2 create_alarm(time="8:30", label="swimming get-up")
3 Hint("Ask to confirm", ref=x2)
4 say(x3)
lucid: Ready to confirm setting the alarm?
user: Yes please
5 confirm(x2)
6 perform(x2)
7 say(x6)
lucid: Alarm set
""",
)

ALL_IN_ONE_TURN = ConversationRule(
    label="all_in_one",
    template="The user must provide all information in a single turn. The user doesn't need to mention about confirmation. When giving this answer, include [ALL_IN_ONE] at the beginning of the text.",
    args_required=0,
    disambig_required=False,
    cancel_request=False,
    tag="ALL_IN_ONE",
    example="""
# Conversation rules:
1) The user must provide all information in a single turn. The user doesn't need to mention about confirmation. When giving this answer, include [ALL_IN_ONE] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="hey!!!")

# Generated conversation:
user: I need to send a message to Tom saying hey!!!
lucid: Sending a message to Tom saying "hey!!!". Can I help with anything else?
user: End conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: [ALL_IN_ONE] set a new alarm for 8:30, and label it swimming get-up
2 create_alarm(time="8:30", label="swimming get-up")
3 Hint("Ask to confirm", ref=x2)
4 say(x3)
lucid: Ready to confirm setting the alarm?
user: Yes please
5 confirm(x2)
6 perform(x2)
7 say(x6)
lucid: Alarm set
""",
)

CANCEL_TASK = ConversationRule(
    label="cancel",
    template="If asked for confirmation, the user decides not to confirm the task. When giving this answer, include [CANCEL] at the beginning of the text.",
    args_required=0,
    disambig_required=False,
    cancel_request=True,
    tag="CANCEL",
    example="""
# Conversation rules:
1) If asked for confirmation, the user decides not to confirm the task. When giving this answer, include [CANCEL] at the beginning of the text.
2) The task the user wants to perform must be: send_message(recipients="Tom", content="hey!!!")

# Generated conversation:
user: I need to send a message
lucid: Who would you like to send the message to?
user: Please send the message to Tom
lucid: Added Tom as a recipient. What would you like to say in the message?
user: Just say hey!!!
lucid: Message content is "hey!!!", do you want to send it?
user: [CANCEL] actually no, don't send it
lucid: message not sent. Can I help with anything else?
user: end conversation
""",
    ssa_example="""
user: hey lucid
1 say()
lucid: Hi! How can I help?
user: set a new alarm for 8:30, and label it swimming get-up
2 create_alarm(time="8:30", label="swimming get-up")
3 Hint("Ask to confirm", ref=x2)
4 say(x3)
lucid: Ready to confirm setting the alarm?
user: what's the point
6 say()
lucid: Alarm has not been set
""",
)

conversation_rules = [
    CORRECTION,
    WRONG_SLOT,
    IRRELEVANT_ANSWER,
    START_WITH_SLOT_VALUES,
    CANCEL_TASK,
    ALL_IN_ONE_TURN,
    QUERY,
    OVERHEARD_ANSWER,
    SARCASTIC_ANSWER,
    TURN_CORRECTION,
    EARLY_END,
    DELAY_CONFIRMATION,
]

default_example = [
    """
# Conversation rules:
1) The task the user wants to perform must be: send_flowers(recipients="Mum", content="happy mothers day")

# Generated conversation:
user: I want to send some flowers
lucid: Who do you want to send them to?
user: My mum!
lucid: What do you want to say in the message?
user: make the content happy mothers day
lucid: Are you ready to confirm sending the flowers?
user: yes please
lucid: flowers ordered! Is there anything else you would like help with?
user: end conversation
"""
]

default_ssa_examples = [
    """
user: I want to send a message to Bob
0 send_message(recipient="Bob")
1 Hint("Ask for value: content", ref=x0)
2 say(x1)
lucid: What would you like to say in your message to Bob?
user: say don't be late again! Make the subject tomorrow
3 x0.subject, x0.content = "tomorrow", "don't be late again!"
4 Hint("Ask to confirm", ref=x0)
5 say(x4)
lucid: Would you like to confirm sending the message to Bob?
user: yes
6 confirm(x0)
7 perform(x0)
8 say(x7)
lucid: Your message has been sent to Bob.
user: great. Lookup the last timer I set
9 find_timers(date_of_timers="last time")
10 perform(x9)
11 say(x10)
lucid: You last set a timer this morning for 2 hours time
user: thanks, now set a new timer
12 set_timer()
13 Hint("Ask for value: duration", ref=x12)
14 say(x13)
lucid: For how long should I set the timer?
user: for 10 minutes, and call it noodle timer
15 x12.label, x12.duration = "noodle timer", "10 minutes"
16 Hint("Ask to confirm", ref=x12)
17 say(x16)
lucid: Are you ready to confirm setting the timer?
user: Wait, I need to specify the timer alarm type
18 say()
lucid: Sure. What type of alarm do you want for the timer?
user: Make the timer alarm classical music
19 x12.alarm_type = "classical music"
20 Hint("Ask to confirm", ref=x12)
21 say(x20)
lucid: The timer alarm is set as classical music. Are you ready to confirm setting the timer?
user: Yes please!
22 confirm(x12)
23 perform(x12)
24 say(x23)
lucid: Your "noodle timer" for 10 minutes is now running. The timer has a classical music alarm. Is there anything else?
user: That's all. Thanks for your help!
25 say()
lucid: You're welcome.
user: end conversation
"""
]
