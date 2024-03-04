# LUCID
This software project accompanies the research paper [LUCID: LLM-Generated Utterances for Complex and Interesting Dialogues](https://arxiv.org/abs/2403.00462).

LUCID is a highly automated, LLM-driven data generation system for task-oriented dialogues. LUCID aims to produce realistic, diverse and challenging conversations, with highly accurate labels. LUCID takes a modularised approach to data generation, compartmentalising the data generation task into manageable steps that an LLM can consistently perform accurately. For more details, please see our paper.

This repo contains the code for the data generation system (which can be used to generate more data), the data we have already generated for our paper (LUCIDv1.0), and the code for our baseline models.

## Documentation

## Getting Started 

# Step 1: Generating intents

To create new intents from a description:

- Open _**lucid_generate_data/run_scripts/create_intents_from_description.py**_
- In this file, update INTENTS, a dictionary containing domains, and the desired intent descriptions within each domain
- Once finished, run the .py file from the root directory (_** python lucid_generate_data/run_scripts/create_intents_from_description.py**)
- The new intents will be generated in **lucid_generate_data/intents_for_data_generation**

# Step 2: Generating conversations

a Open _**lucid_generate_data/run_scripts/run_conversations.py**_
- Inside this file, decide now many conversations to generate per intent (CONVS\_PER\_INTENT), the maximum number of intents for a conversation (MAX\_INTENTS\_IN\_CONVERSATION)
- You also need to specify the conversational phenomena that you would like for the conversation (UNHAPPY\_PATHS). Note that for the data generated for the paper, these were randomly sampled for each conversation (with either 0, 1 or 2 unhappy paths per conversation.
- Your saved conversations will be stored in _**lucid_generate_data/saved_conversations**_


# Step 3: Data formatting and post-processing

- To assemble your generated conversations into your final dataset, run _**lucid_generate_data/compile_data.py**_
- Your final dataset will be called LUCID_data.json

# Step 4: Running our baseline model

To run the LUCID baselines, please use: _**python running_baseline/run_llm.py**_
