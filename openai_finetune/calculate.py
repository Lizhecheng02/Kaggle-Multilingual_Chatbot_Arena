import tiktoken
import json
from tqdm import tqdm


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_message = 4
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}.""")
    num_tokens = 0
    for message in tqdm(messages, total=len(messages)):
        num_tokens += tokens_per_message
        main_message = message["messages"]
        for idx in range(len(main_message)):
            for key, value in main_message[idx].items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


messages = []
with open("train_data.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        message = json.loads(line)
        messages.append(message)

print("The total number of tokens for train jsonl is:", num_tokens_from_messages(messages))

messages = []
with open("validation_data.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        message = json.loads(line)
        messages.append(message)

print("The total number of tokens for validation jsonl is:", num_tokens_from_messages(messages))
