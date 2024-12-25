import pandas as pd
import math
import time
from transformers import AutoTokenizer

train = pd.read_parquet("./train.parquet")
train = train.sample(10000, random_state=7)
train.reset_index(drop=True, inplace=True)

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-9b-it",
    trust_remote_code=True,
    token="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"
)


def count_tokens_gemma(text, tokenizer):
    return len(tokenizer.encode(text))


def truncate_from_tail(text, max_tokens, tokenizer):
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[-max_tokens:]
    return tokenizer.decode(truncated_tokens)


def truncate_by_ratio(response_a, response_b, max_total_tokens, tokenizer):
    a_tokens = count_tokens_gemma(response_a, tokenizer)
    b_tokens = count_tokens_gemma(response_b, tokenizer)
    total_tokens = a_tokens + b_tokens

    if total_tokens <= max_total_tokens:
        return response_a, response_b

    ratio_a = a_tokens / total_tokens if total_tokens != 0 else 0
    a_cut = math.floor(max_total_tokens * ratio_a)
    b_cut = max_total_tokens - a_cut

    truncated_a = truncate_from_tail(response_a, a_cut, tokenizer)
    truncated_b = truncate_from_tail(response_b, b_cut, tokenizer)
    return truncated_a, truncated_b


def process_row(prompt, response_a, response_b, total_threshold):
    prompt_tokens = count_tokens_gemma(prompt, tokenizer)
    response_a_tokens = count_tokens_gemma(response_a, tokenizer)
    response_b_tokens = count_tokens_gemma(response_b, tokenizer)

    total_tokens = prompt_tokens + response_a_tokens + response_b_tokens

    if total_tokens <= total_threshold:
        return prompt, response_a, response_b

    if prompt_tokens > total_threshold // 3:
        return prompt, response_a, response_b

    if (response_a_tokens + response_b_tokens) > 3 * prompt_tokens:
        max_response_tokens = total_threshold - prompt_tokens

        new_response_a, new_response_b = truncate_by_ratio(response_a, response_b, max_response_tokens, tokenizer)
        return prompt, new_response_a, new_response_b

    if (response_a_tokens + response_b_tokens) < total_threshold:
        max_prompt_tokens = total_threshold - response_a_tokens - response_b_tokens
        new_prompt = truncate_from_tail(prompt, max_prompt_tokens, tokenizer)
        return new_prompt, response_a, response_b

    else:
        new_prompt = truncate_from_tail(prompt, total_threshold // 6, tokenizer)
        new_response_a, new_response_b = truncate_by_ratio(response_a, response_b, total_threshold - total_threshold // 6, tokenizer)
        return new_prompt, new_response_a, new_response_b


def truncate_dataframe(df):
    if not {"prompt", "response_a", "response_b"}.issubset(df.columns):
        raise ValueError("DataFrame must have columns 'prompt', 'response_a', 'response_b'")

    new_data = df.apply(lambda row: process_row(row["prompt"], row["response_a"], row["response_b"], 1920), axis=1, result_type="expand")
    df[["prompt", "response_a", "response_b"]] = new_data

    return df


print("Start Processing ...")
start_time = time.time()
processed_train = truncate_dataframe(train)
print("--- %s seconds ---" % (time.time() - start_time))

processed_train["len_prompt"] = processed_train["prompt"].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
processed_train["len_response_a"] = processed_train["response_a"].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
processed_train["len_response_b"] = processed_train["response_b"].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))

print(processed_train["len_prompt"].describe())
print(processed_train["len_response_a"].describe())
print(processed_train["len_response_b"].describe())
