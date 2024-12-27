import os
import pandas as pd
import yaml
import warnings
from openai import OpenAI
warnings.filterwarnings("ignore")

config_path = os.path.join(os.getcwd(), "..", "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

openai_api_key = config["openai"]["api_key"]
openai_organization = config["openai"]["organization"]
client_openai = OpenAI(api_key=openai_api_key, organization=openai_organization)


def get_response_openai_prompt(model, prompt, temperature=0.0, top_p=0.95, max_tokens=8):
    if "gpt" in model:
        completion = client_openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_tokens
        )
        response = completion.choices[0].message.content.strip()
        return response
    else:
        completion = client_openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
        response = completion.choices[0].message.content.strip()
        return response


def sample_validation_dataset(file_path, sample_size):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Invalid file format. Please provide a valid file format.")

    sample_validation_df = df.sample(n=sample_size, random_state=3407)
    sample_validation_df.reset_index(drop=True, inplace=True)
    remain_df = df[~df["id"].isin(sample_validation_df["id"])].sample(frac=1, random_state=3407)
    remain_df.reset_index(drop=True, inplace=True)

    print(f"Sampled {sample_size} rows for validation dataset.")
    print(f"Remaining {len(df) - sample_size} rows for training dataset.")

    return sample_validation_df, remain_df


def construct_few_shot_prompt(df, msicl_num):
    prompt = f"You are a multilingual expert tasked with evaluating and comparing answers from two large language models from a human's perspective to determine which one provides the better response. You will be given examples judged by humans, and you should learn from these examples to understand human preferences and finally make your own judgment. Here are {msicl_num} examples for you to learn from:\n"
    for i in range(msicl_num):
        if i != msicl_num - 1:
            prompt += f"<Example {i+1}>\n"
            prompt += f"<Question>\n{df['prompt'][i]}\n</Question>\n"
            prompt += f"<Response A>\n{df['response_a'][i]}\n</Response A>\n"
            prompt += f"<Response B>\n{df['response_b'][i]}\n</Response B>\n"
            prompt += f"<Judgment>\n{df['winner'][i]}\n</Judgment>\n"
            prompt += f"</Example {i+1}>\n\n"
        else:
            prompt += f"<Example {i+1}>\n"
            prompt += f"<Question>\n{df['prompt'][i]}\n</Question>\n"
            prompt += f"<Response A>\n{df['response_a'][i]}\n</Response A>\n"
            prompt += f"<Response B>\n{df['response_b'][i]}\n</Response B>\n"
            prompt += f"<Judgment>\n{df['winner'][i]}\n</Judgment>\n"
            prompt += f"</Example {i+1}>\n"
    return prompt
