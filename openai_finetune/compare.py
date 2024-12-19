import os
import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

config_path = os.path.join(os.getcwd(), "..", "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

openai_api_key = config["openai"]["api_key"]
openai_organization = config["openai"]["organization"]
client_openai = OpenAI(api_key=openai_api_key, organization=openai_organization)

system = "You are tasked with standing from the human's viewpoint to evaluate and compare answers from two large language models to determine which one provides the better response. You will be given a question and two responses, one from each model."

prompt = """Here is the question that was asked:
<QUESTION>
{{QUESTION}}
</QUESTION>
Here is the response provided by Model A:
<RESPONSE A>
{{MODELA_ANSWER}}
</RESPONSE A>
Here is the response provided by Model B:
<RESPONSE B>
{{MODELB_ANSWER}}
</RESPONSE B>
If you believe Model A's response is better, respond with the letter 'A'. If you believe Model B's response is better, respond with the letter 'B'. Provide only the letter, with no additional explanation or formatting.
"""


def get_response_openai(model_name, system, prompt):
    completion = client_openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()


def compare(file_path, original_model_id, finetuned_model_id):
    original_correct = 0
    finetuned_correct = 0

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        raise ValueError("File Format Error")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        input_prompt = prompt.replace("{{QUESTION}}", row["prompt"]).replace("{{MODELA_ANSWER}}", row["response_a"]).replace("{{MODELB_ANSWER}}", row["response_b"])
        original_response = get_response_openai(original_model_id, system, input_prompt)
        finetuned_response = get_response_openai(finetuned_model_id, system, input_prompt)
        if row["winner"] == "model_a" and original_response == "A":
            original_correct += 1
        elif row["winner"] == "model_b" and original_response == "B":
            original_correct += 1

        if row["winner"] == "model_a" and finetuned_response == "A":
            finetuned_correct += 1
        elif row["winner"] == "model_b" and finetuned_response == "B":
            finetuned_correct += 1

    print(f"Original Model Accuracy: {original_correct / len(df)}")
    print(f"Finetuned Model Accuracy: {finetuned_correct / len(df)}")


if __name__ == "__main__":
    compare("./val_500.csv", original_model_id="gpt-4o-mini-2024-07-18", finetuned_model_id="ft:gpt-4o-mini-2024-07-18:personal:lmsys2:Ag0DMRS0")
