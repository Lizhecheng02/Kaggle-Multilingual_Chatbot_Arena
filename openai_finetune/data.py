import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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


def get_train_and_val(original_train_file_path_list, extra_train_file_path_list, train_select_num=1000, val_select_num=100):
    original_train_df = pd.DataFrame()
    for original_train_file_path in original_train_file_path_list:
        if original_train_file_path.endswith(".csv"):
            original_train_file_df = pd.read_csv(original_train_file_path)
            original_train_df = pd.concat([original_train_df, original_train_file_df])
        elif original_train_file_path.endswith(".parquet"):
            original_train_file_df = pd.read_parquet(original_train_file_path)
            original_train_df = pd.concat([original_train_df, original_train_file_df])
        elif original_train_file_path.endswith(".json"):
            original_train_file_df = pd.read_json(original_train_file_path)
            original_train_df = pd.concat([original_train_df, original_train_file_df])
        else:
            raise ValueError("Unsupported File Format")

    extra_train_df = pd.DataFrame()
    for extra_train_file_path in extra_train_file_path_list:
        if extra_train_file_path.endswith(".csv"):
            extra_train_file_df = pd.read_csv(extra_train_file_path)
            extra_train_df = pd.concat([extra_train_df, extra_train_file_df])
        elif extra_train_file_path.endswith(".parquet"):
            extra_train_file_df = pd.read_parquet(extra_train_file_path)
            extra_train_df = pd.concat([extra_train_df, extra_train_file_df])
        elif extra_train_file_path.endswith(".json"):
            extra_train_file_df = pd.read_json(extra_train_file_path)
            extra_train_df = pd.concat([extra_train_df, extra_train_file_df])
        else:
            raise ValueError("Unsupported File Format")

    train_df, val_df = train_test_split(original_train_df, test_size=0.1, random_state=7)
    train_df = pd.concat([train_df, extra_train_file_df])

    if train_select_num is not None:
        train_df = train_df.sample(n=train_select_num, random_state=7)
    if val_select_num is not None:
        val_df = val_df.sample(n=val_select_num, random_state=7)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    print(f"Train Data Size: {len(train_df)}")
    print(f"Val Data Size: {len(val_df)}")
    return train_df, val_df


def create_dataset(prompt, response):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }


def main(args):
    train_df, val_df = get_train_and_val(args.original_files, args.extra_files, train_select_num=args.train_select_num, val_select_num=args.val_select_num)

    with open("train_data.jsonl", "w") as f:
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
            input_prompt = prompt.replace("{{QUESTION}}", row["prompt"]).replace("{{MODELA_ANSWER}}", row["response_a"]).replace("{{MODELB_ANSWER}}", row["response_b"])
            output_content = "A" if row["winner"] == "model_a" else "B"
            data = json.dumps(create_dataset(prompt=input_prompt, response=output_content))
            f.write(data + "\n")

    with open("validation_data.jsonl", "w") as f:
        for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
            input_prompt = prompt.replace("{{QUESTION}}", row["prompt"]).replace("{{MODELA_ANSWER}}", row["response_a"]).replace("{{MODELB_ANSWER}}", row["response_b"])
            output_content = "A" if row["winner"] == "model_a" else "B"
            data = json.dumps(create_dataset(prompt=input_prompt, response=output_content))
            f.write(data + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Fine-tuning Data")
    parser.add_argument("--original_files", nargs="+", type=str)
    parser.add_argument("--extra_files", nargs="+", type=str)
    parser.add_argument("--train_select_num", type=int, default=1000)
    parser.add_argument("--val_select_num", type=int, default=100)
    main(args=parser.parse_args())
