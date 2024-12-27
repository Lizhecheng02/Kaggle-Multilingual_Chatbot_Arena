from utils import get_response_openai_prompt, sample_validation_dataset, construct_few_shot_prompt
from tqdm import tqdm
import re


def few_shot_direct_output(model, msicl_nums, file_path, sample_size):
    sample_validation_df, remain_df = sample_validation_dataset(file_path, sample_size)
    for msicl_num in msicl_nums:
        final_sample_size = sample_size
        acc_count = 0
        pre_prompt = construct_few_shot_prompt(df=remain_df, msicl_num=msicl_num) + "Now you need to evaluate the following question and responses based on the examples provided. Determine which response is better from a human perspective and return either 'model_a' or 'model_b' only. DO NOT include explanations.\n"
        print(f"Processing MSICL With {msicl_num} Examples")
        for idx, row in tqdm(sample_validation_df.iterrows(), total=len(sample_validation_df)):
            prompt = pre_prompt + f"<Question>\n{row['prompt']}\n</Question>\n"
            prompt += f"<Response A>\n{row['response_a']}\n</Response A>\n"
            prompt += f"<Response B>\n{row['response_b']}\n</Response B>\n"
            prompt += "[YOUR JUDEMENT]:"
            response = get_response_openai_prompt(model, prompt)
            if response == "model_a" and row["winner"] == "model_a":
                acc_count += 1
            elif response == "model_b" and row["winner"] == "model_b":
                acc_count += 1
            elif response != "model_a" and response != "model_b":
                print(f"Error: {response} Row: {idx + 1}")
                final_sample_size -= 1
        print(f"MSICL With {msicl_num} Examples Accuracy: {acc_count / final_sample_size * 100: .2f}%")


def zero_shot_direct_output(model, file_path, sample_size):
    sample_validation_df, _ = sample_validation_dataset(file_path, sample_size)
    final_sample_size = sample_size
    acc_count = 0
    pre_prompt = "You are a multilingual expert tasked with evaluating and comparing answers from two large language models from a human's perspective to determine which one provides the better response. Below is the case that you need to evaluate.\n"
    for idx, row in tqdm(sample_validation_df.iterrows(), total=len(sample_validation_df)):
        prompt = pre_prompt + f"<Question>\n{row['prompt']}\n</Question>\n"
        prompt += f"<Response A>\n{row['response_a']}\n</Response A>\n"
        prompt += f"<Response B>\n{row['response_b']}\n</Response B>\n"
        prompt += "If you think Response A is better, return 'model_a'; Otherwise, return 'model_b'. You should only return either 'model_a' or 'model_b' without any extra explanations."
        response = get_response_openai_prompt(model, prompt)
        if response == "model_a" and row["winner"] == "model_a":
            acc_count += 1
        elif response == "model_b" and row["winner"] == "model_b":
            acc_count += 1
        elif response != "model_a" and response != "model_b":
            print(f"Error: {response} Row: {idx + 1}")
            final_sample_size -= 1
    print(f"Zero Shot Direct Output Accuracy: {acc_count / final_sample_size * 100: .2f}%")


def few_shot_reasoning_output(model, msicl_nums, file_path, sample_size):
    sample_validation_df, remain_df = sample_validation_dataset(file_path, sample_size)
    for msicl_num in msicl_nums:
        final_sample_size = sample_size
        acc_count = 0
        pre_prompt = construct_few_shot_prompt(df=remain_df, msicl_num=msicl_num) + "Now you need to evaluate the following question and responses based on the examples provided.\n"
        print(f"Processing MSICL With {msicl_num} Examples")
        for idx, row in tqdm(sample_validation_df.iterrows(), total=len(sample_validation_df)):
            prompt = pre_prompt + f"<Question>\n{row['prompt']}\n</Question>\n"
            prompt += f"<Response A>\n{row['response_a']}\n</Response A>\n"
            prompt += f"<Response B>\n{row['response_b']}\n</Response B>\n"
            prompt += "You should think carefully according to the example(s) shown above. Please provide a detailed explanation (at most 400 tokens) within <analysis> tags and give your final choice either 'model_a' or 'model_b' within <answer> tags."
            response = get_response_openai_prompt(model, prompt, max_tokens=512)
            try:
                response = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL).group(1).strip()
                if response == "model_a" and row["winner"] == "model_a":
                    acc_count += 1
                elif response == "model_b" and row["winner"] == "model_b":
                    acc_count += 1
                elif response != "model_a" and response != "model_b":
                    print(f"Error: {response} Row: {idx + 1}")
                    final_sample_size -= 1
            except:
                print(f"Error: {response} Row: {idx + 1}")
                final_sample_size -= 1
        print(f"MSICL With {msicl_num} Examples Accuracy: {acc_count / final_sample_size * 100: .2f}%")


def zero_shot_reasoning_output(model, file_path, sample_size):
    sample_validation_df, _ = sample_validation_dataset(file_path, sample_size)
    final_sample_size = sample_size
    acc_count = 0
    pre_prompt = "You are a multilingual expert tasked with evaluating and comparing answers from two large language models from a human's perspective to determine which one provides the better response. Below is the case that you need to evaluate.\n"
    for idx, row in tqdm(sample_validation_df.iterrows(), total=len(sample_validation_df)):
        prompt = pre_prompt + f"<Question>\n{row['prompt']}\n</Question>\n"
        prompt += f"<Response A>\n{row['response_a']}\n</Response A>\n"
        prompt += f"<Response B>\n{row['response_b']}\n</Response B>\n"
        prompt += "You should think carefully according to the example(s) shown above. Please provide a detailed explanation (at most 400 tokens) within <analysis> tags and give your final choice either 'model_a' or 'model_b' within <answer> tags."
        response = get_response_openai_prompt(model, prompt)
        try:
            response = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL).group(1).strip()
            if response == "model_a" and row["winner"] == "model_a":
                acc_count += 1
            elif response == "model_b" and row["winner"] == "model_b":
                acc_count += 1
            elif response != "model_a" and response != "model_b":
                print(f"Error: {response} Row: {idx + 1}")
                final_sample_size -= 1
        except:
            print(f"Error: {response} Row: {idx + 1}")
            final_sample_size -= 1
    print(f"Zero Shot Reasoning Output Accuracy: {acc_count / final_sample_size * 100: .2f}%")


if __name__ == "__main__":
    few_shot_direct_output(model="gpt-4o-mini", msicl_nums=[1, 2, 3, 4], file_path="../original_data/train.parquet", sample_size=500)
    few_shot_reasoning_output(model="gpt-4o-mini", msicl_nums=[1, 2, 3, 4], file_path="../original_data/train.parquet", sample_size=500)
    zero_shot_direct_output(model="gpt-4o-mini", file_path="../original_data/train.parquet", sample_size=500)
    zero_shot_reasoning_output(model="gpt-4o-mini", file_path="../original_data/train.parquet", sample_size=500)
