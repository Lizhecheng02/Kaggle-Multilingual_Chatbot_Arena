from utils import get_response_openai_prompt, sample_validation_dataset, construct_prompt
from tqdm import tqdm


def main(model, msicl_nums, file_path, sample_size):
    sample_validation_df, remain_df = sample_validation_dataset(file_path, sample_size)
    for msicl_num in msicl_nums:
        final_sample_size = sample_size
        acc_count = 0
        pre_prompt = construct_prompt(df=remain_df, msicl_num=msicl_num)
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


if __name__ == "__main__":
    main(model="gpt-4o-mini", msicl_nums=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], file_path="../original_data/train.parquet", sample_size=600)
