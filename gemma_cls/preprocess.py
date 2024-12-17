from pandas import pd
from sklearn.model_selection import train_test_split
from datasets import Dataset


def get_train_and_test(original_train_file_path_list, extra_train_file_path_list):
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

    train_df, test_df = train_test_split(original_train_df, test_size=0.1, random_state=7)
    train_df = pd.concat([train_df, extra_train_file_df])
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    print(f"Train Data Size: {len(train_df)}")
    print(f"Test Data Size: {len(test_df)}")


def get_cls_prompt(train_df, test_df):
    train_df["input"] = "<PROMPT>\n" + train_df["prompt"] + "\n</PROMPT>\n<RESPONSE A>\n" + train_df["response_a"] + "\n</RESPONSE A>\n<RESPONSE B>\n" + train_df["response_b"] + "\n</RESPONSE B>"
    test_df["input"] = "<PROMPT>\n" + test_df["prompt"] + "\n</PROMPT>\n<RESPONSE A>\n" + test_df["response_a"] + "\n</RESPONSE A>\n<RESPONSE B>\n" + test_df["response_b"] + "\n</RESPONSE B>"
    return train_df, test_df


def form_final_dataset(train_df, test_df):
    train_df["labels"] = train_df["winner"].apply(lambda x: 0 if x == "model_a" else 1)
    test_df["labels"] = test_df["winner"].apply(lambda x: 0 if x == "model_a" else 1)
    train_df.drop(columns=["id", "prompt", "response_a", "response_b", "winner", "model_a", "model_b", "language"], inplace=True)
    test_df.drop(columns=["id", "prompt", "response_a", "response_b", "winner", "model_a", "model_b", "language"], inplace=True)
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)
    return train_ds, test_ds
