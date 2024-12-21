import pandas as pd
from sklearn.model_selection import train_test_split


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
    return train_df, test_df
