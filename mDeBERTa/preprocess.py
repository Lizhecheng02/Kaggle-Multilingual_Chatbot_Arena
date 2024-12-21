import pandas as pd
from sklearn.model_selection import train_test_split


def load_train_and_val(train_file_path_list, val_file_path_list, train_select_num=None, val_select_num=None):
    train_df = pd.DataFrame()
    for train_file_path in train_file_path_list:
        if train_file_path.endswith(".csv"):
            train_file_df = pd.read_csv(train_file_path)
            train_df = pd.concat([train_df, train_file_df])
        elif train_file_path.endswith(".parquet"):
            train_file_df = pd.read_parquet(train_file_path)
            train_df = pd.concat([train_df, train_file_df])
        elif train_file_path.endswith(".json"):
            train_file_df = pd.read_json(train_file_path)
            train_df = pd.concat([train_df, train_file_df])
        else:
            raise ValueError("Unsupported File Format")

    val_df = pd.DataFrame()
    for val_file_path in val_file_path_list:
        if val_file_path.endswith(".csv"):
            val_file_df = pd.read_csv(val_file_path)
            val_df = pd.concat([val_df, val_file_df])
        elif val_file_path.endswith(".parquet"):
            val_file_df = pd.read_parquet(val_file_path)
            val_df = pd.concat([val_df, val_file_df])
        elif val_file_path.endswith(".json"):
            val_file_df = pd.read_json(val_file_path)
            val_df = pd.concat([val_df, val_file_df])
        else:
            raise ValueError("Unsupported File Format")

    if train_select_num is not None:
        train_df = train_df.sample(n=train_select_num, random_state=7)
    if val_select_num is not None:
        val_df = val_df.sample(n=val_select_num, random_state=7)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    print(f"Train Data Size: {len(train_df)}")
    print(f"Val Data Size: {len(val_df)}")
    return train_df, val_df


def get_train_and_val(original_train_file_path_list, extra_train_file_path_list, train_select_num=None, val_select_num=None):
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

    if extra_train_file_path_list is not None:
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

    if extra_train_file_path_list is not None:
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
