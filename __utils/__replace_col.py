import pandas as pd


def replace_cols(src_df_path: str, dest_df_path: str, new_df_path: str, cols: list):
    src_df = pd.read_csv(src_df_path)
    dest_df = pd.read_csv(dest_df_path)
    for col in cols:
        dest_df[col] = src_df[col]
    dest_df.to_csv(new_df_path, index=False)


replace_cols(
    src_df_path="./temp/source/rnn_final_result.csv",
    dest_df_path="./temp/source/l1.csv",
    new_df_path="./temp/source/l1_new.csv",
    cols=["RNN"],
)
