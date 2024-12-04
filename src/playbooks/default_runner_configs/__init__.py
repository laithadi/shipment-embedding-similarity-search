import pathlib

DATA_PATH = pathlib.Path.cwd() / "data" / "take_home_dataset.csv"
USER_QUERY_INPUT = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "user_queries.txt"
USER_DF_COLS_INPUT = pathlib.Path.cwd() / "src" / "playbooks" / "user_input" / "df_cols.txt"

# TODO: update output_json_filename is you need to
output_json_filename = "res_output.json"

OUTPUT_FILE_PATH = pathlib.Path.cwd() / "src" / "playbooks" / "outputs" / output_json_filename


__all__ = [
    "DATA_PATH",
    "USER_QUERY_INPUT",
    "USER_DF_COLS_INPUT",
    "OUTPUT_FILE_PATH",
]