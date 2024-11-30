import pathlib

from ..utils import process_user_input


# global params 
user_input_path = pathlib.Path.cwd() / "src" / "playbooks" / "user_queries.txt"

if __name__=="__main__":

    # get the list of queries 
    user_input = process_user_input(
        user_input_path= user_input_path,
    )

    # process one query at a time from the user input 
    for q in user_input:
        ...