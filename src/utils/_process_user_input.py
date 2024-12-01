import unittest
import pathlib

__all__ = ["process_user_input"]

def process_user_input(user_input_path) -> list:
    """
    Reads all lines from a text file and returns them as a list.

    Args:
        text_file_path (str): Path to the text file to be read.

    Returns:
        list: A list where each element is a line from the text file.
              If the text file is empty, this returns an empty list. 
    """
    with open(user_input_path, 'r') as file:
        lines = file.readlines()
    
    # remove any trailing newline characters
    return [line.strip() for line in lines]

class TestProcessUserInput(unittest.TestCase):
    def test_process_user_queries(self):
        file_path = pathlib.Path.cwd() / "src" / "tests" / "mock_user_queries.txt"
        
        res = process_user_input(user_input_path= file_path)

        exp_res = [
            "first query",
            "second query",
            "third query",
        ]

        print(f"result= {res}")

        self.assertEqual(res, exp_res)