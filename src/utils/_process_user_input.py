import pathlib
import unittest

__all__ = ["process_user_input"]

def process_user_input(user_input_path) -> list:
    """
    Reads all lines from a text file and returns them as a list.

    Args:
        user_input_path (str or pathlib.Path): Path to the text file to be read.

    Returns:
        list: A list where each element is a line from the text file.
              If the text file is empty, this returns an empty list.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is an issue reading the file.
    """
    try:
        # open the file and read lines
        with open(user_input_path, 'r') as file:
            lines = file.readlines()
        
        # strip trailing newline characters from each line
        return [line.strip() for line in lines]
    except Exception as e:
        raise IOError(f"Error processing the file at {user_input_path}: {e}")


class TestProcessUserInput(unittest.TestCase):
    """
    Unit test class for the process_user_input function.

    Validates the functionality of reading a text file and returning its contents as a list.
    """

    def test_process_user_queries(self):
        """
        Tests the process_user_input function with a mock text file.

        Verifies that the function correctly reads lines and returns them as a list.
        """
        # construct the path to the mock text file
        file_path = pathlib.Path.cwd() / "src" / "tests" / "mock_user_queries.txt"

        # call the function to process the file
        res = process_user_input(user_input_path=file_path)

        # expected result
        exp_res = [
            "first query",
            "second query",
            "third query",
        ]

        print(f"result= {res}")

        # assert that the processed result matches the expected result
        self.assertEqual(res, exp_res)
