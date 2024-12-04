import pathlib
import unittest
from typing import List, Union


__all__ = ["process_user_input"]

def process_user_input(user_input_path: Union[str, pathlib.Path]) -> List[str]:
    """
    Reads all lines from a text file and returns them as a list.

    Args:
        user_input_path (str or pathlib.Path): Path to the text file to be read.

    Returns:
        List[str]: A list where each element is a line from the text file.
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
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {user_input_path} does not exist.")
    except Exception as e:
        raise IOError(f"Error processing the file at {user_input_path}: {e}")


class TestProcessUserInput(unittest.TestCase):
    """
    Unit test class for the process_user_input function.

    validates the functionality of reading a text file and returning its contents as a list.
    """

    def test_process_user_queries(self):
        """
        tests the process_user_input function with a mock text file.

        verifies that the function correctly reads lines and returns them as a list.
        """
        # create a mock text file
        file_path = pathlib.Path(f"{pathlib.Path(__file__).parent}/tests/mock_user_queries.txt")
        content = "first query\nsecond query\nthird query\n"
        with open(file_path, "w") as mock_file:
            mock_file.write(content)

        try:
            # call the function to process the file
            res = process_user_input(user_input_path=file_path)

            # expected result
            exp_res = [
                "first query",
                "second query",
                "third query",
            ]

            # assert that the processed result matches the expected result
            self.assertEqual(res, exp_res)
        finally:
            # clean up the mock file
            file_path.unlink()

    def test_file_not_found(self):
        """
        tests that the function raises a FileNotFoundError for a non-existent file.
        """
        # ensure the function raises the correct exception
        with self.assertRaises(FileNotFoundError):
            process_user_input(user_input_path="non_existent_file.txt")

    def test_empty_file(self):
        """
        tests that the function returns an empty list for an empty file.
        """
        # create an empty mock text file
        file_path = pathlib.Path(f"{pathlib.Path(__file__).parent}/tests/empty_mock_file.txt")
        with open(file_path, "w"):
            pass

        try:
            # call the function to process the empty file
            res = process_user_input(user_input_path=file_path)

            # assert that the result is an empty list
            self.assertEqual(res, [])
        finally:
            # clean up the mock file
            file_path.unlink()