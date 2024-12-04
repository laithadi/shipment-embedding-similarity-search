import pathlib
import datetime
import re


__all__ = ["generate_versioned_filename"]

def generate_versioned_filename(directory: pathlib.Path, prefix: str = "", suffix: str = ".json") -> str:
    """
    Generates a versioned JSON filename based on the current date and time.

    The version number is automatically incremented based on the latest version found in the directory.

    Args:
        directory (pathlib.Path): The directory to search for existing versioned files.
        prefix (str): Optional prefix for the filename (default is an empty string).
        suffix (str): Optional suffix for the filename (default is ".json").

    Returns:
        str: The generated versioned filename.
    """
    # ensure the directory exists
    directory.mkdir(parents= True, exist_ok= True)

    # get current date and time in the format YYYYMMDD_HHMMSS
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # regex to match files with the pattern: YYYYMMDD_HHMMSS_version#.json
    pattern = re.compile(rf"{prefix}{current_time}_v(\d+){suffix}")

    # find all matching files in the directory
    existing_files = [f.name for f in directory.glob(f"{prefix}{current_time}_v*{suffix}")]
    
    # extract version numbers
    versions = [int(pattern.match(f).group(1)) for f in existing_files if pattern.match(f)]

    # determine the next version number
    next_version = max(versions, default=0) + 1

    # Generate the filename
    filename = f"{prefix}{current_time}_v{next_version}{suffix}"
    return filename
