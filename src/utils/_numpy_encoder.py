import json
import numpy as np
import pandas as pd
from typing import Any


__all__ = ["NumpyEncoder"]

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder to handle NumPy and pandas data types.

    This encoder extends the default `json.JSONEncoder` to ensure that NumPy data types
    such as `np.integer`, `np.floating`, `np.ndarray`, and pandas `Timestamp` objects
    are properly serialized into native Python types or JSON-compatible formats.

    Methods:
        default(obj: Any) -> Any:
            Converts NumPy and pandas types to their JSON-serializable equivalents.

    Usage:
        json_data = json.dumps(your_data, cls=NumpyEncoder)
    """
    def default(self, obj: Any) -> Any:
        """
        Overrides the default method to handle NumPy and pandas types.

        Args:
            obj (Any): The object to be serialized.

        Returns:
            Any: A JSON-serializable representation of the object.

        Raises:
            TypeError: If the object cannot be serialized.
        """
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, pd.Timestamp):
            # Check if the timestamp has a non-midnight time
            if obj.time() == pd.Timestamp("00:00:00").time():
                return obj.strftime("%Y-%m-%d")  # Date only
            return obj.strftime("%Y-%m-%d %H:%M:%S")  # Date and time
        return super().default(obj)