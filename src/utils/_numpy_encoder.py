import json
import numpy as np
from typing import Any


__all__ = ["NumpyEncoder"]

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder to handle NumPy data types.

    This encoder extends the default `json.JSONEncoder` to ensure that NumPy data types
    such as `np.integer`, `np.floating`, and `np.ndarray` are properly serialized into
    native Python types for JSON compatibility.

    Methods:
        default(obj: Any) -> Any:
            Converts NumPy types to their Python equivalents or lists for JSON serialization.

    Usage:
        json_data = json.dumps(your_data, cls=NumpyEncoder)
    """
    def default(self, obj: Any) -> Any:
        """
        Overrides the default method to handle NumPy data types.

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
        return super().default(obj)
