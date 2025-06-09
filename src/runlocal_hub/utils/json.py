"""
JSON encoding utilities for handling RunLocal data types.
"""

import json
from decimal import Decimal
from typing import Any, Dict, List, Union

from pydantic import BaseModel


class RunLocalJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for RunLocal data types.

    Handles:
    - Decimal to float conversion
    - Pydantic model serialization
    - Cleaning of None values and empty collections
    """

    def default(self, o: Any) -> Any:
        """
        Convert special types to JSON-serializable formats.

        Args:
            obj: Object to encode

        Returns:
            JSON-serializable representation
        """
        if isinstance(o, Decimal):
            return float(o)

        if isinstance(o, BaseModel):
            # Convert Pydantic model to dict and clean it
            return self._clean_dict(o.model_dump())

        # Let the base class handle other types
        return super().default(o)

    def encode(self, o: Any) -> str:
        """
        Encode object to JSON string with cleaning.

        Args:
            obj: Object to encode

        Returns:
            JSON string
        """
        # Clean the object before encoding
        cleaned = self._clean_value(o)
        return super().encode(cleaned)

    def _clean_value(self, value: Any) -> Any:
        """
        Recursively clean a value by removing None values and empty collections.

        Args:
            value: Value to clean

        Returns:
            Cleaned value
        """
        if value is None:
            return None

        if isinstance(value, dict):
            return self._clean_dict(value)

        if isinstance(value, list):
            return self._clean_list(value)

        if isinstance(value, Decimal):
            return float(value)

        if isinstance(value, BaseModel):
            return self._clean_dict(value.model_dump())

        return value

    def _clean_dict(self, d: Dict) -> Dict:
        """
        Clean a dictionary by removing None values and empty collections.

        Args:
            d: Dictionary to clean

        Returns:
            Cleaned dictionary
        """
        result = {}

        for key, value in d.items():
            # Skip None values
            if value is None:
                continue

            # Recursively clean the value
            cleaned_value = self._clean_value(value)

            # Skip empty collections after cleaning
            if isinstance(cleaned_value, (list, dict)) and not cleaned_value:
                continue

            # Only include non-None cleaned values
            if cleaned_value is not None:
                result[key] = cleaned_value

        return result

    def _clean_list(self, lst: List) -> List:
        """
        Clean a list by removing None values and cleaning nested structures.

        Args:
            lst: List to clean

        Returns:
            Cleaned list
        """
        result = []

        for item in lst:
            if item is None:
                continue

            # Recursively clean the item
            cleaned_item = self._clean_value(item)

            # Skip empty collections after cleaning
            if isinstance(cleaned_item, (list, dict)) and not cleaned_item:
                continue

            # Only include non-None cleaned items
            if cleaned_item is not None:
                result.append(cleaned_item)

        return result


def convert_to_json_friendly(obj: Any) -> Any:
    """
    Convert any object to a JSON-friendly format.

    This function:
    - Converts Decimal to float
    - Removes None values
    - Removes empty collections
    - Handles nested structures

    Args:
        obj: Object to convert

    Returns:
        JSON-friendly representation
    """
    # Use our custom encoder to convert to JSON string and back
    # This ensures all conversions are applied
    json_str = json.dumps(obj, cls=RunLocalJSONEncoder)
    return json.loads(json_str)


def decimal_to_str(value: Union[Decimal, None]) -> Union[str, None]:
    """
    Convert a Decimal to string for API requests.

    Args:
        value: Decimal value or None

    Returns:
        String representation or None
    """
    if value is None:
        return None
    return str(value)


def decimal_list_to_str(values: Union[List[Decimal], None]) -> Union[List[str], None]:
    """
    Convert a list of Decimals to list of strings for API requests.

    Args:
        values: List of Decimal values or None

    Returns:
        List of string representations or None
    """
    if values is None:
        return None
    return [str(v) for v in values]

