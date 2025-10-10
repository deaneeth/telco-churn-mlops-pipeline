"""
Schema validation utilities for Kafka messages.

This module provides validation functions for telco customer event messages
using JSON Schema validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    import jsonschema
    from jsonschema import Draft7Validator, validators
except ImportError:
    raise ImportError(
        "jsonschema library is required for schema validation. "
        "Install it with: pip install jsonschema>=4.0.0"
    )

logger = logging.getLogger(__name__)

# Path to schema file
SCHEMA_DIR = Path(__file__).parent.parent.parent / "schemas"
TELCO_CUSTOMER_SCHEMA_PATH = SCHEMA_DIR / "telco_customer_schema.json"


class SchemaValidator:
    """Validates messages against JSON Schema."""

    def __init__(self, schema_path: Optional[Path] = None):
        """
        Initialize validator with schema.

        Args:
            schema_path: Path to JSON schema file. If None, uses default telco customer schema.
        """
        self.schema_path = schema_path or TELCO_CUSTOMER_SCHEMA_PATH
        self.schema = self._load_schema()
        self.validator = Draft7Validator(self.schema)

    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema from file."""
        if not self.schema_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {self.schema_path}\n"
                f"Expected location: {self.schema_path.absolute()}"
            )

        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.debug(f"Loaded schema from {self.schema_path}")
            return schema
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file {self.schema_path}: {e}")

    def validate(self, message: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a message against the schema.

        Args:
            message: Dictionary representing the message to validate

        Returns:
            Tuple of (is_valid, error_messages)
            - is_valid: True if message is valid, False otherwise
            - error_messages: List of validation error messages (empty if valid)

        Examples:
            >>> validator = SchemaValidator()
            >>> message = {"customerID": "1234-ABCDE", "gender": "Male", ...}
            >>> is_valid, errors = validator.validate(message)
            >>> if not is_valid:
            ...     print(f"Validation errors: {errors}")
        """
        errors = []

        try:
            # Validate against schema
            for error in self.validator.iter_errors(message):
                # Format error message with path and description
                path = ".".join(str(p) for p in error.path) if error.path else "root"
                error_msg = f"Field '{path}': {error.message}"
                errors.append(error_msg)

            is_valid = len(errors) == 0

            if is_valid:
                logger.debug("Message validation passed")
            else:
                logger.warning(f"Message validation failed with {len(errors)} error(s)")

            return is_valid, errors

        except Exception as e:
            # Catch any unexpected validation errors
            logger.error(f"Unexpected error during validation: {e}")
            return False, [f"Validation error: {str(e)}"]

    def validate_batch(self, messages: List[Dict[str, Any]]) -> Tuple[List[bool], List[List[str]]]:
        """
        Validate multiple messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Tuple of (validity_list, errors_list)
            - validity_list: List of boolean validity flags for each message
            - errors_list: List of error message lists for each message

        Examples:
            >>> validator = SchemaValidator()
            >>> messages = [msg1, msg2, msg3]
            >>> validity, errors = validator.validate_batch(messages)
            >>> for i, (valid, errs) in enumerate(zip(validity, errors)):
            ...     if not valid:
            ...         print(f"Message {i} errors: {errs}")
        """
        validity_list = []
        errors_list = []

        for i, message in enumerate(messages):
            is_valid, errors = self.validate(message)
            validity_list.append(is_valid)
            errors_list.append(errors)

        valid_count = sum(validity_list)
        total_count = len(messages)
        logger.info(f"Batch validation: {valid_count}/{total_count} messages valid")

        return validity_list, errors_list

    def get_required_fields(self) -> List[str]:
        """
        Get list of required fields from schema.

        Returns:
            List of required field names
        """
        return self.schema.get("required", [])

    def get_field_type(self, field_name: str) -> Optional[str]:
        """
        Get the expected type for a field.

        Args:
            field_name: Name of the field

        Returns:
            Type string (e.g., 'string', 'number', 'integer') or None if field not found
        """
        properties = self.schema.get("properties", {})
        if field_name in properties:
            field_type = properties[field_name].get("type")
            # Handle multiple types (like TotalCharges)
            if isinstance(field_type, list):
                return "/".join(field_type)
            return field_type
        return None


# Convenience function for simple validation
def validate(message: Dict[str, Any], schema_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a single message.

    This is a simple wrapper around SchemaValidator.validate() for quick validation.

    Args:
        message: Dictionary representing the message to validate
        schema_path: Optional path to custom schema file

    Returns:
        Tuple of (is_valid, error_messages)

    Examples:
        >>> from src.streaming.schema_validator import validate
        >>> sample = {
        ...     "customerID": "7590-VHVEG",
        ...     "gender": "Female",
        ...     # ... other fields
        ... }
        >>> is_valid, errors = validate(sample)
        >>> print(f"Valid: {is_valid}")
    """
    validator = SchemaValidator(schema_path)
    return validator.validate(message)


def validate_batch(messages: List[Dict[str, Any]], schema_path: Optional[Path] = None) -> Tuple[List[bool], List[List[str]]]:
    """
    Convenience function to validate multiple messages.

    Args:
        messages: List of message dictionaries
        schema_path: Optional path to custom schema file

    Returns:
        Tuple of (validity_list, errors_list)

    Examples:
        >>> from src.streaming.schema_validator import validate_batch
        >>> messages = [msg1, msg2, msg3]
        >>> validity, errors = validate_batch(messages)
    """
    validator = SchemaValidator(schema_path)
    return validator.validate_batch(messages)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create a sample valid message
    valid_sample = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "Churn": "No",
        "event_ts": "2025-10-03T04:00:00Z"
    }

    # Create an invalid sample (missing required fields)
    invalid_sample = {
        "customerID": "1234-ABCDE",
        "gender": "Male",
        "tenure": 5
        # Missing many required fields
    }

    # Create a sample with type errors
    type_error_sample = {
        "customerID": "7590-VHVEG",
        "gender": "Unknown",  # Invalid enum value
        "SeniorCitizen": "Yes",  # Should be integer
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -5,  # Negative value not allowed
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "Churn": "No",
        "event_ts": "2025-10-03T04:00:00Z"
    }

    print("=" * 80)
    print("SCHEMA VALIDATOR TEST")
    print("=" * 80)

    # Test 1: Valid message
    print("\n[Test 1] Valid message:")
    is_valid, errors = validate(valid_sample)
    print(f"✓ Valid: {is_valid}")
    if errors:
        print(f"  Errors: {errors}")

    # Test 2: Invalid message (missing fields)
    print("\n[Test 2] Invalid message (missing fields):")
    is_valid, errors = validate(invalid_sample)
    print(f"✗ Valid: {is_valid}")
    if errors:
        print(f"  Errors ({len(errors)} total):")
        for error in errors[:5]:  # Show first 5 errors
            print(f"    - {error}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors")

    # Test 3: Type errors
    print("\n[Test 3] Type errors:")
    is_valid, errors = validate(type_error_sample)
    print(f"✗ Valid: {is_valid}")
    if errors:
        print(f"  Errors ({len(errors)} total):")
        for error in errors:
            print(f"    - {error}")

    # Test 4: Batch validation
    print("\n[Test 4] Batch validation:")
    messages = [valid_sample, invalid_sample, type_error_sample]
    validity, errors_list = validate_batch(messages)
    for i, (valid, errs) in enumerate(zip(validity, errors_list)):
        status = "✓" if valid else "✗"
        print(f"  Message {i+1}: {status} Valid={valid}, Errors={len(errs)}")

    print("\n" + "=" * 80)
