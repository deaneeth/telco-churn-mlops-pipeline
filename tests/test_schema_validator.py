"""
Unit tests for schema_validator module.

Tests cover:
- Valid message validation
- Invalid messages (missing fields, wrong types, invalid values)
- Batch validation
- Edge cases
- Schema loading errors
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from src.streaming.schema_validator import (
    SchemaValidator,
    validate,
    validate_batch,
    TELCO_CUSTOMER_SCHEMA_PATH
)


# Fixtures

@pytest.fixture
def valid_customer_message() -> Dict[str, Any]:
    """Valid customer message matching the schema."""
    return {
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


@pytest.fixture
def valid_customer_message_with_string_total() -> Dict[str, Any]:
    """Valid customer message with TotalCharges as string."""
    return {
        "customerID": "1345-ZUKID",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "Yes",
        "tenure": 14,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 79.50,
        "TotalCharges": "1113.00",  # String format
        "Churn": "Yes",
        "event_ts": "2025-10-11T14:23:45.123456Z"
    }


@pytest.fixture
def validator():
    """Create a SchemaValidator instance."""
    return SchemaValidator()


# Test Class: Schema Loading

class TestSchemaLoading:
    """Tests for schema file loading."""

    def test_schema_file_exists(self):
        """Test that the schema file exists."""
        assert TELCO_CUSTOMER_SCHEMA_PATH.exists(), f"Schema file not found: {TELCO_CUSTOMER_SCHEMA_PATH}"

    def test_schema_is_valid_json(self):
        """Test that the schema file contains valid JSON."""
        with open(TELCO_CUSTOMER_SCHEMA_PATH, 'r') as f:
            schema = json.load(f)
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "required" in schema

    def test_validator_initialization(self, validator):
        """Test validator initializes correctly."""
        assert validator.schema is not None
        assert validator.validator is not None
        assert isinstance(validator.schema, dict)

    def test_custom_schema_path(self):
        """Test validator with custom schema path."""
        # Use the default schema as custom path
        custom_validator = SchemaValidator(schema_path=TELCO_CUSTOMER_SCHEMA_PATH)
        assert custom_validator.schema is not None

    def test_invalid_schema_path_raises_error(self):
        """Test that invalid schema path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SchemaValidator(schema_path=Path("/nonexistent/schema.json"))

    def test_invalid_json_schema_raises_error(self):
        """Test that invalid JSON in schema file raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                SchemaValidator(schema_path=temp_path)
        finally:
            temp_path.unlink()


# Test Class: Valid Messages

class TestValidMessages:
    """Tests for validating correct messages."""

    def test_valid_message_passes(self, validator, valid_customer_message):
        """Test that a valid message passes validation."""
        is_valid, errors = validator.validate(valid_customer_message)
        assert is_valid is True
        assert errors == []

    def test_valid_message_with_string_total_charges(self, validator, valid_customer_message_with_string_total):
        """Test that TotalCharges as string is valid."""
        is_valid, errors = validator.validate(valid_customer_message_with_string_total)
        assert is_valid is True
        assert errors == []

    def test_valid_message_with_empty_string_total_charges(self, validator, valid_customer_message):
        """Test that empty string TotalCharges is valid (new customers)."""
        msg = valid_customer_message.copy()
        msg["TotalCharges"] = " "  # Empty/whitespace for new customers
        is_valid, errors = validator.validate(msg)
        assert is_valid is True
        assert errors == []

    def test_convenience_function_validate(self, valid_customer_message):
        """Test the convenience validate() function."""
        is_valid, errors = validate(valid_customer_message)
        assert is_valid is True
        assert errors == []


# Test Class: Missing Fields

class TestMissingFields:
    """Tests for messages with missing required fields."""

    def test_missing_customer_id(self, validator, valid_customer_message):
        """Test validation fails when customerID is missing."""
        msg = valid_customer_message.copy()
        del msg["customerID"]
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert len(errors) > 0
        assert any("customerID" in err for err in errors)

    def test_missing_multiple_fields(self, validator):
        """Test validation fails when multiple required fields are missing."""
        msg = {
            "customerID": "1234-ABCDE",
            "gender": "Male",
            "tenure": 5
            # Missing many required fields
        }
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert len(errors) > 10  # Should have many missing field errors

    def test_missing_event_ts(self, validator, valid_customer_message):
        """Test validation fails when event_ts is missing."""
        msg = valid_customer_message.copy()
        del msg["event_ts"]
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert any("event_ts" in err for err in errors)

    def test_empty_message(self, validator):
        """Test validation fails for empty message."""
        is_valid, errors = validator.validate({})
        assert is_valid is False
        assert len(errors) > 0


# Test Class: Type Errors

class TestTypeErrors:
    """Tests for messages with incorrect field types."""

    def test_customer_id_not_string(self, validator, valid_customer_message):
        """Test validation fails when customerID is not a string."""
        msg = valid_customer_message.copy()
        msg["customerID"] = 12345  # Should be string
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert any("customerID" in err for err in errors)

    def test_senior_citizen_not_integer(self, validator, valid_customer_message):
        """Test validation fails when SeniorCitizen is not an integer."""
        msg = valid_customer_message.copy()
        msg["SeniorCitizen"] = "Yes"  # Should be 0 or 1
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert any("SeniorCitizen" in err for err in errors)

    def test_tenure_not_integer(self, validator, valid_customer_message):
        """Test validation fails when tenure is not an integer."""
        msg = valid_customer_message.copy()
        msg["tenure"] = "five"  # Should be integer
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert any("tenure" in err for err in errors)

    def test_monthly_charges_not_number(self, validator, valid_customer_message):
        """Test validation fails when MonthlyCharges is not a number."""
        msg = valid_customer_message.copy()
        msg["MonthlyCharges"] = "twenty dollars"  # Should be number
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert any("MonthlyCharges" in err for err in errors)


# Test Class: Invalid Values

class TestInvalidValues:
    """Tests for messages with values that violate constraints."""

    def test_invalid_gender_enum(self, validator, valid_customer_message):
        """Test validation fails for invalid gender value."""
        msg = valid_customer_message.copy()
        msg["gender"] = "Unknown"  # Not in enum
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert any("gender" in err for err in errors)

    def test_invalid_senior_citizen_value(self, validator, valid_customer_message):
        """Test validation fails when SeniorCitizen is not 0 or 1."""
        msg = valid_customer_message.copy()
        msg["SeniorCitizen"] = 2  # Must be 0 or 1
        is_valid, errors = validator.validate(msg)
        assert is_valid is False

    def test_negative_tenure(self, validator, valid_customer_message):
        """Test validation fails for negative tenure."""
        msg = valid_customer_message.copy()
        msg["tenure"] = -5  # Must be >= 0
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert any("tenure" in err for err in errors)

    def test_excessive_tenure(self, validator, valid_customer_message):
        """Test validation fails for tenure > 100."""
        msg = valid_customer_message.copy()
        msg["tenure"] = 150  # Must be <= 100
        is_valid, errors = validator.validate(msg)
        assert is_valid is False

    def test_negative_monthly_charges(self, validator, valid_customer_message):
        """Test validation fails for negative MonthlyCharges."""
        msg = valid_customer_message.copy()
        msg["MonthlyCharges"] = -10.00
        is_valid, errors = validator.validate(msg)
        assert is_valid is False

    def test_invalid_contract_type(self, validator, valid_customer_message):
        """Test validation fails for invalid Contract value."""
        msg = valid_customer_message.copy()
        msg["Contract"] = "Three year"  # Not in enum
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert any("Contract" in err for err in errors)

    def test_invalid_payment_method(self, validator, valid_customer_message):
        """Test validation fails for invalid PaymentMethod."""
        msg = valid_customer_message.copy()
        msg["PaymentMethod"] = "PayPal"  # Not in enum
        is_valid, errors = validator.validate(msg)
        assert is_valid is False

    def test_invalid_customer_id_pattern(self, validator, valid_customer_message):
        """Test validation fails for customerID not matching pattern."""
        msg = valid_customer_message.copy()
        msg["customerID"] = "INVALID"  # Should be ####-LLLLL format
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert any("customerID" in err for err in errors)


# Test Class: Edge Cases

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_tenure(self, validator, valid_customer_message):
        """Test that tenure=0 is valid (new customer)."""
        msg = valid_customer_message.copy()
        msg["tenure"] = 0
        is_valid, errors = validator.validate(msg)
        assert is_valid is True

    def test_max_tenure(self, validator, valid_customer_message):
        """Test that tenure=100 is valid (boundary)."""
        msg = valid_customer_message.copy()
        msg["tenure"] = 100
        is_valid, errors = validator.validate(msg)
        assert is_valid is True

    def test_zero_monthly_charges(self, validator, valid_customer_message):
        """Test that MonthlyCharges=0 is valid."""
        msg = valid_customer_message.copy()
        msg["MonthlyCharges"] = 0.0
        is_valid, errors = validator.validate(msg)
        assert is_valid is True

    def test_high_monthly_charges(self, validator, valid_customer_message):
        """Test that high MonthlyCharges within range is valid."""
        msg = valid_customer_message.copy()
        msg["MonthlyCharges"] = 199.99
        is_valid, errors = validator.validate(msg)
        assert is_valid is True

    def test_all_yes_no_internet_service(self, validator, valid_customer_message):
        """Test message with 'No internet service' values."""
        msg = valid_customer_message.copy()
        msg["InternetService"] = "No"
        msg["OnlineSecurity"] = "No internet service"
        msg["OnlineBackup"] = "No internet service"
        msg["DeviceProtection"] = "No internet service"
        msg["TechSupport"] = "No internet service"
        msg["StreamingTV"] = "No internet service"
        msg["StreamingMovies"] = "No internet service"
        is_valid, errors = validator.validate(msg)
        assert is_valid is True

    def test_additional_properties_rejected(self, validator, valid_customer_message):
        """Test that additional properties are rejected."""
        msg = valid_customer_message.copy()
        msg["extraField"] = "should not be here"
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert any("additional" in err.lower() for err in errors)


# Test Class: Batch Validation

class TestBatchValidation:
    """Tests for batch validation functionality."""

    def test_batch_all_valid(self, validator, valid_customer_message, valid_customer_message_with_string_total):
        """Test batch validation with all valid messages."""
        messages = [valid_customer_message, valid_customer_message_with_string_total]
        validity, errors_list = validator.validate_batch(messages)
        assert all(validity)
        assert all(len(errs) == 0 for errs in errors_list)

    def test_batch_mixed_validity(self, validator, valid_customer_message):
        """Test batch validation with mixed valid/invalid messages."""
        invalid_msg = {"customerID": "1234-ABCDE"}  # Missing fields
        messages = [valid_customer_message, invalid_msg]
        validity, errors_list = validator.validate_batch(messages)
        assert validity[0] is True
        assert validity[1] is False
        assert len(errors_list[0]) == 0
        assert len(errors_list[1]) > 0

    def test_batch_all_invalid(self, validator):
        """Test batch validation with all invalid messages."""
        messages = [{}, {"customerID": "bad"}, {"tenure": -1}]
        validity, errors_list = validator.validate_batch(messages)
        assert all(not v for v in validity)
        assert all(len(errs) > 0 for errs in errors_list)

    def test_convenience_function_validate_batch(self, valid_customer_message):
        """Test the convenience validate_batch() function."""
        messages = [valid_customer_message, valid_customer_message]
        validity, errors_list = validate_batch(messages)
        assert all(validity)


# Test Class: Validator Helper Methods

class TestValidatorHelperMethods:
    """Tests for validator helper methods."""

    def test_get_required_fields(self, validator):
        """Test getting list of required fields."""
        required = validator.get_required_fields()
        assert isinstance(required, list)
        assert "customerID" in required
        assert "event_ts" in required
        assert "tenure" in required
        assert len(required) == 22  # 21 original fields + event_ts

    def test_get_field_type_string(self, validator):
        """Test getting field type for string fields."""
        field_type = validator.get_field_type("customerID")
        assert field_type == "string"

    def test_get_field_type_integer(self, validator):
        """Test getting field type for integer fields."""
        field_type = validator.get_field_type("tenure")
        assert field_type == "integer"

    def test_get_field_type_number(self, validator):
        """Test getting field type for number fields."""
        field_type = validator.get_field_type("MonthlyCharges")
        assert field_type == "number"

    def test_get_field_type_multiple_types(self, validator):
        """Test getting field type for fields with multiple types."""
        field_type = validator.get_field_type("TotalCharges")
        assert "number" in field_type or "string" in field_type

    def test_get_field_type_nonexistent(self, validator):
        """Test getting field type for nonexistent field."""
        field_type = validator.get_field_type("nonexistent_field")
        assert field_type is None


# Test Class: Error Messages

class TestErrorMessages:
    """Tests for error message clarity and formatting."""

    def test_error_messages_contain_field_names(self, validator, valid_customer_message):
        """Test that error messages contain field names."""
        msg = valid_customer_message.copy()
        del msg["customerID"]
        del msg["gender"]
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        # Check that error messages mention the missing fields
        error_text = " ".join(errors)
        assert "customerID" in error_text or "customer" in error_text.lower()

    def test_error_messages_for_type_mismatch(self, validator, valid_customer_message):
        """Test error messages for type mismatches."""
        msg = valid_customer_message.copy()
        msg["tenure"] = "not a number"
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        assert len(errors) > 0
        # Error should mention tenure and type issue
        error_text = " ".join(errors).lower()
        assert "tenure" in error_text

    def test_multiple_errors_all_reported(self, validator):
        """Test that all validation errors are reported."""
        msg = {
            "customerID": 12345,  # Wrong type
            "gender": "Unknown",  # Invalid enum
            "SeniorCitizen": "Yes",  # Wrong type
            "Partner": "Maybe",  # Invalid enum
            # Missing many fields
        }
        is_valid, errors = validator.validate(msg)
        assert is_valid is False
        # Should have multiple errors
        assert len(errors) > 5


# Test Class: Integration Tests

class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_producer_generated_message_format(self, validator):
        """Test validation of message in producer format."""
        # Simulate a message as generated by producer
        msg = {
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
            "TotalCharges": "29.85",  # Producer may send as string
            "Churn": "No",
            "event_ts": "2025-10-11T10:30:00.123Z"  # ISO format with milliseconds
        }
        is_valid, errors = validator.validate(msg)
        assert is_valid is True, f"Validation errors: {errors}"

    def test_csv_row_conversion(self, validator):
        """Test validation of message converted from CSV row."""
        # Simulate a message from CSV (TotalCharges might be empty for new customers)
        msg = {
            "customerID": "0001-NEWCX",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 0,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.0,
            "TotalCharges": " ",  # Empty for new customer
            "Churn": "No",
            "event_ts": "2025-10-11T00:00:00Z"
        }
        is_valid, errors = validator.validate(msg)
        assert is_valid is True, f"Validation errors: {errors}"


if __name__ == "__main__":
    # Run pytest when module is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
