# Kafka Message Schema Documentation

**Version:** 1.0.0  
**Last Updated:** October 11, 2025  
**Project:** Telco Customer Churn Prediction - Mini Project 2

---

## Table of Contents

1. [Overview](#overview)
2. [Message Schema](#message-schema)
3. [Field Definitions](#field-definitions)
4. [Validation Rules](#validation-rules)
5. [Usage Examples](#usage-examples)
6. [Error Handling](#error-handling)
7. [Integration Guide](#integration-guide)

---

## Overview

### Purpose

This document defines the JSON schema for telco customer event messages used in the Kafka-based streaming pipeline for real-time churn prediction. All messages published to and consumed from Kafka topics must conform to this schema.

### Schema Location

- **JSON Schema File:** `schemas/telco_customer_schema.json`
- **Validator Module:** `src/streaming/schema_validator.py`
- **Test Suite:** `tests/test_schema_validator.py`

### Topics Using This Schema

- **`telco.raw.customers`** - Raw customer events from producer
- **`telco.churn.predictions`** - Prediction results (extends this schema)
- **`telco.deadletter`** - Invalid messages for debugging

---

## Message Schema

### Schema Format

The schema follows **JSON Schema Draft-07** specification.

### Required Fields (22 total)

All fields listed below are **required** in every message:

1. `customerID` (string)
2. `gender` (string)
3. `SeniorCitizen` (integer)
4. `Partner` (string)
5. `Dependents` (string)
6. `tenure` (integer)
7. `PhoneService` (string)
8. `MultipleLines` (string)
9. `InternetService` (string)
10. `OnlineSecurity` (string)
11. `OnlineBackup` (string)
12. `DeviceProtection` (string)
13. `TechSupport` (string)
14. `StreamingTV` (string)
15. `StreamingMovies` (string)
16. `Contract` (string)
17. `PaperlessBilling` (string)
18. `PaymentMethod` (string)
19. `MonthlyCharges` (number)
20. `TotalCharges` (number or string)
21. `Churn` (string)
22. `event_ts` (string, ISO 8601 format)

### Example Message

```json
{
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
```

---

## Field Definitions

### Customer Identifier

#### `customerID`
- **Type:** String
- **Pattern:** `^[0-9]{4}-[A-Z]{5}$` (4 digits, hyphen, 5 uppercase letters)
- **Description:** Unique customer identifier
- **Examples:** `"7590-VHVEG"`, `"1345-ZUKID"`
- **Constraints:** Must match the pattern exactly

### Demographics

#### `gender`
- **Type:** String (enum)
- **Allowed Values:** `"Male"`, `"Female"`
- **Description:** Customer gender
- **Example:** `"Female"`

#### `SeniorCitizen`
- **Type:** Integer
- **Allowed Values:** `0` (No), `1` (Yes)
- **Range:** 0-1 (inclusive)
- **Description:** Whether customer is a senior citizen
- **Example:** `0`

#### `Partner`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`
- **Description:** Whether customer has a partner
- **Example:** `"Yes"`

#### `Dependents`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`
- **Description:** Whether customer has dependents
- **Example:** `"No"`

### Account Information

#### `tenure`
- **Type:** Integer
- **Range:** 0-100 (months)
- **Description:** Number of months customer has stayed with the company
- **Example:** `1`
- **Notes:** 0 indicates new customer

### Phone Services

#### `PhoneService`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`
- **Description:** Whether customer has phone service
- **Example:** `"No"`

#### `MultipleLines`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`, `"No phone service"`
- **Description:** Whether customer has multiple phone lines
- **Example:** `"No phone service"`
- **Notes:** Use `"No phone service"` when `PhoneService` is `"No"`

### Internet Services

#### `InternetService`
- **Type:** String (enum)
- **Allowed Values:** `"DSL"`, `"Fiber optic"`, `"No"`
- **Description:** Type of internet service
- **Example:** `"DSL"`

#### `OnlineSecurity`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`, `"No internet service"`
- **Description:** Whether customer has online security add-on
- **Example:** `"No"`

#### `OnlineBackup`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`, `"No internet service"`
- **Description:** Whether customer has online backup add-on
- **Example:** `"Yes"`

#### `DeviceProtection`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`, `"No internet service"`
- **Description:** Whether customer has device protection add-on
- **Example:** `"No"`

#### `TechSupport`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`, `"No internet service"`
- **Description:** Whether customer has tech support add-on
- **Example:** `"No"`

### Streaming Services

#### `StreamingTV`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`, `"No internet service"`
- **Description:** Whether customer has streaming TV service
- **Example:** `"No"`

#### `StreamingMovies`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`, `"No internet service"`
- **Description:** Whether customer has streaming movies service
- **Example:** `"No"`

### Billing

#### `Contract`
- **Type:** String (enum)
- **Allowed Values:** `"Month-to-month"`, `"One year"`, `"Two year"`
- **Description:** Type of contract
- **Example:** `"Month-to-month"`

#### `PaperlessBilling`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`
- **Description:** Whether customer uses paperless billing
- **Example:** `"Yes"`

#### `PaymentMethod`
- **Type:** String (enum)
- **Allowed Values:** 
  - `"Electronic check"`
  - `"Mailed check"`
  - `"Bank transfer (automatic)"`
  - `"Credit card (automatic)"`
- **Description:** Customer payment method
- **Example:** `"Electronic check"`

#### `MonthlyCharges`
- **Type:** Number
- **Range:** 0-200 (dollars)
- **Description:** Monthly charges in dollars
- **Example:** `29.85`
- **Format:** Decimal number with up to 2 decimal places

#### `TotalCharges`
- **Type:** Number or String
- **Formats:**
  - Number: `29.85` (preferred)
  - String (numeric): `"29.85"` (legacy compatibility)
  - String (empty): `" "` (new customers with no history)
- **Range:** 0+ (when numeric)
- **Description:** Total charges to date in dollars
- **Example:** `29.85` or `"29.85"` or `" "`
- **Notes:** Empty string indicates new customer (tenure=0)

### Churn Status

#### `Churn`
- **Type:** String (enum)
- **Allowed Values:** `"Yes"`, `"No"`
- **Description:** Whether customer has churned (historical data) or will churn (prediction target)
- **Example:** `"No"`

### Event Metadata

#### `event_ts`
- **Type:** String (ISO 8601 datetime)
- **Format:** `YYYY-MM-DDTHH:MM:SS[.ffffff]Z`
- **Pattern:** `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?$`
- **Description:** Event timestamp in UTC timezone
- **Examples:** 
  - `"2025-10-03T04:00:00Z"`
  - `"2025-10-11T14:23:45.123456Z"`
- **Notes:** 
  - Must be in UTC (trailing 'Z')
  - Microseconds are optional
  - Added by producer when event is generated

---

## Validation Rules

### Schema Validation

Messages are validated using **JSON Schema Draft-07** against `schemas/telco_customer_schema.json`.

### Validation Constraints

1. **Required Fields:** All 22 fields must be present
2. **No Additional Properties:** Extra fields not in schema are **rejected**
3. **Type Checking:** Each field must match its specified type
4. **Enum Validation:** String fields with enums must use exact values (case-sensitive)
5. **Range Validation:** Numeric fields must be within specified min/max ranges
6. **Pattern Matching:** `customerID` and `event_ts` must match regex patterns

### Common Validation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| Missing required property | Field omitted from message | Include all 22 required fields |
| Not of type 'integer' | Wrong data type (e.g., string instead of int) | Convert to correct type |
| Not one of enum values | Invalid enum value (e.g., "Unknown" for gender) | Use only allowed enum values |
| Less than minimum | Value below range (e.g., tenure=-5) | Ensure value is within valid range |
| Pattern mismatch | customerID format wrong | Use ####-LLLLL format |
| Additional properties not allowed | Extra fields in message | Remove fields not in schema |

---

## Usage Examples

### Python - Validating Messages

#### Basic Validation

```python
from src.streaming.schema_validator import validate

# Create a message
message = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    # ... all other required fields
    "event_ts": "2025-10-03T04:00:00Z"
}

# Validate
is_valid, errors = validate(message)

if is_valid:
    print("✓ Message is valid")
else:
    print(f"✗ Validation failed with {len(errors)} error(s):")
    for error in errors:
        print(f"  - {error}")
```

#### Batch Validation

```python
from src.streaming.schema_validator import validate_batch

messages = [msg1, msg2, msg3]
validity, errors_list = validate_batch(messages)

for i, (valid, errs) in enumerate(zip(validity, errors_list)):
    if valid:
        print(f"Message {i+1}: ✓ Valid")
    else:
        print(f"Message {i+1}: ✗ {len(errs)} error(s)")
```

#### Using SchemaValidator Class

```python
from src.streaming.schema_validator import SchemaValidator

# Initialize validator
validator = SchemaValidator()

# Get schema info
required_fields = validator.get_required_fields()
print(f"Required fields: {len(required_fields)}")

# Check field type
field_type = validator.get_field_type("MonthlyCharges")
print(f"MonthlyCharges type: {field_type}")

# Validate message
is_valid, errors = validator.validate(message)
```

### Producer - Using Validation Flag

#### Enable Validation in Producer

```bash
# Streaming mode with validation
python src/streaming/producer.py \
    --mode streaming \
    --events-per-sec 5 \
    --broker localhost:19092 \
    --validate

# Batch mode with validation
python src/streaming/producer.py \
    --mode batch \
    --batch-size 100 \
    --validate
```

#### Producer Output with Validation

```
2025-10-11 15:30:00 - kafka_producer - INFO - Validation: ENABLED
2025-10-11 15:30:00 - kafka_producer - INFO - Schema validator initialized successfully
2025-10-11 15:30:00 - kafka_producer - INFO - Starting STREAMING mode: 5.0 events/sec
...
2025-10-11 15:30:45 - kafka_producer - INFO - STREAMING MODE SUMMARY
2025-10-11 15:30:45 - kafka_producer - INFO - Total messages sent: 225
2025-10-11 15:30:45 - kafka_producer - INFO - Total failures: 0
2025-10-11 15:30:45 - kafka_producer - INFO - Validation failures: 0
```

### Consumer - Validating Consumed Messages

```python
from kafka import KafkaConsumer
from src.streaming.schema_validator import SchemaValidator
import json

# Initialize consumer and validator
consumer = KafkaConsumer('telco.raw.customers', ...)
validator = SchemaValidator()

for message in consumer:
    # Decode message
    data = json.loads(message.value.decode('utf-8'))
    
    # Validate
    is_valid, errors = validator.validate(data)
    
    if is_valid:
        # Process valid message
        process_customer(data)
    else:
        # Route to dead letter queue
        send_to_dlq(data, errors)
```

---

## Error Handling

### Validation Error Format

Errors are returned as a list of strings with format:
```
Field '<field_name>': <error_description>
```

### Example Error Messages

```python
[
    "Field 'customerID': '12345' does not match '^[0-9]{4}-[A-Z]{5}$'",
    "Field 'gender': 'Unknown' is not one of ['Male', 'Female']",
    "Field 'tenure': -5 is less than the minimum of 0",
    "Field 'root': 'event_ts' is a required property"
]
```

### Invalid Message Handling

#### Strategy 1: Reject and Log

```python
is_valid, errors = validate(message)
if not is_valid:
    logger.error(f"Invalid message: {errors}")
    return False  # Skip message
```

#### Strategy 2: Dead Letter Queue

```python
is_valid, errors = validate(message)
if not is_valid:
    dead_letter_message = {
        "original_message": message,
        "validation_errors": errors,
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "source_topic": "telco.raw.customers"
    }
    send_to_dlq(dead_letter_message)
```

#### Strategy 3: Metrics and Alerting

```python
is_valid, errors = validate(message)
if not is_valid:
    metrics.increment('validation.failed', tags=['source:producer'])
    if len(errors) > 5:
        alert_ops_team(f"High validation error count: {len(errors)}")
```

---

## Integration Guide

### Producer Integration

1. **Enable Validation (Optional)**
   ```bash
   python src/streaming/producer.py --mode streaming --validate
   ```

2. **Producer automatically:**
   - Initializes `SchemaValidator` when `--validate` flag is set
   - Validates each message before publishing
   - Logs validation failures
   - Tracks validation failure metrics
   - Skips invalid messages (not published to Kafka)

### Consumer Integration

1. **Import Validator**
   ```python
   from src.streaming.schema_validator import SchemaValidator
   ```

2. **Initialize in Consumer**
   ```python
   class ChurnConsumer:
       def __init__(self):
           self.validator = SchemaValidator()
   ```

3. **Validate on Consumption**
   ```python
   def process_message(self, message):
       is_valid, errors = self.validator.validate(message)
       if not is_valid:
           self.handle_invalid_message(message, errors)
           return
       # Process valid message...
   ```

### Testing Integration

1. **Unit Tests**
   ```bash
   pytest tests/test_schema_validator.py -v
   ```

2. **Integration Tests**
   ```python
   def test_producer_messages_are_valid():
       # Produce messages
       produce_test_messages()
       
       # Consume and validate
       validator = SchemaValidator()
       for msg in consume_test_messages():
           is_valid, errors = validator.validate(msg)
           assert is_valid, f"Invalid message: {errors}"
   ```

### Performance Considerations

- **Validation overhead:** ~0.5-1ms per message
- **Recommended:** Use validation in development/testing
- **Production:** Consider sampling (validate 10% of messages) or disable for high-throughput scenarios
- **Batch validation:** More efficient for bulk processing

---

## Troubleshooting

### Common Issues

#### Issue 1: jsonschema not installed

**Error:**
```
ImportError: jsonschema library is required for schema validation
```

**Solution:**
```bash
pip install jsonschema>=4.0.0
```

#### Issue 2: Schema file not found

**Error:**
```
FileNotFoundError: Schema file not found: schemas/telco_customer_schema.json
```

**Solution:**
- Ensure you're running from project root directory
- Verify schema file exists: `ls schemas/telco_customer_schema.json`

#### Issue 3: All messages fail validation

**Error:**
```
Field 'root': 'event_ts' is a required property
```

**Solution:**
- Ensure `event_ts` is added to all messages
- Use `customer_to_message(customer, add_timestamp=True)` in producer

#### Issue 4: TotalCharges validation fails

**Error:**
```
Field 'TotalCharges': 'NaN' does not match pattern
```

**Solution:**
- Convert pandas NaN to None: `message = {k: (None if pd.isna(v) else v) for k, v in message.items()}`
- Or use empty string `" "` for new customers

---

## Schema Versioning

### Current Version: 1.0.0

**Version History:**

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-10-11 | Initial schema release for Mini Project 2 |

### Future Enhancements

Planned for future versions:
- [ ] Optional fields for prediction confidence scores
- [ ] Nested objects for service bundles
- [ ] Support for multiple timestamp formats
- [ ] Backward compatibility with v1.0.0

---

## References

- **JSON Schema Specification:** https://json-schema.org/draft-07/schema
- **JSON Schema Validator Docs:** https://python-jsonschema.readthedocs.io/
- **Mini Project 2 PDF:** `Mini-Project-1-Productionising-Telco-Churn-Prediction.pdf`
- **Producer Documentation:** `README.md` Section 7
- **Source Code:** `src/streaming/schema_validator.py`

---

**Document Status:** ✅ Complete  
**Last Reviewed:** October 11, 2025  
**Maintained By:** Telco Churn MLOps Team
