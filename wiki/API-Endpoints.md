# 🌐 API Endpoints Reference

Complete documentation for the Flask REST API endpoints.

---

## Base URL

```
Local Development: http://localhost:5000
Production: http://your-domain.com:5000
Docker Container: http://0.0.0.0:5000
```

---

## Authentication

Currently **no authentication** required (suitable for internal/demo use).

For production deployment, consider adding:
- API key authentication
- OAuth 2.0
- JWT tokens

---

## Endpoints Overview

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/ping` | GET | Health check | ❌ No |
| `/predict` | POST | Churn prediction | ❌ No |

---

## 1. Health Check Endpoint

### `GET /ping`

**Purpose**: Verify API server is running and healthy.

**Request**:
```bash
# cURL
curl http://localhost:5000/ping

# PowerShell
Invoke-WebRequest -Uri "http://localhost:5000/ping" -Method GET

# Python
import requests
response = requests.get("http://localhost:5000/ping")
print(response.text)
```

**Response**:
```
pong
```

**Status Codes**:
- `200 OK` - Server is healthy
- `500 Internal Server Error` - Server error

**Response Headers**:
```
Content-Type: text/html; charset=utf-8
Content-Length: 4
```

---

## 2. Prediction Endpoint

### `POST /predict`

**Purpose**: Predict customer churn probability and classification.

**Request Format**:
```json
POST /predict
Content-Type: application/json

{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 65.50,
  "TotalCharges": 786.0
}
```

**Field Specifications**:

| Field | Type | Required | Valid Values | Example |
|-------|------|----------|--------------|---------|
| `gender` | string | ✅ Yes | "Male", "Female" | "Female" |
| `SeniorCitizen` | integer | ✅ Yes | 0, 1 | 0 |
| `Partner` | string | ✅ Yes | "Yes", "No" | "Yes" |
| `Dependents` | string | ✅ Yes | "Yes", "No" | "No" |
| `tenure` | integer | ✅ Yes | 0-72 | 12 |
| `PhoneService` | string | ✅ Yes | "Yes", "No" | "Yes" |
| `MultipleLines` | string | ✅ Yes | "Yes", "No", "No phone service" | "No" |
| `InternetService` | string | ✅ Yes | "DSL", "Fiber optic", "No" | "DSL" |
| `OnlineSecurity` | string | ✅ Yes | "Yes", "No", "No internet service" | "Yes" |
| `OnlineBackup` | string | ✅ Yes | "Yes", "No", "No internet service" | "No" |
| `DeviceProtection` | string | ✅ Yes | "Yes", "No", "No internet service" | "Yes" |
| `TechSupport` | string | ✅ Yes | "Yes", "No", "No internet service" | "No" |
| `StreamingTV` | string | ✅ Yes | "Yes", "No", "No internet service" | "Yes" |
| `StreamingMovies` | string | ✅ Yes | "Yes", "No", "No internet service" | "No" |
| `Contract` | string | ✅ Yes | "Month-to-month", "One year", "Two year" | "Month-to-month" |
| `PaperlessBilling` | string | ✅ Yes | "Yes", "No" | "Yes" |
| `PaymentMethod` | string | ✅ Yes | "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)" | "Electronic check" |
| `MonthlyCharges` | float | ✅ Yes | 0.0 - 150.0 | 65.50 |
| `TotalCharges` | float | ✅ Yes | 0.0 - 10000.0 | 786.0 |

**Success Response**:
```json
{
  "success": true,
  "message": "Prediction successful",
  "prediction": 0,
  "probability": 0.2925608692250003
}
```

**Response Fields**:
- `success` (boolean): Whether prediction succeeded
- `message` (string): Human-readable status message
- `prediction` (integer): Binary prediction (0 = No Churn, 1 = Churn)
- `probability` (float): Probability of churn (0.0 - 1.0)

**Interpretation**:
```python
if prediction == 0:
    result = "Customer will NOT churn"
    risk_level = "Low" if probability < 0.3 else "Medium"
else:
    result = "Customer WILL churn"
    risk_level = "High" if probability > 0.7 else "Medium-High"
```

**Error Response** (Missing Fields):
```json
{
  "success": false,
  "error": "Missing required fields: tenure, MonthlyCharges"
}
```

**Error Response** (Invalid Data):
```json
{
  "success": false,
  "error": "Invalid value for field 'gender': must be 'Male' or 'Female'"
}
```

**Status Codes**:
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid input data
- `500 Internal Server Error` - Server error

---

## Usage Examples

### Python (requests)

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# Customer data
customer = {
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 255.0
}

# Make request
response = requests.post(url, json=customer)

# Parse response
result = response.json()
print(f"Prediction: {'Churn' if result['prediction'] == 1 else 'No Churn'}")
print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {'High' if result['probability'] > 0.7 else 'Medium' if result['probability'] > 0.3 else 'Low'}")
```

### PowerShell

```powershell
# Customer data
$customer = @{
    gender = "Female"
    SeniorCitizen = 0
    Partner = "Yes"
    Dependents = "Yes"
    tenure = 48
    PhoneService = "Yes"
    MultipleLines = "Yes"
    InternetService = "Fiber optic"
    OnlineSecurity = "Yes"
    OnlineBackup = "Yes"
    DeviceProtection = "Yes"
    TechSupport = "Yes"
    StreamingTV = "Yes"
    StreamingMovies = "Yes"
    Contract = "Two year"
    PaperlessBilling = "No"
    PaymentMethod = "Bank transfer (automatic)"
    MonthlyCharges = 105.50
    TotalCharges = 5064.0
} | ConvertTo-Json

# Make request
$response = Invoke-WebRequest -Uri "http://localhost:5000/predict" `
    -Method POST `
    -Body $customer `
    -ContentType "application/json"

# Parse response
$result = $response.Content | ConvertFrom-Json
Write-Host "Prediction: $(if($result.prediction -eq 1){'Churn'}else{'No Churn'})"
Write-Host "Probability: $($result.probability * 100)%"
```

### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.50,
    "TotalCharges": 786.0
  }'
```

### JavaScript (fetch)

```javascript
const customer = {
  gender: "Male",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "Yes",
  tenure: 24,
  PhoneService: "Yes",
  MultipleLines: "Yes",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "Yes",
  DeviceProtection: "Yes",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "Yes",
  Contract: "One year",
  PaperlessBilling: "Yes",
  PaymentMethod: "Credit card (automatic)",
  MonthlyCharges: 89.99,
  TotalCharges: 2159.76
};

fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(customer)
})
  .then(response => response.json())
  .then(data => {
    console.log('Prediction:', data.prediction === 1 ? 'Churn' : 'No Churn');
    console.log('Probability:', (data.probability * 100).toFixed(2) + '%');
  })
  .catch(error => console.error('Error:', error));
```

---

## Batch Predictions

For multiple customers, make individual requests in a loop or use async processing:

```python
import requests
import asyncio
import aiohttp

customers = [customer1, customer2, customer3, ...]  # List of customer dicts

async def predict_async(session, customer):
    async with session.post('http://localhost:5000/predict', json=customer) as response:
        return await response.json()

async def batch_predict(customers):
    async with aiohttp.ClientSession() as session:
        tasks = [predict_async(session, customer) for customer in customers]
        results = await asyncio.gather(*tasks)
        return results

# Run batch predictions
results = asyncio.run(batch_predict(customers))
```

---

## Response Time

**Typical Performance**:
- `/ping`: < 5ms
- `/predict`: < 50ms (single prediction)
- Throughput: ~1000 requests/second (single instance)

**Factors Affecting Performance**:
- Model complexity (GradientBoosting: fast)
- Server resources (CPU, RAM)
- Network latency
- Request payload size

---

## Error Handling

**Common Errors**:

1. **400 Bad Request - Missing Fields**
   ```json
   {"success": false, "error": "Missing required field: tenure"}
   ```
   **Solution**: Ensure all 19 required fields are present

2. **400 Bad Request - Invalid Value**
   ```json
   {"success": false, "error": "Invalid Contract value"}
   ```
   **Solution**: Check field value against valid options

3. **500 Internal Server Error**
   ```json
   {"success": false, "error": "Model prediction failed"}
   ```
   **Solution**: Check server logs, ensure model file exists

---

## Rate Limiting

Currently **no rate limiting** implemented.

For production, consider:
- 100 requests/minute per IP
- 1000 requests/hour per API key
- Use Redis for distributed rate limiting

---

## Testing the API

### Automated Tests

```bash
# Run API tests
python test_api_validation.py

# Expected output:
# ✅ /ping endpoint working
# ✅ /predict endpoint working
```

### Manual Testing Tools

- **Postman**: Import collection from `docs/postman_collection.json`
- **Insomnia**: Use REST client
- **curl**: Command-line testing
- **Browser**: For GET requests only

---

## Next Steps

- **[Flask API](Flask-API)** - API implementation details
- **[Docker Deployment](Docker-Deployment)** - Deploy API in container
- **[Production Deployment](Production-Deployment)** - Production setup

---

[← Back to Home](Home) | [Next: Flask API Implementation →](Flask-API)
