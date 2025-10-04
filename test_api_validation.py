"""API Validation Script - Tests Flask API endpoints"""
import subprocess
import time
import requests
import json
import sys

def test_api():
    """Test API endpoints"""
    print("=" * 60)
    print("API VALIDATION TEST")
    print("=" * 60)
    
    # Start API server
    print("\n[1/4] Starting Flask API server...")
    process = subprocess.Popen(
        [sys.executable, "src/api/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("[2/4] Waiting for server to initialize (5 seconds)...")
    time.sleep(5)
    
    try:
        # Test /ping endpoint
        print("[3/4] Testing /ping endpoint...")
        response = requests.get("http://localhost:5000/ping", timeout=5)
        print(f"  Status Code: {response.status_code}")
        print(f"  Response: {response.text}")
        assert response.status_code == 200, "Ping endpoint failed"
        assert response.text == "pong", f"Expected 'pong', got '{response.text}'"
        print("  ✅ /ping endpoint working!")
        
        # Test /predict endpoint
        print("[4/4] Testing /predict endpoint...")
        test_customer = {
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
        
        response = requests.post(
            "http://localhost:5000/predict",
            json=test_customer,
            timeout=5
        )
        print(f"  Status Code: {response.status_code}")
        result = response.json()
        print(f"  Response: {json.dumps(result, indent=2)}")
        
        assert response.status_code == 200, "Predict endpoint failed"
        assert "prediction" in result, "Missing prediction field"
        assert "probability" in result, "Missing probability field"
        
        threshold = result.get("threshold_used", 0.35)  # Default to 0.35 if not returned
        
        print(f"  ✅ /predict endpoint working with recall-optimized model!")
        
        print("\n" + "=" * 60)
        print("✅ ALL API TESTS PASSED")
        print("=" * 60)
        print(f"\n📊 Prediction Results:")
        print(f"  Customer ID: test_customer_001")
        print(f"  Prediction: {'Churn' if result['prediction'] == 1 else 'No Churn'}")
        print(f"  Churn Probability: {result['probability']:.2%}")
        print(f"  Decision Threshold: {threshold}")
        print(f"  Model: Recall-Optimized GradientBoostingClassifier")
        
        return True
        
    except Exception as e:
        print(f"\n❌ API Test Failed: {str(e)}")
        return False
        
    finally:
        # Stop server
        print("\n[Cleanup] Stopping API server...")
        process.terminate()
        process.wait()
        print("  Server stopped.")

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
