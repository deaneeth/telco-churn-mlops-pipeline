from flask import Flask, request, jsonify
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from inference.predict import load_model, predict_from_dict

# Initialize Flask app
app = Flask(__name__)

# Load the model once when the app starts
# Use absolute path to ensure model is found regardless of working directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts/models/sklearn_pipeline.joblib"
model = None

def initialize_model():
    """Initialize the model when the app starts"""
    global model
    try:
        model = load_model(str(MODEL_PATH))
        print("[OK] Flask app initialized with model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load model on startup: {e}")
        model = None

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return "pong"

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "message": "The ML model failed to load on startup"
            }), 500
        
        # Get JSON data from request
        if not request.is_json:
            return jsonify({
                "error": "Invalid request",
                "message": "Request must be JSON"
            }), 400
        
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({
                "error": "Empty request",
                "message": "Request body cannot be empty"
            }), 400
        
        # Make prediction
        result = predict_from_dict(model, input_data)
        
        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "probability": result["probability"],
            "message": "Prediction successful"
        })
        
    except KeyError as e:
        return jsonify({
            "error": "Missing features",
            "message": str(e)
        }), 400
        
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    # Initialize model before starting the server
    initialize_model()
    
    # Try to use waitress (production server) if available, otherwise fall back to Flask dev server
    try:
        from waitress import serve
        print("[OK] Starting production server with Waitress on http://0.0.0.0:5000")
        print("   Press CTRL+C to quit\n")
        serve(app, host='0.0.0.0', port=5000, threads=4)
    except ImportError:
        print("[WARN] Waitress not installed. Using Flask development server...")
        print("   For production, install waitress: pip install waitress\n")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)