# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Copy the trained model
COPY artifacts/models/sklearn_pipeline.joblib artifacts/models/sklearn_pipeline.joblib

# Expose port 5000
EXPOSE 5000

# Run the Flask API
CMD ["python", "src/api/app.py"]