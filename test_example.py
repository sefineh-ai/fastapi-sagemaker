#!/usr/bin/env python3
"""
Example script to test the FastAPI SageMaker Hugging Face integration
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint with question-answering data"""
    print("\nTesting single prediction...")
    
    # Example question-answering data matching the Hugging Face model format
    test_data = {
        "data": {
            "question": "Which name is also used to describe the Amazon rainforest in English?",
            "context": "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America."
        },
        "request_id": "test-qa-001",
        "metadata": {
            "source": "test-script",
            "model_type": "question-answering",
            "timestamp": time.time()
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Single prediction failed: {e}")
        return False

def test_single_prediction_alternative_format():
    """Test single prediction with alternative input format"""
    print("\nTesting single prediction (alternative format)...")
    
    # Alternative format with 'inputs' wrapper
    test_data = {
        "data": {
            "inputs": {
                "question": "What is the capital of France?",
                "context": "Paris is the capital and most populous city of France. It is known for its art, fashion, gastronomy and culture."
            }
        },
        "request_id": "test-qa-002"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Alternative format prediction failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint with question-answering data"""
    print("\nTesting batch prediction...")
    
    # Example batch data with multiple question-answering pairs
    batch_data = [
        {
            "data": {
                "question": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
            },
            "request_id": "batch-qa-001"
        },
        {
            "data": {
                "question": "Where is the Eiffel Tower located?",
                "context": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889."
            },
            "request_id": "batch-qa-002"
        },
        {
            "data": {
                "question": "What is the largest planet in our solar system?",
                "context": "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than twice that of Saturn."
            },
            "request_id": "batch-qa-003"
        }
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Root endpoint failed: {e}")
        return False

def test_documentation_endpoints():
    """Test API documentation endpoints"""
    print("\nTesting documentation endpoints...")
    
    endpoints = [
        ("/docs", "Swagger UI"),
        ("/redoc", "ReDoc")
    ]
    
    results = []
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            status = "✅ Available" if response.status_code == 200 else "❌ Not available"
            print(f"{name} ({endpoint}): {status}")
            results.append(response.status_code == 200)
        except Exception as e:
            print(f"{name} ({endpoint}): ❌ Error - {e}")
            results.append(False)
    
    return all(results)

def main():
    """Run all tests"""
    print("=" * 60)
    print("FastAPI SageMaker Hugging Face Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Single Prediction (Alternative Format)", test_single_prediction_alternative_format),
        ("Batch Prediction", test_batch_prediction),
        ("Documentation Endpoints", test_documentation_endpoints),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    print("""
Example curl commands for testing:

1. Single Question-Answering:
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "data": {
         "question": "What is the capital of France?",
         "context": "Paris is the capital and most populous city of France."
       },
       "request_id": "test-001"
     }'

2. Batch Question-Answering:
curl -X POST "http://localhost:8000/predict/batch" \\
     -H "Content-Type: application/json" \\
     -d '[
       {
         "data": {
           "question": "What is machine learning?",
           "context": "Machine learning is a subset of artificial intelligence."
         },
         "request_id": "batch-001"
       }
     ]'

3. Check Model Info:
curl -X GET "http://localhost:8000/model/info"

4. Health Check:
curl -X GET "http://localhost:8000/health"
    """)

if __name__ == "__main__":
    main() 