from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Any, Dict, List, Optional
import json
import logging

from sagemaker_client import SageMakerClient
from models import PredictionRequest, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FastAPI SageMaker Integration",
    description="A FastAPI application that integrates with SageMaker deployed models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SageMaker client
sagemaker_client = SageMakerClient()

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "FastAPI SageMaker Integration API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if SageMaker client is properly configured
        is_configured = sagemaker_client.is_configured()
        return {
            "status": "healthy",
            "sagemaker_configured": is_configured,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction using the SageMaker model
    """
    try:
        logger.info(f"Received prediction request: {request}")
        
        # Convert request to format expected by SageMaker
        input_data = sagemaker_client.prepare_input(request.data)
        
        # Make prediction
        prediction = await sagemaker_client.predict(input_data)
        
        # Process the response
        response_data = sagemaker_client.process_response(prediction)
        
        return PredictionResponse(
            prediction=response_data,
            model_name=sagemaker_client.model_name,
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[PredictionRequest]):
    """
    Make batch predictions using the SageMaker model
    """
    try:
        logger.info(f"Received batch prediction request with {len(requests)} items")
        
        results = []
        for request in requests:
            try:
                # Convert request to format expected by SageMaker
                input_data = sagemaker_client.prepare_input(request.data)
                
                # Make prediction
                prediction = await sagemaker_client.predict(input_data)
                
                # Process the response
                response_data = sagemaker_client.process_response(prediction)
                
                results.append(PredictionResponse(
                    prediction=response_data,
                    model_name=sagemaker_client.model_name,
                    request_id=request.request_id
                ))
                
            except Exception as e:
                logger.error(f"Failed to process request {request.request_id}: {str(e)}")
                results.append(PredictionResponse(
                    prediction=None,
                    model_name=sagemaker_client.model_name,
                    request_id=request.request_id,
                    error=str(e)
                ))
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """
    Get information about the deployed SageMaker model
    """
    try:
        info = sagemaker_client.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 