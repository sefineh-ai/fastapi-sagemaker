import boto3
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SageMakerClient:
    """
    Client for interacting with SageMaker deployed Hugging Face models
    """
    
    def __init__(self):
        """Initialize the SageMaker client"""
        self.endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME")
        self.region = os.getenv("AWS_REGION", "eu-north-1")
        self.model_name = os.getenv("MODEL_NAME", "distilbert-base-uncased-distilled-squad")
        
        # Initialize AWS clients
        self.sagemaker_runtime = None
        self.sagemaker = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS SageMaker clients"""
        try:
            # Initialize SageMaker Runtime client for predictions
            self.sagemaker_runtime = boto3.client(
                'sagemaker-runtime',
                region_name=self.region
            )
            
            # Initialize SageMaker client for endpoint management
            self.sagemaker = boto3.client(
                'sagemaker',
                region_name=self.region
            )
            
            logger.info(f"SageMaker clients initialized for region: {self.region}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker clients: {str(e)}")
            raise
    
    def is_configured(self) -> bool:
        """Check if the client is properly configured"""
        return (
            self.endpoint_name is not None and
            self.sagemaker_runtime is not None and
            self.sagemaker is not None
        )
    
    def _convert_to_dict(self, obj):
        """Convert Pydantic models or other objects to dictionaries"""
        if isinstance(obj, BaseModel):
            return obj.dict()
        elif isinstance(obj, dict):
            return {k: self._convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_dict(item) for item in obj]
        else:
            return obj
    
    def prepare_input(self, data: Union[Dict[str, Any], List[Any], str, float, int]) -> str:
        """
        Prepare input data for Hugging Face SageMaker endpoint
        
        Args:
            data: Input data - can be a dict with 'question' and 'context', 
                  or a dict with 'inputs' containing question-answering data
            
        Returns:
            JSON string ready for SageMaker endpoint
        """
        try:
            # Convert Pydantic models to dictionaries
            data = self._convert_to_dict(data)
            
            # Handle different input formats
            if isinstance(data, dict):
                # If data already has 'inputs' key, use it directly
                if 'inputs' in data:
                    input_data = data
                # If data has 'question' and 'context' keys, wrap in 'inputs'
                elif 'question' in data and 'context' in data:
                    input_data = {
                        "inputs": {
                            "question": data['question'],
                            "context": data['context']
                        }
                    }
                # If data has other format, assume it's the inputs directly
                else:
                    input_data = {"inputs": data}
            else:
                # For other data types, wrap in inputs
                input_data = {"inputs": data}
            
            # Convert to JSON string
            json_data = json.dumps(input_data)
            logger.debug(f"Prepared input data: {json_data[:200]}...")
            
            return json_data
            
        except Exception as e:
            logger.error(f"Failed to prepare input data: {str(e)}")
            raise ValueError(f"Failed to prepare input data: {str(e)}")
    
    async def predict(self, input_data: str) -> Dict[str, Any]:
        """
        Make a prediction using the SageMaker Hugging Face endpoint
        
        Args:
            input_data: JSON string of input data
            
        Returns:
            Prediction response from SageMaker
        """
        try:
            start_time = time.time()
            
            # Make prediction request
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=input_data
            )
            
            # Read response body
            response_body = response['Body'].read().decode('utf-8')
            
            # Parse JSON response
            prediction = json.loads(response_body)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            logger.info(f"Prediction completed in {processing_time:.2f}ms")
            
            return {
                'prediction': prediction,
                'processing_time_ms': processing_time,
                'response_metadata': response.get('ResponseMetadata', {})
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def process_response(self, prediction_response: Dict[str, Any]) -> Any:
        """
        Process the prediction response from Hugging Face SageMaker endpoint
        
        Args:
            prediction_response: Raw response from SageMaker
            
        Returns:
            Processed prediction result
        """
        try:
            prediction = prediction_response.get('prediction')
            
            # For Hugging Face question-answering models, the response typically contains:
            # - answer: the predicted answer text
            # - score: confidence score
            # - start: start position in context
            # - end: end position in context
            
            # If prediction is a list (common for Hugging Face models), take the first result
            if isinstance(prediction, list) and len(prediction) > 0:
                prediction = prediction[0]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to process prediction response: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the deployed SageMaker model
        
        Returns:
            Dictionary containing model information
        """
        try:
            if not self.endpoint_name:
                return {
                    "error": "Endpoint name not configured",
                    "model_name": self.model_name,
                    "region": self.region
                }
            
            # Get endpoint details
            response = self.sagemaker.describe_endpoint(
                EndpointName=self.endpoint_name
            )
            
            # Get endpoint configuration
            config_response = self.sagemaker.describe_endpoint_config(
                EndpointConfigName=response['EndpointConfigName']
            )
            
            # Extract instance type from production variants
            instance_type = None
            if config_response['ProductionVariants']:
                instance_type = config_response['ProductionVariants'][0].get('InstanceType')
            
            return {
                "model_name": self.model_name,
                "model_type": "Hugging Face Question-Answering",
                "model_id": "distilbert-base-uncased-distilled-squad",
                "endpoint_name": self.endpoint_name,
                "region": self.region,
                "status": response['EndpointStatus'],
                "instance_type": instance_type,
                "created_at": response.get('CreationTime'),
                "last_modified": response.get('LastModifiedTime'),
                "endpoint_arn": response.get('EndpointArn')
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {
                "error": str(e),
                "model_name": self.model_name,
                "region": self.region
            }
    
    def check_endpoint_status(self) -> str:
        """
        Check the current status of the SageMaker endpoint
        
        Returns:
            Endpoint status string
        """
        try:
            if not self.endpoint_name:
                return "Not configured"
            
            response = self.sagemaker.describe_endpoint(
                EndpointName=self.endpoint_name
            )
            
            return response['EndpointStatus']
            
        except Exception as e:
            logger.error(f"Failed to check endpoint status: {str(e)}")
            return f"Error: {str(e)}" 