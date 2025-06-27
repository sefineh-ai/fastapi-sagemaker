from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Any, Dict, List, Optional, Union
import uuid
from datetime import datetime

class QuestionAnsweringInput(BaseModel):
    """Input model for question-answering tasks"""
    question: str = Field(..., description="The question to be answered", min_length=1)
    context: str = Field(..., description="The context/passage to search for the answer", min_length=1)

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    data: Union[QuestionAnsweringInput, Dict[str, Any], List[Any], str, float, int] = Field(
        ..., 
        description="Input data for prediction. For question-answering, use question and context fields."
    )
    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the request"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the request"
    )
    
    @field_validator('data')
    @classmethod
    def validate_data(cls, v):
        """Validate input data format"""
        if isinstance(v, dict):
            # Check if it's a question-answering format
            if 'question' in v and 'context' in v:
                return v
            # Check if it has inputs format
            elif 'inputs' in v:
                return v
            # Otherwise, assume it's valid
            return v
        return v

class QuestionAnsweringResponse(BaseModel):
    """Response model for question-answering predictions"""
    answer: str = Field(..., description="The predicted answer text")
    score: float = Field(..., description="Confidence score of the prediction")
    start: int = Field(..., description="Start position of the answer in the context")
    end: int = Field(..., description="End position of the answer in the context")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: Optional[Union[QuestionAnsweringResponse, Dict[str, Any], List[Any]]] = Field(
        default=None,
        description="Model prediction result. For question-answering, contains answer, score, start, and end positions."
    )
    model_name: str = Field(
        ..., 
        description="Name of the model used for prediction"
    )
    request_id: str = Field(
        ..., 
        description="Unique identifier for the request"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if prediction failed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the prediction"
    )
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken to process the request in milliseconds"
    )

class ModelInfo(BaseModel):
    """Model information response"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str = Field(..., description="Name of the deployed model")
    model_type: str = Field(..., description="Type of the model (e.g., Hugging Face Question-Answering)")
    model_id: str = Field(..., description="Hugging Face model ID")
    endpoint_name: str = Field(..., description="SageMaker endpoint name")
    region: str = Field(..., description="AWS region where the model is deployed")
    status: str = Field(..., description="Current status of the endpoint")
    instance_type: Optional[str] = Field(default=None, description="Instance type used for deployment")
    created_at: Optional[datetime] = Field(default=None, description="When the endpoint was created")
    last_modified: Optional[datetime] = Field(default=None, description="When the endpoint was last modified")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    sagemaker_configured: bool = Field(..., description="Whether SageMaker is properly configured")
    timestamp: str = Field(..., description="Timestamp of the health check") 