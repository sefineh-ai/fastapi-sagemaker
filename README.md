# FastAPI SageMaker Hugging Face Integration

A FastAPI application that integrates with AWS SageMaker deployed Hugging Face models for question-answering tasks.

## Features

- 🚀 FastAPI-based REST API
- 🔗 SageMaker Hugging Face endpoint integration
- 🤖 Question-Answering model support (DistilBERT)
- 📊 Single and batch prediction endpoints
- 🔍 Model information and health monitoring
- 🛡️ Error handling and logging
- 📝 Request/response validation with Pydantic
- 🌐 CORS support for web applications

## Model Details

This integration is specifically configured for:
- **Model**: `distilbert-base-uncased-distilled-squad`
- **Task**: Question-Answering
- **Framework**: Hugging Face Transformers
- **Deployment**: SageMaker Inference Endpoint
- **Region**: `eu-north-1`
- **Endpoint**: `huggingface-pytorch-inference-2025-06-27-04-56-19-392`

## Project Structure

```
fastapi_sagemaker/
├── main.py                 # FastAPI application entry point
├── models.py              # Pydantic models for request/response validation
├── sagemaker_client.py    # SageMaker Hugging Face integration client
├── requirements.txt       # Python dependencies
├── env.example           # Environment variables template
├── test_example.py       # Test script with question-answering examples
└── README.md             # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and configure your SageMaker settings:

```bash
cp env.example .env
```

Edit `.env` with your SageMaker endpoint details and AWS credentials:

```env
SAGEMAKER_ENDPOINT_NAME=huggingface-pytorch-inference-2025-06-27-04-56-19-392
AWS_REGION=eu-north-1
MODEL_NAME=distilbert-base-uncased-distilled-squad

# AWS Credentials
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Hugging Face Model Configuration
MODEL_TYPE=question-answering
HF_MODEL_ID=distilbert-base-uncased-distilled-squad
HF_TASK=question-answering

# SageMaker Session Details
SAGEMAKER_BUCKET=sagemaker-eu-north-1-740269538003
SAGEMAKER_ROLE_ARN=arn:aws:iam::740269538003:role/sagemaker-full-access-role
```

### 3. AWS Configuration

Ensure you have AWS credentials configured. You can either:

- Use AWS CLI: `aws configure`
- Set environment variables in `.env` file (recommended)
- Use IAM roles (if running on EC2 or ECS)

**Note**: Make sure your AWS credentials are configured for the `eu-north-1` region.

### 4. Run the Application

```bash
# Activate virtual environment
source .venv/bin/activate

# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python main.py
```

**Note**: If you see `ERROR: [Errno 98] Address already in use`, it means the server is already running on port 8000. You can either:
- Stop the existing server (Ctrl+C in the terminal where it's running)
- Use a different port: `uvicorn main:app --reload --host 0.0.0.0 --port 8001`

## API Endpoints

### Health Check
- **GET** `/health` - Check application and SageMaker connection status

### Predictions
- **POST** `/predict` - Single question-answering prediction
- **POST** `/predict/batch` - Batch question-answering predictions
- **GET** `/model/info` - Get SageMaker model information

### Documentation
- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/redoc` - Alternative API documentation

## Usage Examples

### Single Question-Answering

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "question": "Which name is also used to describe the Amazon rainforest in English?",
         "context": "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America."
       },
       "request_id": "req-123"
     }'
```

**Example Response:**
```json
{
    "prediction": {
        "answer": "Amazonia",
        "score": 0.9540701508522034,
        "start": 201,
        "end": 209
    },
    "model_name": "distilbert-base-uncased-distilled-squad",
    "request_id": "req-123",
    "error": null,
    "timestamp": "2025-06-27T05:40:08.438513",
    "processing_time_ms": null
}
```

### Batch Question-Answering

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '[
       {
         "data": {
           "question": "What is machine learning?",
           "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
         },
         "request_id": "req-1"
       },
       {
         "data": {
           "question": "Where is the Eiffel Tower located?",
           "context": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France."
         },
         "request_id": "req-2"
       }
     ]'
```

### Check Model Information

```bash
curl -X GET "http://localhost:8000/model/info"
```

## Request/Response Formats

### Question-Answering Request
```json
{
  "data": {
    "question": "Your question here?",
    "context": "The context passage where the answer can be found."
  },
  "request_id": "optional-request-id",
  "metadata": "optional-metadata"
}
```

### Question-Answering Response
```json
{
  "prediction": {
    "answer": "The predicted answer text",
    "score": 0.95,
    "start": 10,
    "end": 25
  },
  "model_name": "distilbert-base-uncased-distilled-squad",
  "request_id": "req-123",
  "error": null,
  "timestamp": "2024-01-01T00:00:00",
  "processing_time_ms": 150.5
}
```

## Input Format Flexibility

The API supports multiple input formats:

### Format 1: Direct question and context
```json
{
  "data": {
    "question": "What is the capital of France?",
    "context": "Paris is the capital of France."
  }
}
```

### Format 2: With inputs wrapper
```json
{
  "data": {
    "inputs": {
      "question": "What is the capital of France?",
      "context": "Paris is the capital of France."
    }
  }
}
```

## Model Response Details

The Hugging Face question-answering model returns:
- **answer**: The extracted answer text from the context
- **score**: Confidence score (0-1) of the prediction
- **start**: Starting character position of the answer in the context
- **end**: Ending character position of the answer in the context

## Testing

Run the included test script to verify the integration:

```bash
python test_example.py
```

This will test all endpoints with sample question-answering data.

## Customization

### Model-Specific Processing

The `sagemaker_client.py` is already configured for Hugging Face question-answering models, but you can customize:

1. **Input Preparation**: Modify the `prepare_input()` method if your model requires different formatting
2. **Response Processing**: Update the `process_response()` method to handle different output formats

### Environment Variables

Additional configuration options:

- `MODEL_TYPE`: Specify model type (default: question-answering)
- `HF_MODEL_ID`: Hugging Face model identifier
- `HF_TASK`: Hugging Face task type
- `SAGEMAKER_BUCKET`: Your SageMaker bucket (sagemaker-eu-north-1-740269538003)
- `SAGEMAKER_ROLE_ARN`: Your SageMaker execution role ARN

## Error Handling

The application includes comprehensive error handling:

- Invalid input data validation
- SageMaker endpoint connection errors
- AWS credential issues
- Model prediction failures
- Question-answering format validation

All errors are logged and returned with appropriate HTTP status codes.

## Monitoring

- Application logs are written to stdout
- Request/response logging for debugging
- Processing time tracking
- SageMaker endpoint status monitoring
- Question-answering confidence scores

## Security Considerations

- Use HTTPS in production
- Implement authentication/authorization as needed
- Secure AWS credentials
- Consider rate limiting for production use
- Validate input data to prevent injection attacks

## Troubleshooting

### Common Issues

1. **AWS Credentials**: Ensure AWS credentials are properly configured for `eu-north-1` region
2. **Endpoint Name**: Verify the SageMaker endpoint name matches your deployment
3. **Region**: Make sure the AWS region is set to `eu-north-1`
4. **Permissions**: Ensure your AWS user/role has SageMaker permissions
5. **Model Format**: Ensure input data follows the question-answering format
6. **Port Already in Use**: If you get `ERROR: [Errno 98] Address already in use`:
   - Stop the existing server (Ctrl+C)
   - Or use a different port: `uvicorn main:app --reload --host 0.0.0.0 --port 8001`

### Debug Mode

Enable debug logging by setting the log level:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Model Deployment Verification

To verify your SageMaker endpoint is working:

```python
import boto3

runtime = boto3.client('sagemaker-runtime', region_name='eu-north-1')
response = runtime.invoke_endpoint(
    EndpointName='huggingface-pytorch-inference-2025-06-27-04-56-19-392',
    ContentType='application/json',
    Body='{"inputs": {"question": "test", "context": "test"}}'
)
print(response['Body'].read().decode())
```

### Region-Specific Configuration

Since your model is deployed in `eu-north-1`, make sure:

1. Your AWS CLI is configured for the correct region:
   ```bash
   aws configure set region eu-north-1
   ```

2. Your environment variables are set correctly:
   ```bash
   export AWS_REGION=eu-north-1
   ```

### Virtual Environment Issues

If you encounter import errors (e.g., "Import 'pydantic' could not be resolved"):

1. Make sure your virtual environment is activated:
   ```bash
   source .venv/bin/activate
   ```

2. Verify packages are installed:
   ```bash
   pip show pydantic
   ```

3. In VSCode, select the correct Python interpreter:
   - Press `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose the interpreter from your `.venv` directory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. 