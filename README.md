# 🚀 FastAPI SageMaker Integration

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![AWS SageMaker](https://img.shields.io/badge/AWS%20SageMaker-ML%20Hosting-FF9900?style=for-the-badge&logo=amazon-aws)](https://aws.amazon.com/sagemaker/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Boto3](https://img.shields.io/badge/Boto3-1.34.0-orange.svg?style=for-the-badge)](https://boto3.amazonaws.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.5.0-blue.svg?style=for-the-badge)](https://pydantic.dev/)

> **Enterprise-grade FastAPI application** that seamlessly integrates with AWS SageMaker deployed Hugging Face models for advanced question-answering tasks. Built with modern Python technologies and cloud-native architecture.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Usage Examples](#-usage-examples)
- [Model Details](#-model-details)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

FastAPI SageMaker Integration is a production-ready solution that bridges the gap between modern web APIs and AWS SageMaker's powerful machine learning capabilities. This application provides a robust, scalable interface for deploying and consuming Hugging Face models through SageMaker endpoints.

### Key Benefits

- **🔗 Seamless Integration**: Direct connection to AWS SageMaker endpoints
- **⚡ High Performance**: FastAPI's async capabilities for optimal throughput
- **🤖 AI-Powered**: Advanced question-answering with DistilBERT
- **🔐 Enterprise Ready**: Production-grade security and monitoring
- **📊 Batch Processing**: Support for both single and batch predictions
- **🛡️ Robust Error Handling**: Comprehensive error management and logging

## 🚀 Features

### Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **🔗 SageMaker Integration** | Direct connection to AWS SageMaker endpoints | ✅ Implemented |
| **🤖 Question-Answering** | Advanced QA with DistilBERT model | ✅ Implemented |
| **📊 Batch Predictions** | Efficient batch processing capabilities | ✅ Implemented |
| **🔍 Model Information** | Real-time model status and metadata | ✅ Implemented |
| **🛡️ Input Validation** | Pydantic models for request/response validation | ✅ Implemented |
| **🌐 CORS Support** | Cross-origin resource sharing for web apps | ✅ Implemented |

### Technical Features

| Feature | Description | Status |
|---------|-------------|--------|
| **⚡ Async Processing** | Non-blocking API operations | ✅ Implemented |
| **📝 Comprehensive Logging** | Detailed request/response logging | ✅ Implemented |
| **🔐 AWS Authentication** | Secure credential management | ✅ Implemented |
| **📊 Health Monitoring** | Endpoint health checks and status | ✅ Implemented |
| **🔄 Error Recovery** | Graceful error handling and recovery | ✅ Implemented |
| **📈 Performance Metrics** | Processing time and performance tracking | ✅ Implemented |

### Planned Features

- [ ] **🔐 Authentication & Authorization** - JWT-based security
- [ ] **📊 Advanced Analytics** - Prediction analytics dashboard
- [ ] **🔄 Model Versioning** - Support for multiple model versions
- [ ] **⚡ Caching Layer** - Redis caching for improved performance
- [ ] **📱 WebSocket Support** - Real-time streaming capabilities
- [ ] **🔒 Rate Limiting** - API rate limiting and throttling

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   FastAPI App   │    │   AWS SageMaker │
│   (Web/Mobile)  │◄──►│   (Backend)     │◄──►│   (ML Endpoint) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HTTP Client   │    │   SageMaker     │    │   Hugging Face  │
│   (REST API)    │    │   Client        │    │   DistilBERT    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Request Processing**
   ```
   Client Request → FastAPI → Input Validation → SageMaker Client
   ```

2. **Model Inference**
   ```
   Validated Input → SageMaker Endpoint → Hugging Face Model → Prediction
   ```

3. **Response Generation**
   ```
   Model Output → Response Processing → Validation → Client Response
   ```

4. **Error Handling**
   ```
   Any Error → Logging → Error Response → Client Notification
   ```

## 🛠️ Tech Stack

### Backend Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **FastAPI** | 0.104.1 | High-performance web framework for building APIs |
| **Uvicorn** | 0.24.0 | ASGI server for production deployment |
| **Boto3** | 1.34.0 | AWS SDK for Python |
| **Pydantic** | 2.5.0 | Data validation and settings management |
| **Python-multipart** | 0.0.6 | File upload support |
| **Requests** | 2.31.0 | HTTP library for API calls |
| **NumPy** | 1.26.0+ | Numerical computing |
| **Pandas** | Latest | Data manipulation |

### AWS Services

| Service | Purpose |
|---------|---------|
| **AWS SageMaker** | Machine learning model hosting and inference |
| **AWS IAM** | Identity and access management |
| **AWS CloudWatch** | Monitoring and logging |
| **AWS S3** | Model artifacts and data storage |

### Infrastructure & DevOps

| Technology | Purpose |
|------------|---------|
| **Docker** | Containerization for consistent deployment |
| **Environment Variables** | Secure configuration management |
| **Git** | Version control and collaboration |
| **AWS CLI** | Command-line interface for AWS |

## 📁 Project Structure

```
fastapi_sagemaker/
├── 📁 aws/                        # AWS configuration files
├── 🐍 main.py                     # FastAPI application entry point
├── 📋 models.py                   # Pydantic models for validation
├── 🔗 sagemaker_client.py         # SageMaker integration client
├── 📋 requirements.txt            # Python dependencies
├── 🔧 env.example                 # Environment variables template
├── 🧪 test_example.py             # Comprehensive test suite
├── 📄 .gitignore                  # Git ignore rules
└── 📖 README.md                   # Project documentation
```

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
- **AWS Account** - [Create AWS Account](https://aws.amazon.com/)
- **AWS CLI** - [Install AWS CLI](https://aws.amazon.com/cli/)
- **Git** - [Download Git](https://git-scm.com/)

### Required AWS Resources

You'll need the following AWS resources:

- **SageMaker Endpoint** - Deployed Hugging Face model
- **IAM Role** - SageMaker execution role with proper permissions
- **AWS Credentials** - Access key and secret key
- **S3 Bucket** - For model artifacts (optional)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fastapi_sagemaker.git
cd fastapi_sagemaker
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env with your configuration
nano .env
```

### 5. Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_DEFAULT_REGION=eu-north-1
```

### 6. Start the Application

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 7. Access the Application

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## ⚙️ Configuration

### Environment Variables

Edit the `.env` file with your specific configuration:

```env
# SageMaker Configuration
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
SAGEMAKER_BUCKET=your-sagemaker-bucket
SAGEMAKER_ROLE_ARN=your-sagemaker-execution-role

# Application Configuration
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Logging Configuration
LOG_LEVEL=INFO

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend-domain.com
```

### SageMaker Endpoint Setup

1. **Deploy Hugging Face Model**
   ```python
   import sagemaker
   from sagemaker.huggingface import HuggingFaceModel
   
   # Create Hugging Face model
   huggingface_model = HuggingFaceModel(
       model_data='s3://your-bucket/model.tar.gz',
       role='your-sagemaker-role',
       transformers_version='4.26.0',
       pytorch_version='1.13.1',
       py_version='py39',
   )
   
   # Deploy model
   predictor = huggingface_model.deploy(
       initial_instance_count=1,
       instance_type='ml.m5.large',
       endpoint_name='huggingface-pytorch-inference-2025-06-27-04-56-19-392'
   )
   ```

2. **Configure IAM Permissions**
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "sagemaker:InvokeEndpoint",
                   "sagemaker:DescribeEndpoint"
               ],
               "Resource": "*"
           }
       ]
   }
   ```

## 📚 API Documentation

### Authentication

Currently, the API uses AWS IAM authentication. In production, implement additional API key authentication.

### Endpoints

#### Health Check

**GET** `/health`

Check application and SageMaker connection status.

**Response:**
```json
{
  "status": "healthy",
  "sagemaker_configured": true,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### Single Prediction

**POST** `/predict`

Make a single question-answering prediction.

**Request Body:**
```json
{
  "data": {
    "question": "Which name is also used to describe the Amazon rainforest in English?",
    "context": "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America."
  },
  "request_id": "req-123"
}
```

**Response:**
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
  "timestamp": "2024-01-01T00:00:00Z",
  "processing_time_ms": 150.5
}
```

#### Batch Predictions

**POST** `/predict/batch`

Make batch question-answering predictions.

**Request Body:**
```json
[
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
]
```

#### Model Information

**GET** `/model/info`

Get information about the deployed SageMaker model.

**Response:**
```json
{
  "model_name": "distilbert-base-uncased-distilled-squad",
  "endpoint_name": "huggingface-pytorch-inference-2025-06-27-04-56-19-392",
  "region": "eu-north-1",
  "status": "InService",
  "instance_type": "ml.m5.large",
  "creation_time": "2024-01-01T00:00:00Z"
}
```

## 💡 Usage Examples

### Python Client

```python
import requests
import json

# Single prediction
def predict_question(question, context):
    url = "http://localhost:8000/predict"
    payload = {
        "data": {
            "question": question,
            "context": context
        },
        "request_id": "python-client-1"
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
result = predict_question(
    "What is the capital of France?",
    "Paris is the capital of France and is known for the Eiffel Tower."
)
print(f"Answer: {result['prediction']['answer']}")
print(f"Confidence: {result['prediction']['score']:.2f}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "question": "What is AI?",
         "context": "Artificial Intelligence (AI) is the simulation of human intelligence in machines."
       },
       "request_id": "curl-test"
     }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '[
       {
         "data": {
           "question": "What is Python?",
           "context": "Python is a high-level programming language known for its simplicity."
         },
         "request_id": "batch-1"
       }
     ]'
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

async function predictQuestion(question, context) {
  try {
    const response = await axios.post('http://localhost:8000/predict', {
      data: {
        question: question,
        context: context
      },
      request_id: 'js-client-1'
    });
    
    return response.data;
  } catch (error) {
    console.error('Prediction failed:', error.response.data);
    throw error;
  }
}

// Example usage
predictQuestion(
  'What is machine learning?',
  'Machine learning is a subset of artificial intelligence.'
)
.then(result => {
  console.log('Answer:', result.prediction.answer);
  console.log('Confidence:', result.prediction.score);
});
```

## 🤖 Model Details

### Hugging Face DistilBERT

This integration is specifically configured for:

| Parameter | Value |
|-----------|-------|
| **Model** | `distilbert-base-uncased-distilled-squad` |
| **Task** | Question-Answering |
| **Framework** | Hugging Face Transformers |
| **Deployment** | SageMaker Inference Endpoint |
| **Region** | `eu-north-1` |
| **Endpoint** | `huggingface-pytorch-inference-2025-06-27-04-56-19-392` |

### Model Capabilities

- **Question-Answering**: Extract answers from context passages
- **Confidence Scoring**: Probability scores for answer confidence
- **Position Tracking**: Start and end positions of answers
- **Context Understanding**: Deep understanding of text context

### Input Format

The model accepts two input formats:

#### Format 1: Direct Input
```json
{
  "data": {
    "question": "Your question here?",
    "context": "The context passage where the answer can be found."
  }
}
```

#### Format 2: Wrapped Input
```json
{
  "data": {
    "inputs": {
      "question": "Your question here?",
      "context": "The context passage where the answer can be found."
    }
  }
}
```

### Output Format

```json
{
  "prediction": {
    "answer": "The extracted answer text",
    "score": 0.95,
    "start": 10,
    "end": 25
  }
}
```

## 🧪 Testing

### Run Test Suite

```bash
# Run comprehensive tests
python test_example.py
```

### Test Individual Endpoints

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "question": "What is the capital of France?",
      "context": "Paris is the capital of France."
    }
  }'

# Test batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "data": {
        "question": "What is AI?",
        "context": "Artificial Intelligence is a field of computer science."
      }
    }
  ]'
```

### Performance Testing

```bash
# Load testing with Apache Bench
ab -n 100 -c 10 -T application/json -p test_data.json http://localhost:8000/predict

# Or use Python for custom load testing
python -c "
import asyncio
import aiohttp
import time

async def load_test():
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        tasks = []
        for i in range(100):
            task = session.post('http://localhost:8000/predict', json={
                'data': {
                    'question': f'Test question {i}?',
                    'context': 'This is a test context for load testing.'
                }
            })
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        print(f'Processed {len(responses)} requests in {end_time - start_time:.2f} seconds')

asyncio.run(load_test())
"
```

## 🚀 Deployment

### Docker Deployment

1. **Create Dockerfile**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Build and Run**

```bash
# Build image
docker build -t fastapi-sagemaker .

# Run container
docker run -p 8000:8000 --env-file .env fastapi-sagemaker
```

### Docker Compose

```yaml
version: '3.8'
services:
  fastapi-sagemaker:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SAGEMAKER_ENDPOINT_NAME=${SAGEMAKER_ENDPOINT_NAME}
      - AWS_REGION=${AWS_REGION}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    volumes:
      - ./logs:/app/logs
```

### AWS Deployment

1. **EC2 Deployment**

```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-12345678

# Install dependencies
sudo yum update -y
sudo yum install python3 pip git -y

# Clone and setup application
git clone https://github.com/your-username/fastapi_sagemaker.git
cd fastapi_sagemaker
pip3 install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your settings

# Run application
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

2. **ECS Deployment**

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name fastapi-sagemaker

# Create task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster fastapi-sagemaker \
  --service-name fastapi-sagemaker-service \
  --task-definition fastapi-sagemaker:1 \
  --desired-count 2
```

### Production Considerations

- **Environment Variables**: Use AWS Systems Manager Parameter Store
- **Authentication**: Implement API key or JWT authentication
- **Rate Limiting**: Add rate limiting to prevent abuse
- **Monitoring**: Set up CloudWatch monitoring and alarms
- **HTTPS**: Use Application Load Balancer with SSL certificate
- **Auto Scaling**: Configure auto-scaling based on CPU/memory usage
- **Logging**: Centralized logging with CloudWatch Logs
- **Security**: Use VPC and security groups for network isolation

## 🐛 Troubleshooting

### Common Issues

#### AWS Credentials

**Issue**: `NoCredentialsError: Unable to locate credentials`
```bash
# Solution: Configure AWS credentials
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
```

**Issue**: `AccessDenied: User is not authorized to perform: sagemaker:InvokeEndpoint`
```bash
# Solution: Add SageMaker permissions to IAM user/role
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint",
        "sagemaker:DescribeEndpoint"
      ],
      "Resource": "*"
    }
  ]
}
```

#### SageMaker Endpoint

**Issue**: `Endpoint not found`
```bash
# Solution: Verify endpoint name and region
aws sagemaker list-endpoints --region eu-north-1
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name your-endpoint-name
```

**Issue**: `Endpoint not in service`
```bash
# Solution: Wait for endpoint to be ready
aws sagemaker wait endpoint-in-service --endpoint-name your-endpoint-name
```

#### Application Issues

**Issue**: `ModuleNotFoundError: No module named 'fastapi'`
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue**: `Address already in use`
```bash
# Solution: Use different port
uvicorn main:app --port 8001
# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check

Test endpoint connectivity:

```python
import boto3

runtime = boto3.client('sagemaker-runtime', region_name='eu-north-1')
response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body='{"inputs": {"question": "test", "context": "test"}}'
)
print(response['Body'].read().decode())
```

### Logs

Check application logs:

```bash
# Application logs
tail -f logs/app.log

# SageMaker logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/fastapi_sagemaker.git
   cd fastapi_sagemaker
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Changes**
   - Follow PEP 8 for Python code
   - Write tests for new features
   - Update documentation

4. **Test Your Changes**
   ```bash
   python test_example.py
   ```

5. **Commit and Push**
   ```bash
   git commit -m "Add amazing feature"
   git push origin feature/amazing-feature
   ```

6. **Create Pull Request**
   - Open a pull request on GitHub
   - Provide clear description of changes
   - Include tests and documentation updates

### Development Guidelines

- **Code Style**: Follow PEP 8 (Python)
- **Testing**: Write unit tests for new features
- **Documentation**: Update README and API docs
- **Commits**: Use conventional commit messages
- **Reviews**: All PRs require code review

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [AWS SageMaker](https://aws.amazon.com/sagemaker/) for ML model hosting
- [Hugging Face](https://huggingface.co/) for pre-trained models
- [Pydantic](https://pydantic.dev/) for data validation
- [Uvicorn](https://www.uvicorn.org/) for ASGI server
- [Boto3](https://boto3.amazonaws.com/) for AWS SDK

## 📞 Support

### Getting Help

- **Documentation**: Check this README and API docs
- **Issues**: [GitHub Issues](https://github.com/your-username/fastapi_sagemaker/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/fastapi_sagemaker/discussions)

### Community

- **Discord**: Join our [Discord server](https://discord.gg/fastapi-sagemaker)
- **Twitter**: Follow [@FastAPISageMaker](https://twitter.com/FastAPISageMaker)
- **Blog**: Read our [blog posts](https://blog.fastapi-sagemaker.com)

### Enterprise Support

For enterprise customers, we offer:
- **Priority Support**: 24/7 technical support
- **Custom Development**: Tailored features and integrations
- **Training**: Team training and workshops
- **Consulting**: Architecture and deployment guidance

Contact us at: enterprise@fastapi-sagemaker.com

---

**Made with ❤️ by the FastAPI SageMaker Team**

*Empowering AI applications with enterprise-grade SageMaker integration* 