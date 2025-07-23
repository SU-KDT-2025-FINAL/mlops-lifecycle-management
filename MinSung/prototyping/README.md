# MLOps Lifecycle Management

A comprehensive MLOps lifecycle management system that demonstrates best practices for data versioning, experiment tracking, model deployment, monitoring, and automation.

## Project Overview

This project implements a complete MLOps pipeline for a machine learning system that predicts customer churn. The system includes:

- **Data Version Control**: Using DVC for tracking data changes
- **Experiment Tracking**: MLflow for model experiment management
- **Model Training**: Automated training pipeline with hyperparameter optimization
- **Model Deployment**: FastAPI-based REST API with containerization
- **Monitoring**: Real-time model performance and data drift monitoring
- **CI/CD**: Automated testing, building, and deployment pipeline
- **Security**: JWT authentication and input validation
- **Testing**: Comprehensive test coverage with unit, integration, and E2E tests

## MLOps Lifecycle

The project follows a complete MLOps lifecycle:

1. **Data Management**: Raw data → Feature engineering → Data validation
2. **Model Development**: Experiment tracking → Model training → Model evaluation
3. **Model Deployment**: Model packaging → Containerization → API serving
4. **Monitoring**: Performance monitoring → Data drift detection → Alerting
5. **Automation**: CI/CD pipeline → Automated retraining → Deployment

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Data Pipeline  │───▶│  Feature Store  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model Store   │◀───│  Model Training │◀───│  Experiment     │
└─────────────────┘    └─────────────────┘    │   Tracking      │
                                │             └─────────────────┘
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│  Model Serving  │◀───│  Model Registry │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## How to Run

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git

### Quick Start (Recommended)

1. **Clone and setup the project**
   ```bash
   git clone <repository-url>
   cd mlops-lifecycle-management
   python scripts/setup.py
   ```

2. **Configure environment**
   ```bash
   # Edit .env file with your configuration
   nano .env
   ```

3. **Start infrastructure services**
   ```bash
   docker-compose up -d
   ```

4. **Start the API server**
   ```bash
   python -m src.api.main
   ```

5. **Access the services**
   - API Documentation: http://localhost:8000/docs
   - Grafana Dashboard: http://localhost:3000 (admin/admin)
   - MLflow Tracking: http://localhost:5000
   - Prometheus Metrics: http://localhost:9090

### Manual Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-lifecycle-management
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   pip install -e ".[dev]"
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize DVC**
   ```bash
   dvc init
   dvc remote add -d storage s3://your-bucket/mlops-data
   ```

5. **Start infrastructure services**
   ```bash
   docker-compose up -d
   ```

### Running the Complete Pipeline

1. **Data Processing**
   ```bash
   python -m src.data.process_data
   ```

2. **Model Training**
   ```bash
   python -m src.models.train_model
   ```

3. **Start API Server**
   ```bash
   python -m src.api.main
   ```

4. **Run Monitoring**
   ```bash
   python -m src.monitoring.monitor
   ```

### Docker Deployment

```bash
# Build the image
docker build -t mlops-api .

# Run the container
docker run -p 8000:8000 mlops-api
```

### Development Workflow

1. **Run tests**
   ```bash
   pytest
   ```

2. **Code formatting**
   ```bash
   black src/
   ruff check src/
   ```

3. **Type checking**
   ```bash
   mypy src/
   ```

4. **Pre-commit hooks (if installed)**
   ```bash
   pre-commit run --all-files
   ```

### Development

1. **Run tests**
   ```bash
   pytest
   ```

2. **Code formatting**
   ```bash
   black src/
   ruff check src/
   ```

3. **Type checking**
   ```bash
   mypy src/
   ```

### Docker Deployment

```bash
# Build the image
docker build -t mlops-api .

# Run the container
docker run -p 8000:8000 mlops-api
```

## Project Structure

```
mlops-lifecycle-management/
├── src/
│   ├── api/                 # FastAPI application
│   ├── data/               # Data processing pipeline
│   ├── models/             # Model training and evaluation
│   ├── monitoring/         # Model monitoring and alerting
│   ├── utils/              # Shared utilities
│   └── cli/               # Command-line interfaces
├── tests/                 # Test suite
├── config/                # Configuration files
├── data/                  # Data directory (DVC managed)
├── notebooks/             # Jupyter notebooks
├── infrastructure/        # Terraform configurations
├── .github/              # GitHub Actions workflows
├── docker/               # Docker configurations
└── docs/                 # Documentation
```

## Key Features

- **Data Version Control**: Track data changes with DVC
- **Experiment Tracking**: MLflow for reproducible experiments
- **Model Registry**: Centralized model storage and versioning
- **API Serving**: FastAPI with automatic documentation
- **Monitoring**: Real-time performance and drift monitoring
- **Security**: JWT authentication and input validation
- **Testing**: Comprehensive test coverage
- **CI/CD**: Automated pipeline with GitHub Actions
- **Containerization**: Docker-based deployment
- **Infrastructure**: Terraform for infrastructure as code

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 