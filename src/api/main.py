"""FastAPI main application for the MLOps system."""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import psutil
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.performance import PerformanceMonitor
from ..utils.security import SecurityManager
from .auth import init_auth
from .models import (
    DataProcessingRequest,
    DataProcessingResponse,
    ErrorResponse,
    HealthResponse,
    MonitoringResponse,
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
)


# Global variables
config: Config
logger: Any
performance_monitor: PerformanceMonitor
security_manager: SecurityManager
start_time: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global config, logger, performance_monitor, security_manager, start_time
    
    # Startup
    config = Config.load_config()
    logger = get_logger("api", config)
    performance_monitor = PerformanceMonitor(config)
    security_manager = SecurityManager(config)
    
    # Initialize authentication
    init_auth(config)
    
    start_time = time.time()
    
    # Start periodic monitoring
    performance_monitor.start_periodic_monitoring()
    
    logger.info(
        "API server started",
        host=config.api.host,
        port=config.api.port,
        environment=config.app.environment,
    )
    
    yield
    
    # Shutdown
    logger.info("API server shutting down")


# Create FastAPI app
app = FastAPI(
    title=config.app.name if 'config' in globals() else "MLOps API",
    version=config.app.version if 'config' in globals() else "0.1.0",
    description="MLOps Lifecycle Management API",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins if 'config' in globals() else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Track performance
    if 'performance_monitor' in globals():
        performance_monitor.track_request(
            endpoint=str(request.url.path),
            method=request.method,
            status=response.status_code,
            duration=processing_time,
        )
    
    # Log request
    if 'logger' in globals():
        logger.info(
            f"{request.method} {request.url.path}",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            processing_time=processing_time,
            client_ip=request.client.host if request.client else "unknown",
        )
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    error_response = ErrorResponse(
        error=str(exc),
        error_type=type(exc).__name__,
        details={"path": str(request.url.path), "method": request.method},
    )
    
    if 'logger' in globals():
        logger.error(
            "Unhandled exception",
            error=str(exc),
            error_type=type(exc).__name__,
            path=str(request.url.path),
            method=request.method,
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict(),
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "MLOps Lifecycle Management API",
        "version": config.app.version if 'config' in globals() else "0.1.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if 'performance_monitor' not in globals():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized",
        )
    
    health_status = performance_monitor.get_health_status()
    
    return HealthResponse(
        status=health_status["status"],
        version=config.app.version,
        uptime=time.time() - start_time,
        memory_usage=health_status["memory_usage_percent"],
        cpu_usage=health_status["cpu_usage_percent"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using trained models."""
    try:
        # Validate request
        validation_result = security_manager.validate_api_request(request.dict())
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid request: {validation_result['errors']}",
            )
        
        # Track prediction performance
        with performance_monitor.track_operation("prediction"):
            # Mock prediction for demo
            prediction = 0.75  # Mock prediction
            probability = 0.85  # Mock probability
            
            # In real implementation, load model and make prediction
            # model = load_model(request.model_name, request.version)
            # prediction = model.predict(request.input_data)
            # probability = model.predict_proba(request.input_data)
        
        # Track prediction
        performance_monitor.track_prediction(
            model_name=request.model_name,
            status="success",
            duration=0.1,  # Mock duration
        )
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_name=request.model_name,
            version=request.version or "latest",
            processing_time=0.1,
        )
        
    except Exception as e:
        logger.error(
            "Prediction failed",
            error=str(e),
            model_name=request.model_name,
            version=request.version,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Train a new model."""
    try:
        from ..models.trainer import ModelTrainer
        
        trainer = ModelTrainer(config)
        
        # Track training performance
        with performance_monitor.track_operation("training"):
            pipeline_results = trainer.train_pipeline(
                data_path=request.data_path,
                model_name=request.model_name,
                version=request.version,
            )
        
        return TrainingResponse(
            model_name=pipeline_results["model_name"],
            version=pipeline_results["version"],
            model_path=pipeline_results["model_path"],
            mlflow_run_id=pipeline_results["mlflow_run_id"],
            training_results=pipeline_results["training_results"],
            status="completed",
        )
        
    except Exception as e:
        logger.error(
            "Training failed",
            error=str(e),
            model_name=request.model_name,
            version=request.version,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        )


@app.post("/process-data", response_model=DataProcessingResponse)
async def process_data(request: DataProcessingRequest):
    """Process data through the pipeline."""
    try:
        from ..data.processor import DataProcessor
        
        processor = DataProcessor(config)
        
        # Track data processing performance
        with performance_monitor.track_operation("data_processing"):
            df, processing_metadata = processor.process_data_pipeline(
                input_file=request.input_file,
                output_file=request.output_file,
            )
        
        return DataProcessingResponse(
            input_file=request.input_file,
            output_file=request.output_file,
            output_path=processing_metadata["output_path"],
            processing_metadata=processing_metadata,
            status="completed",
        )
        
    except Exception as e:
        logger.error(
            "Data processing failed",
            error=str(e),
            input_file=request.input_file,
            output_file=request.output_file,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data processing failed: {str(e)}",
        )


@app.get("/monitor/{model_name}", response_model=MonitoringResponse)
async def get_monitoring_data(model_name: str, version: str = "latest"):
    """Get monitoring data for a model."""
    try:
        # Mock monitoring data
        drift_scores = {
            "age": 0.02,
            "income": 0.05,
            "tenure": 0.03,
        }
        
        performance_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80,
        }
        
        alerts = []
        
        # Check for drift
        for feature, score in drift_scores.items():
            if score > config.monitoring.drift_threshold:
                alerts.append({
                    "type": "data_drift",
                    "feature": feature,
                    "score": score,
                    "threshold": config.monitoring.drift_threshold,
                    "severity": "warning" if score < 0.1 else "critical",
                })
        
        # Check for performance degradation
        if performance_metrics["accuracy"] < config.monitoring.performance_threshold:
            alerts.append({
                "type": "performance_degradation",
                "metric": "accuracy",
                "current": performance_metrics["accuracy"],
                "threshold": config.monitoring.performance_threshold,
                "severity": "critical",
            })
        
        return MonitoringResponse(
            model_name=model_name,
            version=version,
            drift_scores=drift_scores,
            performance_metrics=performance_metrics,
            alerts=alerts,
        )
        
    except Exception as e:
        logger.error(
            "Failed to get monitoring data",
            error=str(e),
            model_name=model_name,
            version=version,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring data: {str(e)}",
        )


@app.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List available models."""
    try:
        # Mock model list
        models = [
            {
                "name": "churn_predictor",
                "version": "v1.0.0",
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "performance": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.78,
                },
            },
            {
                "name": "churn_predictor",
                "version": "v0.9.0",
                "status": "archived",
                "created_at": "2023-12-01T00:00:00Z",
                "performance": {
                    "accuracy": 0.82,
                    "precision": 0.79,
                    "recall": 0.75,
                },
            },
        ]
        
        return {"models": models}
        
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@app.get("/performance/{operation_name}", response_model=Dict[str, Any])
async def get_performance_stats(operation_name: str):
    """Get performance statistics for an operation."""
    try:
        if 'performance_monitor' not in globals():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Performance monitor not available",
            )
        
        stats = performance_monitor.get_performance_stats(operation_name)
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No performance data for operation: {operation_name}",
            )
        
        return {
            "operation_name": operation_name,
            "statistics": stats,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get performance stats",
            error=str(e),
            operation_name=operation_name,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance stats: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host if 'config' in globals() else "0.0.0.0",
        port=config.api.port if 'config' in globals() else 8000,
        reload=config.api.reload if 'config' in globals() else True,
    ) 