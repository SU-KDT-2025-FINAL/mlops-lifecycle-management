"""Pydantic models for the API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    
    model_name: str = Field(..., description="Name of the model to use")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    version: Optional[str] = Field(None, description="Model version to use")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    prediction: Any = Field(..., description="Model prediction")
    probability: Optional[float] = Field(None, description="Prediction probability")
    model_name: str = Field(..., description="Name of the model used")
    version: str = Field(..., description="Version of the model used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    processing_time: float = Field(..., description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    memory_usage: float = Field(..., description="Memory usage percentage")
    cpu_usage: float = Field(..., description="CPU usage percentage")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of the model")
    feature_columns: List[str] = Field(..., description="Feature columns")
    target_column: str = Field(..., description="Target column")
    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    created_at: datetime = Field(..., description="Model creation timestamp")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")


class TrainingRequest(BaseModel):
    """Request model for model training."""
    
    data_path: str = Field(..., description="Path to training data")
    model_name: str = Field(..., description="Name for the model")
    version: str = Field(..., description="Model version")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")


class TrainingResponse(BaseModel):
    """Response model for training results."""
    
    model_name: str = Field(..., description="Name of the trained model")
    version: str = Field(..., description="Model version")
    model_path: str = Field(..., description="Path to saved model")
    mlflow_run_id: str = Field(..., description="MLflow run ID")
    training_results: Dict[str, Any] = Field(..., description="Training results")
    status: str = Field(..., description="Training status")


class DataProcessingRequest(BaseModel):
    """Request model for data processing."""
    
    input_file: str = Field(..., description="Path to input data file")
    output_file: str = Field(..., description="Path for output data file")


class DataProcessingResponse(BaseModel):
    """Response model for data processing results."""
    
    input_file: str = Field(..., description="Path to input data file")
    output_file: str = Field(..., description="Path to output data file")
    output_path: str = Field(..., description="Full path to output file")
    processing_metadata: Dict[str, Any] = Field(..., description="Processing metadata")
    status: str = Field(..., description="Processing status")


class MonitoringResponse(BaseModel):
    """Response model for monitoring data."""
    
    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    drift_scores: Dict[str, float] = Field(..., description="Data drift scores")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    alerts: List[Dict[str, Any]] = Field(..., description="Active alerts")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Monitoring timestamp")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class UserLoginRequest(BaseModel):
    """Request model for user login."""
    
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class UserLoginResponse(BaseModel):
    """Response model for user login."""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user_id: str = Field(..., description="User ID")


class UserInfoResponse(BaseModel):
    """Response model for user information."""
    
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="User email")
    scopes: List[str] = Field(..., description="User permissions")
    created_at: datetime = Field(..., description="User creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp") 