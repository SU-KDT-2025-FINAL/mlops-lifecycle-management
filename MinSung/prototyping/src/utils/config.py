"""Configuration management for the MLOps system."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Application configuration model."""
    
    name: str = Field(default="MLOps Lifecycle Management")
    version: str = Field(default="0.1.0")
    environment: str = Field(default="production")
    debug: bool = Field(default=False)


class APIConfig(BaseModel):
    """API configuration model."""
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    reload: bool = Field(default=True)
    cors_origins: List[str] = Field(default_factory=list)


class SecurityConfig(BaseModel):
    """Security configuration model."""
    
    secret_key: str = Field(default="your-secret-key-here")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    password_min_length: int = Field(default=8)


class DatabaseConfig(BaseModel):
    """Database configuration model."""
    
    url: str = Field(default="postgresql://user:password@localhost:5432/mlops_db")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    pool_timeout: int = Field(default=30)


class RedisConfig(BaseModel):
    """Redis configuration model."""
    
    url: str = Field(default="redis://localhost:6379/0")
    pool_size: int = Field(default=10)
    decode_responses: bool = Field(default=True)


class MLflowConfig(BaseModel):
    """MLflow configuration model."""
    
    tracking_uri: str = Field(default="http://localhost:5000")
    experiment_name: str = Field(default="churn_prediction")
    registry_uri: Optional[str] = Field(default=None)


class DVCConfig(BaseModel):
    """DVC configuration model."""
    
    remote_url: str = Field(default="s3://your-bucket/mlops-data")
    remote_name: str = Field(default="storage")


class MonitoringConfig(BaseModel):
    """Monitoring configuration model."""
    
    prometheus_port: int = Field(default=9090)
    grafana_port: int = Field(default=3000)
    alert_webhook_url: Optional[str] = Field(default=None)
    drift_threshold: float = Field(default=0.05)
    performance_threshold: float = Field(default=0.8)
    latency_threshold: int = Field(default=1000)


class LoggingConfig(BaseModel):
    """Logging configuration model."""
    
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    file: str = Field(default="logs/mlops.log")
    max_bytes: int = Field(default=10485760)  # 10MB
    backup_count: int = Field(default=5)


class DataConfig(BaseModel):
    """Data configuration model."""
    
    raw_path: str = Field(default="data/raw/")
    processed_path: str = Field(default="data/processed/")
    features_path: str = Field(default="data/features/")
    test_size: float = Field(default=0.2)
    validation_size: float = Field(default=0.2)
    random_state: int = Field(default=42)


class FeaturesConfig(BaseModel):
    """Features configuration model."""
    
    numerical_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    target_column: str = Field(default="churn")
    feature_columns: List[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Model configuration model."""
    
    name: str = Field(default="churn_predictor")
    version: str = Field(default="v1.0.0")
    path: str = Field(default="models/")
    type: str = Field(default="random_forest")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    evaluation_metrics: List[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    """Training configuration model."""
    
    cross_validation_folds: int = Field(default=5)
    early_stopping_patience: int = Field(default=10)
    max_epochs: int = Field(default=100)
    batch_size: int = Field(default=32)
    learning_rate: float = Field(default=0.01)


class TestingConfig(BaseModel):
    """Testing configuration model."""
    
    coverage_threshold: int = Field(default=80)
    test_timeout: int = Field(default=300)
    parallel_workers: int = Field(default=4)


class DockerConfig(BaseModel):
    """Docker configuration model."""
    
    registry: str = Field(default="your-registry.com")
    image_name: str = Field(default="mlops-api")
    tag: str = Field(default="latest")
    build_context: str = Field(default=".")
    dockerfile: str = Field(default="Dockerfile")


class CICDConfig(BaseModel):
    """CI/CD configuration model."""
    
    github_actions: Dict[str, Any] = Field(default_factory=dict)
    automated_testing: Dict[str, Any] = Field(default_factory=dict)
    automated_deployment: Dict[str, Any] = Field(default_factory=dict)


class NotificationsConfig(BaseModel):
    """Notifications configuration model."""
    
    email: Dict[str, Any] = Field(default_factory=dict)
    slack: Dict[str, Any] = Field(default_factory=dict)


class AWSConfig(BaseModel):
    """AWS configuration model."""
    
    region: str = Field(default="us-east-1")
    access_key_id: Optional[str] = Field(default=None)
    secret_access_key: Optional[str] = Field(default=None)
    s3_bucket: Optional[str] = Field(default=None)


class Config(BaseModel):
    """Main configuration class."""
    
    app: AppConfig = Field(default_factory=AppConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    dvc: DVCConfig = Field(default_factory=DVCConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    ci_cd: CICDConfig = Field(default_factory=CICDConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)

    @classmethod
    def load_config(cls, config_path: Optional[Union[str, Path]] = None) -> "Config":
        """Load configuration from file and environment variables.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
            
        Returns:
            Config instance with loaded configuration.
        """
        # Load environment variables
        load_dotenv()
        
        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML configuration
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        # Replace environment variable placeholders
        config_data = cls._replace_env_vars(config_data)
        
        return cls(**config_data)
    
    @staticmethod
    def _replace_env_vars(data: Any) -> Any:
        """Replace environment variable placeholders in configuration data.
        
        Args:
            data: Configuration data (dict, list, or primitive type).
            
        Returns:
            Data with environment variables replaced.
        """
        if isinstance(data, dict):
            return {key: Config._replace_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [Config._replace_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var, data)
        else:
            return data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'api.port').
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        keys = key.split(".")
        value = self.dict()
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if configuration is valid.
        """
        try:
            # Validate required fields
            if not self.security.secret_key or self.security.secret_key == "your-secret-key-here":
                raise ValueError("SECRET_KEY must be set in environment variables")
            
            # Validate paths exist
            data_path = Path(self.data.raw_path)
            if not data_path.exists():
                data_path.mkdir(parents=True, exist_ok=True)
            
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False 