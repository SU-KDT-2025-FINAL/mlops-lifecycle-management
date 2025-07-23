"""Logging configuration for the MLOps system."""

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format.
            
        Returns:
            JSON formatted log string.
        """
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class StructuredLogger:
    """Structured logger with additional context support."""
    
    def __init__(self, name: str, config: Config):
        """Initialize structured logger.
        
        Args:
            name: Logger name.
            config: Configuration object.
        """
        self.logger = logging.getLogger(name)
        self.config = config
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up logger with handlers and formatters."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        level = getattr(logging, self.config.logging.level.upper())
        self.logger.setLevel(level)
        
        # Create formatter
        if self.config.logging.format.lower() == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.logging.file:
            log_file = Path(self.config.logging.file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.logging.max_bytes,
                backupCount=self.config.logging.backup_count,
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_with_context(self, level: str, message: str, **kwargs: Any) -> None:
        """Log message with additional context.
        
        Args:
            level: Log level.
            message: Log message.
            **kwargs: Additional context fields.
        """
        log_method = getattr(self.logger, level.lower())
        
        # Create a custom log record with extra fields
        record = self.logger.makeRecord(
            self.logger.name,
            getattr(logging, level.upper()),
            "",
            0,
            message,
            (),
            None,
        )
        record.extra_fields = kwargs
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self.log_with_context("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self.log_with_context("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self.log_with_context("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self.log_with_context("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with context."""
        self.log_with_context("CRITICAL", message, **kwargs)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with context."""
        self.log_with_context("ERROR", message, **kwargs)


def setup_logging(config: Config, logger_name: str = "mlops") -> StructuredLogger:
    """Set up logging for the application.
    
    Args:
        config: Configuration object.
        logger_name: Name for the logger.
        
    Returns:
        Configured structured logger.
    """
    return StructuredLogger(logger_name, config)


def get_logger(name: str, config: Optional[Config] = None) -> StructuredLogger:
    """Get a logger instance.
    
    Args:
        name: Logger name.
        config: Configuration object. If None, loads default config.
        
    Returns:
        Structured logger instance.
    """
    if config is None:
        config = Config.load_config()
    
    return StructuredLogger(name, config)


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, config: Config):
        """Initialize performance logger.
        
        Args:
            config: Configuration object.
        """
        self.logger = get_logger("performance", config)
    
    def log_prediction(self, model_name: str, prediction_time: float, 
                      prediction: Any, input_data: Dict[str, Any]) -> None:
        """Log prediction performance.
        
        Args:
            model_name: Name of the model.
            prediction_time: Time taken for prediction in seconds.
            prediction: Model prediction.
            input_data: Input data used for prediction.
        """
        self.logger.info(
            "Model prediction completed",
            model_name=model_name,
            prediction_time_ms=prediction_time * 1000,
            prediction=str(prediction),
            input_size=len(str(input_data)),
        )
    
    def log_training_metrics(self, model_name: str, metrics: Dict[str, float],
                           training_time: float, dataset_size: int) -> None:
        """Log training metrics.
        
        Args:
            model_name: Name of the model.
            metrics: Training metrics.
            training_time: Time taken for training in seconds.
            dataset_size: Size of training dataset.
        """
        self.logger.info(
            "Model training completed",
            model_name=model_name,
            training_time_seconds=training_time,
            dataset_size=dataset_size,
            **metrics,
        )
    
    def log_data_processing(self, step: str, processing_time: float,
                          data_size: int, output_size: int) -> None:
        """Log data processing metrics.
        
        Args:
            step: Processing step name.
            processing_time: Time taken for processing in seconds.
            data_size: Input data size.
            output_size: Output data size.
        """
        self.logger.info(
            "Data processing completed",
            step=step,
            processing_time_seconds=processing_time,
            input_size=data_size,
            output_size=output_size,
        )
    
    def log_error(self, error_type: str, error_message: str, 
                 context: Dict[str, Any]) -> None:
        """Log error with context.
        
        Args:
            error_type: Type of error.
            error_message: Error message.
            context: Additional context.
        """
        self.logger.error(
            "Error occurred",
            error_type=error_type,
            error_message=error_message,
            **context,
        ) 