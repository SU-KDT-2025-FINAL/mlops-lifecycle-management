"""Performance monitoring for the MLOps system."""

import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
from prometheus_client import Counter, Gauge, Histogram, Summary

from .config import Config
from .logging import get_logger


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self, config: Config):
        """Initialize performance monitor.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = get_logger("performance", config)
        
        # Prometheus metrics
        self.request_counter = Counter(
            "mlops_requests_total",
            "Total number of requests",
            ["endpoint", "method", "status"]
        )
        
        self.request_duration = Histogram(
            "mlops_request_duration_seconds",
            "Request duration in seconds",
            ["endpoint", "method"]
        )
        
        self.prediction_counter = Counter(
            "mlops_predictions_total",
            "Total number of predictions",
            ["model_name", "status"]
        )
        
        self.prediction_duration = Histogram(
            "mlops_prediction_duration_seconds",
            "Prediction duration in seconds",
            ["model_name"]
        )
        
        self.model_accuracy = Gauge(
            "mlops_model_accuracy",
            "Model accuracy score",
            ["model_name", "version"]
        )
        
        self.data_drift_score = Gauge(
            "mlops_data_drift_score",
            "Data drift score",
            ["model_name", "feature_name"]
        )
        
        self.system_memory = Gauge(
            "mlops_system_memory_bytes",
            "System memory usage in bytes"
        )
        
        self.system_cpu = Gauge(
            "mlops_system_cpu_percent",
            "System CPU usage percentage"
        )
        
        # Performance tracking
        self._performance_data: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}
    
    @contextmanager
    def track_operation(self, operation_name: str, **kwargs: Any):
        """Context manager to track operation performance.
        
        Args:
            operation_name: Name of the operation.
            **kwargs: Additional context for logging.
        """
        start_time = time.time()
        self._start_times[operation_name] = start_time
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._record_operation(operation_name, duration, **kwargs)
            del self._start_times[operation_name]
    
    @asynccontextmanager
    async def track_async_operation(self, operation_name: str, **kwargs: Any):
        """Async context manager to track operation performance.
        
        Args:
            operation_name: Name of the operation.
            **kwargs: Additional context for logging.
        """
        start_time = time.time()
        self._start_times[operation_name] = start_time
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._record_operation(operation_name, duration, **kwargs)
            del self._start_times[operation_name]
    
    def _record_operation(self, operation_name: str, duration: float, **kwargs: Any) -> None:
        """Record operation performance.
        
        Args:
            operation_name: Name of the operation.
            duration: Duration in seconds.
            **kwargs: Additional context.
        """
        if operation_name not in self._performance_data:
            self._performance_data[operation_name] = []
        
        self._performance_data[operation_name].append(duration)
        
        # Keep only last 1000 measurements
        if len(self._performance_data[operation_name]) > 1000:
            self._performance_data[operation_name] = self._performance_data[operation_name][-1000:]
        
        self.logger.info(
            f"Operation '{operation_name}' completed",
            operation_name=operation_name,
            duration_seconds=duration,
            **kwargs
        )
    
    def track_request(self, endpoint: str, method: str, status: int, duration: float) -> None:
        """Track API request performance.
        
        Args:
            endpoint: API endpoint.
            method: HTTP method.
            status: HTTP status code.
            duration: Request duration in seconds.
        """
        self.request_counter.labels(endpoint=endpoint, method=method, status=status).inc()
        self.request_duration.labels(endpoint=endpoint, method=method).observe(duration)
        
        self.logger.info(
            f"API request to {endpoint}",
            endpoint=endpoint,
            method=method,
            status=status,
            duration_seconds=duration,
        )
    
    def track_prediction(self, model_name: str, status: str, duration: float) -> None:
        """Track model prediction performance.
        
        Args:
            model_name: Name of the model.
            status: Prediction status (success/error).
            duration: Prediction duration in seconds.
        """
        self.prediction_counter.labels(model_name=model_name, status=status).inc()
        self.prediction_duration.labels(model_name=model_name).observe(duration)
        
        self.logger.info(
            f"Model prediction for {model_name}",
            model_name=model_name,
            status=status,
            duration_seconds=duration,
        )
    
    def update_model_accuracy(self, model_name: str, version: str, accuracy: float) -> None:
        """Update model accuracy metric.
        
        Args:
            model_name: Name of the model.
            version: Model version.
            accuracy: Accuracy score.
        """
        self.model_accuracy.labels(model_name=model_name, version=version).set(accuracy)
        
        self.logger.info(
            f"Model accuracy updated for {model_name}",
            model_name=model_name,
            version=version,
            accuracy=accuracy,
        )
    
    def update_data_drift(self, model_name: str, feature_name: str, drift_score: float) -> None:
        """Update data drift metric.
        
        Args:
            model_name: Name of the model.
            feature_name: Name of the feature.
            drift_score: Drift score.
        """
        self.data_drift_score.labels(model_name=model_name, feature_name=feature_name).set(drift_score)
        
        self.logger.info(
            f"Data drift detected for {model_name}",
            model_name=model_name,
            feature_name=feature_name,
            drift_score=drift_score,
        )
    
    def update_system_metrics(self) -> None:
        """Update system-level metrics."""
        # Memory usage
        memory = psutil.virtual_memory()
        self.system_memory.set(memory.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_cpu.set(cpu_percent)
        
        self.logger.debug(
            "System metrics updated",
            memory_used_gb=memory.used / (1024**3),
            memory_percent=memory.percent,
            cpu_percent=cpu_percent,
        )
    
    def get_performance_stats(self, operation_name: str) -> Dict[str, float]:
        """Get performance statistics for an operation.
        
        Args:
            operation_name: Name of the operation.
            
        Returns:
            Dictionary with performance statistics.
        """
        if operation_name not in self._performance_data:
            return {}
        
        durations = self._performance_data[operation_name]
        if not durations:
            return {}
        
        return {
            "count": len(durations),
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "p50": self._percentile(durations, 50),
            "p95": self._percentile(durations, 95),
            "p99": self._percentile(durations, 99),
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data.
        
        Args:
            data: List of values.
            percentile: Percentile to calculate (0-100).
            
        Returns:
            Percentile value.
        """
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def check_performance_thresholds(self) -> Dict[str, Any]:
        """Check if performance metrics exceed thresholds.
        
        Returns:
            Dictionary with threshold violations.
        """
        violations = {}
        
        # Check request duration
        for operation_name in self._performance_data:
            stats = self.get_performance_stats(operation_name)
            if stats.get("p95", 0) > self.config.monitoring.latency_threshold / 1000:
                violations[f"{operation_name}_p95_latency"] = {
                    "current": stats["p95"],
                    "threshold": self.config.monitoring.latency_threshold / 1000,
                }
        
        # Check system resources
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            violations["high_memory_usage"] = {
                "current": memory.percent,
                "threshold": 90,
            }
        
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            violations["high_cpu_usage"] = {
                "current": cpu_percent,
                "threshold": 90,
            }
        
        if violations:
            self.logger.warning(
                "Performance thresholds exceeded",
                violations=violations,
            )
        
        return violations
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status.
        
        Returns:
            Dictionary with health status information.
        """
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Calculate overall health score
        health_score = 100
        
        # Deduct points for high resource usage
        if memory.percent > 80:
            health_score -= (memory.percent - 80) * 2
        if cpu_percent > 80:
            health_score -= (cpu_percent - 80) * 2
        
        # Check for performance violations
        violations = self.check_performance_thresholds()
        if violations:
            health_score -= len(violations) * 10
        
        health_score = max(0, health_score)
        
        return {
            "status": "healthy" if health_score > 70 else "degraded" if health_score > 30 else "unhealthy",
            "health_score": health_score,
            "memory_usage_percent": memory.percent,
            "cpu_usage_percent": cpu_percent,
            "performance_violations": len(violations),
            "timestamp": time.time(),
        }
    
    def start_periodic_monitoring(self, interval_seconds: int = 60) -> None:
        """Start periodic monitoring of system metrics.
        
        Args:
            interval_seconds: Monitoring interval in seconds.
        """
        async def monitor_loop():
            """Periodic monitoring loop."""
            while True:
                try:
                    self.update_system_metrics()
                    health_status = self.get_health_status()
                    
                    if health_status["status"] != "healthy":
                        self.logger.warning(
                            "System health degraded",
                            health_status=health_status,
                        )
                    
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    self.logger.error(
                        "Error in monitoring loop",
                        error=str(e),
                    )
                    await asyncio.sleep(interval_seconds)
        
        # Start monitoring in background
        asyncio.create_task(monitor_loop())
    
    def decorator(self, operation_name: str):
        """Decorator for tracking function performance.
        
        Args:
            operation_name: Name of the operation.
            
        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                with self.track_operation(operation_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def async_decorator(self, operation_name: str):
        """Async decorator for tracking function performance.
        
        Args:
            operation_name: Name of the operation.
            
        Returns:
            Async decorator function.
        """
        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                async with self.track_async_operation(operation_name):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator 