"""Tests for utility modules."""

import pytest
from unittest.mock import Mock, patch

from src.utils.config import Config
from src.utils.logging import get_logger, setup_logging
from src.utils.security import SecurityManager
from src.utils.performance import PerformanceMonitor


class TestConfig:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = Config.load_config()
        assert config is not None
        assert config.app.name == "MLOps Lifecycle Management"
        assert config.api.port == 8000
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config.load_config()
        assert config.validate() is True
    
    def test_config_get_method(self):
        """Test configuration get method."""
        config = Config.load_config()
        assert config.get("api.port") == 8000
        assert config.get("nonexistent.key", "default") == "default"


class TestLogging:
    """Test logging functionality."""
    
    def test_logger_creation(self):
        """Test logger creation."""
        config = Config.load_config()
        logger = get_logger("test_logger", config)
        assert logger is not None
    
    def test_logger_with_default_config(self):
        """Test logger with default config."""
        logger = get_logger("test_logger")
        assert logger is not None


class TestSecurity:
    """Test security functionality."""
    
    def test_security_manager_creation(self):
        """Test security manager creation."""
        config = Config.load_config()
        security_manager = SecurityManager(config)
        assert security_manager is not None
    
    def test_password_hashing(self):
        """Test password hashing."""
        config = Config.load_config()
        security_manager = SecurityManager(config)
        
        password = "test_password_123"
        hashed = security_manager.get_password_hash(password)
        
        assert hashed != password
        assert security_manager.verify_password(password, hashed)
        assert not security_manager.verify_password("wrong_password", hashed)
    
    def test_password_strength_validation(self):
        """Test password strength validation."""
        config = Config.load_config()
        security_manager = SecurityManager(config)
        
        # Test weak password
        weak_password = "123"
        result = security_manager.validate_password_strength(weak_password)
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        
        # Test strong password
        strong_password = "StrongPassword123!"
        result = security_manager.validate_password_strength(strong_password)
        assert result["is_valid"] is True
        assert result["strength_score"] > 70
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        config = Config.load_config()
        security_manager = SecurityManager(config)
        
        # Test string sanitization
        dirty_input = "<script>alert('xss')</script>"
        clean_input = security_manager.sanitize_input(dirty_input)
        assert "<script>" not in clean_input
        assert "alert" not in clean_input
        
        # Test dictionary sanitization
        dirty_dict = {
            "name": "<script>alert('xss')</script>",
            "data": {"field": "javascript:alert('xss')"}
        }
        clean_dict = security_manager.sanitize_input(dirty_dict)
        assert "<script>" not in str(clean_dict)
        assert "javascript:" not in str(clean_dict)
    
    def test_api_request_validation(self):
        """Test API request validation."""
        config = Config.load_config()
        security_manager = SecurityManager(config)
        
        # Test valid request
        valid_request = {
            "model_name": "test_model",
            "input_data": {"feature1": 1.0, "feature2": 2.0}
        }
        result = security_manager.validate_api_request(valid_request)
        assert result["is_valid"] is True
        
        # Test invalid request
        invalid_request = {
            "input_data": "not_a_dict"
        }
        result = security_manager.validate_api_request(invalid_request)
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0


class TestPerformance:
    """Test performance monitoring."""
    
    def test_performance_monitor_creation(self):
        """Test performance monitor creation."""
        config = Config.load_config()
        monitor = PerformanceMonitor(config)
        assert monitor is not None
    
    def test_operation_tracking(self):
        """Test operation tracking."""
        config = Config.load_config()
        monitor = PerformanceMonitor(config)
        
        with monitor.track_operation("test_operation"):
            # Simulate some work
            import time
            time.sleep(0.1)
        
        stats = monitor.get_performance_stats("test_operation")
        assert stats["count"] == 1
        assert stats["mean"] > 0
    
    def test_health_status(self):
        """Test health status."""
        config = Config.load_config()
        monitor = PerformanceMonitor(config)
        
        health = monitor.get_health_status()
        assert "status" in health
        assert "health_score" in health
        assert "memory_usage_percent" in health
        assert "cpu_usage_percent" in health
    
    def test_performance_thresholds(self):
        """Test performance threshold checking."""
        config = Config.load_config()
        monitor = PerformanceMonitor(config)
        
        violations = monitor.check_performance_thresholds()
        assert isinstance(violations, dict)
    
    @pytest.mark.asyncio
    async def test_async_operation_tracking(self):
        """Test async operation tracking."""
        config = Config.load_config()
        monitor = PerformanceMonitor(config)
        
        async with monitor.track_async_operation("async_test"):
            # Simulate async work
            import asyncio
            await asyncio.sleep(0.1)
        
        stats = monitor.get_performance_stats("async_test")
        assert stats["count"] == 1
        assert stats["mean"] > 0


class TestIntegration:
    """Integration tests."""
    
    def test_config_logging_integration(self):
        """Test configuration and logging integration."""
        config = Config.load_config()
        logger = setup_logging(config, "integration_test")
        assert logger is not None
        
        # Test logging
        logger.info("Test message", test_field="test_value")
    
    def test_security_logging_integration(self):
        """Test security and logging integration."""
        config = Config.load_config()
        security_manager = SecurityManager(config)
        logger = get_logger("security_test", config)
        
        # Test audit logging
        security_manager.audit_log(
            user_id="test_user",
            action="test_action",
            resource="test_resource",
            success=True
        )
        
        assert logger is not None
    
    def test_performance_logging_integration(self):
        """Test performance and logging integration."""
        config = Config.load_config()
        monitor = PerformanceMonitor(config)
        logger = get_logger("performance_test", config)
        
        # Test performance logging
        monitor.track_prediction(
            model_name="test_model",
            status="success",
            duration=0.1
        )
        
        assert logger is not None 