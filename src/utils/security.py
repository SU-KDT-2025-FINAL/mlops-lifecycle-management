"""Security management for the MLOps system."""

import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import Config


class TokenData(BaseModel):
    """Token data model."""
    
    username: Optional[str] = None
    scopes: list[str] = []


class SecurityManager:
    """Security manager for authentication and authorization."""
    
    def __init__(self, config: Config):
        """Initialize security manager.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = config.security.secret_key
        self.algorithm = config.security.algorithm
        self.access_token_expire_minutes = config.security.access_token_expire_minutes
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            plain_password: Plain text password.
            hashed_password: Hashed password.
            
        Returns:
            True if password matches hash.
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash.
        
        Args:
            password: Plain text password.
            
        Returns:
            Hashed password.
        """
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], 
                          expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token.
        
        Args:
            data: Token data.
            expires_delta: Token expiration time.
            
        Returns:
            JWT token string.
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify JWT token.
        
        Args:
            token: JWT token string.
            
        Returns:
            Token data if valid, None otherwise.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            scopes: list[str] = payload.get("scopes", [])
            if username is None:
                return None
            token_data = TokenData(username=username, scopes=scopes)
            return token_data
        except JWTError:
            return None
    
    def generate_api_key(self) -> str:
        """Generate a secure API key.
        
        Returns:
            Secure API key.
        """
        return secrets.token_urlsafe(32)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage.
        
        Args:
            api_key: Plain API key.
            
        Returns:
            Hashed API key.
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength.
        
        Args:
            password: Password to validate.
            
        Returns:
            Validation result with details.
        """
        errors = []
        warnings = []
        
        # Check minimum length
        if len(password) < self.config.security.password_min_length:
            errors.append(f"Password must be at least {self.config.security.password_min_length} characters long")
        
        # Check for uppercase letters
        if not any(c.isupper() for c in password):
            warnings.append("Password should contain at least one uppercase letter")
        
        # Check for lowercase letters
        if not any(c.islower() for c in password):
            warnings.append("Password should contain at least one lowercase letter")
        
        # Check for numbers
        if not any(c.isdigit() for c in password):
            warnings.append("Password should contain at least one number")
        
        # Check for special characters
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            warnings.append("Password should contain at least one special character")
        
        # Check for common patterns
        common_patterns = ["123", "abc", "qwerty", "password", "admin"]
        password_lower = password.lower()
        for pattern in common_patterns:
            if pattern in password_lower:
                errors.append(f"Password contains common pattern: {pattern}")
                break
        
        is_valid = len(errors) == 0
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "strength_score": self._calculate_strength_score(password)
        }
    
    def _calculate_strength_score(self, password: str) -> int:
        """Calculate password strength score (0-100).
        
        Args:
            password: Password to score.
            
        Returns:
            Strength score from 0 to 100.
        """
        score = 0
        
        # Length contribution
        score += min(len(password) * 4, 25)
        
        # Character variety contribution
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        score += (has_upper + has_lower + has_digit + has_special) * 10
        
        # Complexity contribution
        unique_chars = len(set(password))
        score += min(unique_chars * 2, 20)
        
        return min(score, 100)
    
    def sanitize_input(self, input_data: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """Sanitize user input to prevent injection attacks.
        
        Args:
            input_data: Input data to sanitize.
            
        Returns:
            Sanitized input data.
        """
        if isinstance(input_data, str):
            return self._sanitize_string(input_data)
        elif isinstance(input_data, dict):
            return {k: self.sanitize_input(v) for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]
        else:
            return input_data
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string input.
        
        Args:
            text: String to sanitize.
            
        Returns:
            Sanitized string.
        """
        # Remove potentially dangerous characters
        dangerous_chars = ["<", ">", "'", '"', "&", ";", "(", ")", "{", "}"]
        sanitized = text
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        
        # Limit length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def validate_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API request data.
        
        Args:
            request_data: Request data to validate.
            
        Returns:
            Validation result with sanitized data.
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "sanitized_data": {}
        }
        
        try:
            # Sanitize input
            sanitized_data = self.sanitize_input(request_data)
            validation_result["sanitized_data"] = sanitized_data
            
            # Check for required fields
            required_fields = ["model_name", "input_data"]
            for field in required_fields:
                if field not in sanitized_data:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"Missing required field: {field}")
            
            # Validate data types
            if "input_data" in sanitized_data:
                if not isinstance(sanitized_data["input_data"], dict):
                    validation_result["is_valid"] = False
                    validation_result["errors"].append("input_data must be a dictionary")
            
            # Check for suspicious patterns
            suspicious_patterns = ["javascript:", "data:", "vbscript:"]
            data_str = str(sanitized_data).lower()
            for pattern in suspicious_patterns:
                if pattern in data_str:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"Suspicious pattern detected: {pattern}")
                    break
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def rate_limit_check(self, user_id: str, action: str, 
                        max_requests: int = 100, window_seconds: int = 3600) -> bool:
        """Check if user has exceeded rate limit.
        
        Args:
            user_id: User identifier.
            action: Action being performed.
            max_requests: Maximum requests allowed.
            window_seconds: Time window in seconds.
            
        Returns:
            True if request is allowed, False if rate limited.
        """
        # This is a simplified implementation
        # In production, you would use Redis or similar for rate limiting
        current_time = int(time.time())
        key = f"rate_limit:{user_id}:{action}"
        
        # Mock implementation - in real system, check against Redis/database
        return True  # Always allow for demo purposes
    
    def audit_log(self, user_id: str, action: str, resource: str, 
                  success: bool, details: Optional[Dict[str, Any]] = None) -> None:
        """Log security audit event.
        
        Args:
            user_id: User identifier.
            action: Action performed.
            resource: Resource accessed.
            success: Whether action was successful.
            details: Additional details.
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "success": success,
            "ip_address": "127.0.0.1",  # In real system, get from request
            "user_agent": "MLOps-Client/1.0",  # In real system, get from request
        }
        
        if details:
            audit_entry["details"] = details
        
        # In production, log to secure audit log
        print(f"AUDIT: {audit_entry}") 