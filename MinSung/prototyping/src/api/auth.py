"""Authentication utilities for the API."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from ..utils.config import Config
from ..utils.security import SecurityManager

# Security scheme
security = HTTPBearer()

# Global security manager (will be initialized in main.py)
security_manager: Optional[SecurityManager] = None


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token.
    
    Args:
        credentials: HTTP Bearer token credentials.
        
    Returns:
        User data from token.
        
    Raises:
        HTTPException: If token is invalid or expired.
    """
    if security_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security manager not initialized"
        )
    
    token = credentials.credentials
    try:
        token_data = security_manager.verify_token(token)
        if token_data is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token_data
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create access token.
    
    Args:
        data: Data to encode in token.
        expires_delta: Token expiration time.
        
    Returns:
        JWT token string.
    """
    if security_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security manager not initialized"
        )
    
    return security_manager.create_access_token(data, expires_delta)


def init_auth(config: Config):
    """Initialize authentication module.
    
    Args:
        config: Configuration object.
    """
    global security_manager
    security_manager = SecurityManager(config) 