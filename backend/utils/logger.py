"""
Production logger configuration.
Replaces print() statements with structured JSON logging or standard Python logging.
"""

import sys
import logging
from functools import lru_cache

@lru_cache()
def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if get_logger is called repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Default level: INFO (can be overridden by env vars in a real app)
        logger.setLevel(logging.INFO)
    
    return logger
