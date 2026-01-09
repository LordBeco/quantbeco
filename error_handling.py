"""
Error Handling Module

Comprehensive error handling classes and utilities for the AI Strategy Builder 
and Backtesting Engine components.
"""

import logging
import traceback
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import time
import random


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    DATA_ERROR = "data_error"
    STRATEGY_ERROR = "strategy_error"
    EXECUTION_ERROR = "execution_error"
    CONFIGURATION_ERROR = "configuration_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorDetails:
    """Detailed error information"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    technical_details: Optional[str] = None
    user_message: Optional[str] = None
    suggestions: List[str] = None
    error_code: Optional[str] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.timestamp is None:
            self.timestamp = time.time()


class AIStrategyBuilderError(Exception):
    """Base exception for AI Strategy Builder errors"""
    
    def __init__(self, error_details: ErrorDetails):
        self.error_details = error_details
        super().__init__(error_details.message)


class BacktestingEngineError(Exception):
    """Base exception for Backtesting Engine errors"""
    
    def __init__(self, error_details: ErrorDetails):
        self.error_details = error_details
        super().__init__(error_details.message)


class DataProcessingError(Exception):
    """Exception for data processing errors"""
    
    def __init__(self, error_details: ErrorDetails):
        self.error_details = error_details
        super().__init__(error_details.message)


class APIError(Exception):
    """Exception for API-related errors"""
    
    def __init__(self, error_details: ErrorDetails):
        self.error_details = error_details
        super().__init__(error_details.message)


class RetryableError(Exception):
    """Exception for errors that can be retried"""
    
    def __init__(self, error_details: ErrorDetails, max_retries: int = 3):
        self.error_details = error_details
        self.max_retries = max_retries
        super().__init__(error_details.message)


class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self, logger_name: str = "trade_analyzer_pro"):
        self.logger = logging.getLogger(logger_name)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorDetails:
        """Handle and log errors with context"""
        if isinstance(error, (AIStrategyBuilderError, BacktestingEngineError, 
                             DataProcessingError, APIError)):
            error_details = error.error_details
        else:
            # Create error details for unexpected errors
            error_details = ErrorDetails(
                category=ErrorCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.ERROR,
                message=str(error),
                technical_details=traceback.format_exc(),
                user_message="An unexpected error occurred. Please try again or contact support."
            )
        
        # Log the error
        self._log_error(error_details, context)
        
        return error_details
    
    def _log_error(self, error_details: ErrorDetails, context: Optional[Dict[str, Any]] = None):
        """Log error with appropriate level"""
        log_message = f"[{error_details.category.value}] {error_details.message}"
        
        if context:
            log_message += f" | Context: {context}"
        
        if error_details.technical_details:
            log_message += f" | Technical: {error_details.technical_details}"
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_details.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
        elif error_details.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)


class RetryHandler:
    """Handles retry logic with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.error_handler = ErrorHandler()
    
    def retry_with_backoff(self, func, *args, retryable_exceptions=None, **kwargs):
        """Execute function with exponential backoff retry logic"""
        if retryable_exceptions is None:
            retryable_exceptions = (RetryableError, ConnectionError, TimeoutError)
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    # Final attempt failed
                    error_details = ErrorDetails(
                        category=ErrorCategory.API_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message=f"Operation failed after {self.max_retries} retries: {str(e)}",
                        technical_details=traceback.format_exc(),
                        user_message="The operation failed after multiple attempts. Please check your connection and try again.",
                        suggestions=[
                            "Check your internet connection",
                            "Verify API credentials",
                            "Try again in a few minutes"
                        ]
                    )
                    raise APIError(error_details)
                
                # Calculate delay with jitter
                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay
                )
                jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
                total_delay = delay + jitter
                
                self.error_handler.logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {total_delay:.2f}s: {str(e)}"
                )
                
                time.sleep(total_delay)
            except Exception as e:
                # Non-retryable exception
                error_details = self.error_handler.handle_error(e, {
                    'function': func.__name__,
                    'attempt': attempt + 1
                })
                raise e
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception


class ValidationErrorCollector:
    """Collects and manages validation errors"""
    
    def __init__(self):
        self.errors: List[ErrorDetails] = []
        self.warnings: List[ErrorDetails] = []
    
    def add_error(self, message: str, category: ErrorCategory = ErrorCategory.VALIDATION_ERROR,
                  suggestions: List[str] = None, error_code: str = None):
        """Add a validation error"""
        error_details = ErrorDetails(
            category=category,
            severity=ErrorSeverity.ERROR,
            message=message,
            user_message=message,
            suggestions=suggestions or [],
            error_code=error_code
        )
        self.errors.append(error_details)
    
    def add_warning(self, message: str, category: ErrorCategory = ErrorCategory.VALIDATION_ERROR,
                    suggestions: List[str] = None):
        """Add a validation warning"""
        warning_details = ErrorDetails(
            category=category,
            severity=ErrorSeverity.WARNING,
            message=message,
            user_message=message,
            suggestions=suggestions or []
        )
        self.warnings.append(warning_details)
    
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0
    
    def get_error_summary(self) -> str:
        """Get a summary of all errors"""
        if not self.has_errors():
            return "No errors found"
        
        summary = f"Found {len(self.errors)} error(s):\n"
        for i, error in enumerate(self.errors, 1):
            summary += f"{i}. {error.message}\n"
            if error.suggestions:
                summary += f"   Suggestions: {', '.join(error.suggestions)}\n"
        
        return summary
    
    def get_warning_summary(self) -> str:
        """Get a summary of all warnings"""
        if not self.has_warnings():
            return "No warnings found"
        
        summary = f"Found {len(self.warnings)} warning(s):\n"
        for i, warning in enumerate(self.warnings, 1):
            summary += f"{i}. {warning.message}\n"
            if warning.suggestions:
                summary += f"   Suggestions: {', '.join(warning.suggestions)}\n"
        
        return summary
    
    def clear(self):
        """Clear all errors and warnings"""
        self.errors.clear()
        self.warnings.clear()


def create_user_friendly_error(error: Exception, context: str = "") -> str:
    """Create user-friendly error message from exception"""
    if isinstance(error, (AIStrategyBuilderError, BacktestingEngineError, 
                         DataProcessingError, APIError)):
        error_details = error.error_details
        message = error_details.user_message or error_details.message
        
        if error_details.suggestions:
            message += "\n\nSuggestions:\n"
            for suggestion in error_details.suggestions:
                message += f"• {suggestion}\n"
        
        return message
    
    # Generic error handling
    error_message = str(error)
    
    # Common error patterns and user-friendly messages
    if "connection" in error_message.lower():
        return "Connection error. Please check your internet connection and try again."
    elif "timeout" in error_message.lower():
        return "Request timed out. Please try again in a few moments."
    elif "api" in error_message.lower() and "key" in error_message.lower():
        return "API key error. Please check your API configuration."
    elif "permission" in error_message.lower() or "unauthorized" in error_message.lower():
        return "Permission denied. Please check your credentials and permissions."
    elif "not found" in error_message.lower():
        return "Resource not found. Please check your input and try again."
    else:
        return f"An error occurred: {error_message}"


def safe_execute(func, default_value=None, error_handler: ErrorHandler = None):
    """Safely execute a function and return default value on error"""
    try:
        return func()
    except Exception as e:
        if error_handler:
            error_handler.handle_error(e)
        return default_value