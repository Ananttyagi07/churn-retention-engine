"""
Structured JSON Logging Module

This module provides a comprehensive, production-ready logging system with:
1. Structured JSON logging for machine-readable output
2. AWS OpenSearch integration for centralized log management
3. AWS CloudWatch Logs support for AWS-native logging
4. Contextual logging with correlation IDs and request tracking
5. Multiple handlers (console, file, remote)
6. Log rotation and retention policies
7. Performance-optimized with async log shipping (optional)

Author: Customer Churn & Retention Engine Team
Date: October 2025
"""

import logging
import logging.handlers
import os
import sys
import time
import traceback
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4

from pythonjsonlogger import jsonlogger


# ============================================
# Context Variables for Request Tracking
# ============================================

# Store request-specific context across async boundaries
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
request_path_ctx: ContextVar[Optional[str]] = ContextVar("request_path", default=None)


# ============================================
# Custom JSON Formatter
# ============================================


class StructuredFormatter(jsonlogger.JsonFormatter):
    """
    Enhanced JSON formatter with additional context fields.
    
    Adds:
    - Timestamp in ISO 8601 format
    - Correlation ID for request tracking
    - User ID for user action tracking
    - Environment information
    - Exception details with stack traces
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        validate: bool = True,
        *,
        defaults: Optional[Dict[str, Any]] = None,
        add_timestamp: bool = True,
        add_environment: bool = True,
    ):
        """
        Initialize structured formatter.

        Args:
            fmt: Log message format
            datefmt: Date format string
            style: Format style ('%', '{', or '$')
            validate: Whether to validate format string
            defaults: Default fields to add to every log
            add_timestamp: Whether to add ISO timestamp
            add_environment: Whether to add environment info
        """
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.add_timestamp = add_timestamp
        self.add_environment = add_environment
        self.environment = os.getenv("FLASK_ENV", "development")
        self.app_name = os.getenv("APP_NAME", "churn-retention-engine")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        """
        Add custom fields to log record.

        Args:
            log_record: Dictionary to be logged
            record: Python LogRecord
            message_dict: Message dictionary
        """
        super().add_fields(log_record, record, message_dict)

        # Add timestamp
        if self.add_timestamp:
            log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add standard fields
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno

        # Add environment information
        if self.add_environment:
            log_record["environment"] = self.environment
            log_record["app_name"] = self.app_name
            log_record["app_version"] = self.app_version

        # Add context from contextvars
        correlation_id = correlation_id_ctx.get()
        if correlation_id:
            log_record["correlation_id"] = correlation_id

        user_id = user_id_ctx.get()
        if user_id:
            log_record["user_id"] = user_id

        request_path = request_path_ctx.get()
        if request_path:
            log_record["request_path"] = request_path

        # Add exception details if present
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add process and thread info
        log_record["process_id"] = record.process
        log_record["thread_id"] = record.thread
        log_record["thread_name"] = record.threadName


# ============================================
# OpenSearch Handler
# ============================================


class OpenSearchHandler(logging.Handler):
    """
    Custom logging handler for AWS OpenSearch integration.
    
    Buffers logs and sends them to OpenSearch in batches for performance.
    Falls back gracefully if OpenSearch is unavailable.
    """

    def __init__(
        self,
        hosts: list,
        index_prefix: str = "churn-logs",
        auth: Optional[tuple] = None,
        use_ssl: bool = True,
        verify_certs: bool = True,
        buffer_size: int = 100,
        flush_interval: float = 5.0,
    ):
        """
        Initialize OpenSearch handler.

        Args:
            hosts: List of OpenSearch host URLs
            index_prefix: Prefix for log indices
            auth: Tuple of (username, password)
            use_ssl: Whether to use SSL
            verify_certs: Whether to verify SSL certificates
            buffer_size: Number of logs to buffer before flushing
            flush_interval: Seconds between automatic flushes
        """
        super().__init__()
        self.hosts = hosts
        self.index_prefix = index_prefix
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.last_flush = time.time()

        try:
            from opensearchpy import OpenSearch

            self.client = OpenSearch(
                hosts=hosts,
                http_auth=auth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
                connection_class=None,
            )
            self.available = True
        except Exception as e:
            print(f"Warning: OpenSearch not available: {e}", file=sys.stderr)
            self.available = False

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to OpenSearch (buffered).

        Args:
            record: Log record to emit
        """
        if not self.available:
            return

        try:
            log_entry = self.format(record)
            self.buffer.append(log_entry)

            # Flush if buffer is full or interval exceeded
            if len(self.buffer) >= self.buffer_size or (
                time.time() - self.last_flush >= self.flush_interval
            ):
                self.flush()

        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        """Flush buffered logs to OpenSearch."""
        if not self.buffer or not self.available:
            return

        try:
            from opensearchpy import helpers

            # Prepare bulk index operations
            actions = [
                {
                    "_index": f"{self.index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}",
                    "_source": log,
                }
                for log in self.buffer
            ]

            # Bulk index
            helpers.bulk(self.client, actions)

            # Clear buffer
            self.buffer.clear()
            self.last_flush = time.time()

        except Exception as e:
            print(f"Error flushing logs to OpenSearch: {e}", file=sys.stderr)

    def close(self) -> None:
        """Close handler and flush remaining logs."""
        self.flush()
        super().close()


# ============================================
# CloudWatch Handler
# ============================================


class CloudWatchHandler(logging.Handler):
    """
    Custom logging handler for AWS CloudWatch Logs integration.
    
    Sends logs to CloudWatch with automatic log stream management.
    Falls back gracefully if CloudWatch is unavailable.
    """

    def __init__(
        self,
        log_group: str,
        log_stream: Optional[str] = None,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Initialize CloudWatch handler.

        Args:
            log_group: CloudWatch log group name
            log_stream: CloudWatch log stream name (auto-generated if None)
            region_name: AWS region
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
        """
        super().__init__()
        self.log_group = log_group
        self.log_stream = log_stream or f"{os.uname().nodename}-{uuid4().hex[:8]}"
        self.region_name = region_name

        try:
            import boto3

            self.client = boto3.client(
                "logs",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

            # Create log group and stream if they don't exist
            self._ensure_log_group_exists()
            self._ensure_log_stream_exists()
            self.available = True

        except Exception as e:
            print(f"Warning: CloudWatch not available: {e}", file=sys.stderr)
            self.available = False

    def _ensure_log_group_exists(self) -> None:
        """Ensure CloudWatch log group exists."""
        try:
            self.client.create_log_group(logGroupName=self.log_group)
        except self.client.exceptions.ResourceAlreadyExistsException:
            pass

    def _ensure_log_stream_exists(self) -> None:
        """Ensure CloudWatch log stream exists."""
        try:
            self.client.create_log_stream(
                logGroupName=self.log_group, logStreamName=self.log_stream
            )
        except self.client.exceptions.ResourceAlreadyExistsException:
            pass

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to CloudWatch.

        Args:
            record: Log record to emit
        """
        if not self.available:
            return

        try:
            log_entry = {
                "timestamp": int(record.created * 1000),  # Milliseconds
                "message": self.format(record),
            }

            self.client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
                logEvents=[log_entry],
            )

        except Exception:
            self.handleError(record)


# ============================================
# Logger Setup Function
# ============================================


def setup_logger(
    name: str = __name__,
    level: Union[int, str] = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_file_path: Optional[Path] = None,
    log_to_opensearch: bool = False,
    opensearch_config: Optional[Dict[str, Any]] = None,
    log_to_cloudwatch: bool = False,
    cloudwatch_config: Optional[Dict[str, Any]] = None,
    json_format: bool = True,
    rotation_max_bytes: int = 10 * 1024 * 1024,  # 10MB
    rotation_backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a comprehensive structured logger with multiple handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Enable console logging
        log_to_file: Enable file logging
        log_file_path: Path to log file (default: data/logs/app.log)
        log_to_opensearch: Enable OpenSearch logging
        opensearch_config: OpenSearch configuration dict
        log_to_cloudwatch: Enable CloudWatch logging
        cloudwatch_config: CloudWatch configuration dict
        json_format: Use JSON formatting
        rotation_max_bytes: Max log file size before rotation
        rotation_backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(
        ...     name="churn_engine",
        ...     level=logging.INFO,
        ...     log_to_opensearch=True,
        ...     opensearch_config={
        ...         "hosts": ["https://localhost:9200"],
        ...         "auth": ("admin", "admin"),
        ...     }
        ... )
        >>> logger.info("Application started", extra={"user_id": "12345"})
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers

    # Create formatter
    if json_format:
        formatter = StructuredFormatter(
            fmt="%(timestamp)s %(level)s %(name)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_to_file:
        log_file_path = log_file_path or Path("data/logs/app.log")
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=rotation_max_bytes,
            backupCount=rotation_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # OpenSearch handler
    if log_to_opensearch and opensearch_config:
        try:
            opensearch_handler = OpenSearchHandler(**opensearch_config)
            opensearch_handler.setLevel(level)
            opensearch_handler.setFormatter(formatter)
            logger.addHandler(opensearch_handler)
        except Exception as e:
            logger.warning(f"Failed to setup OpenSearch handler: {e}")

    # CloudWatch handler
    if log_to_cloudwatch and cloudwatch_config:
        try:
            cloudwatch_handler = CloudWatchHandler(**cloudwatch_config)
            cloudwatch_handler.setLevel(level)
            cloudwatch_handler.setFormatter(formatter)
            logger.addHandler(cloudwatch_handler)
        except Exception as e:
            logger.warning(f"Failed to setup CloudWatch handler: {e}")

    return logger


# ============================================
# Context Manager for Request Tracking
# ============================================


class LogContext:
    """
    Context manager for setting request-specific logging context.
    
    Usage:
        with LogContext(correlation_id="abc123", user_id="user456"):
            logger.info("Processing request")  # Will include context
    """

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_path: Optional[str] = None,
    ):
        """
        Initialize log context.

        Args:
            correlation_id: Unique ID for request tracking
            user_id: User identifier
            request_path: Request path or operation name
        """
        self.correlation_id = correlation_id or str(uuid4())
        self.user_id = user_id
        self.request_path = request_path
        self.tokens = []

    def __enter__(self):
        """Set context variables."""
        self.tokens.append(correlation_id_ctx.set(self.correlation_id))
        if self.user_id:
            self.tokens.append(user_id_ctx.set(self.user_id))
        if self.request_path:
            self.tokens.append(request_path_ctx.set(self.request_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Reset context variables."""
        for token in self.tokens:
            token.var.reset(token)


# ============================================
# Convenience Functions
# ============================================


def get_logger(name: str = __name__, **kwargs) -> logging.Logger:
    """
    Get or create a logger with default configuration.

    Args:
        name: Logger name
        **kwargs: Additional setup_logger arguments

    Returns:
        Configured logger instance
    """
    return setup_logger(name=name, **kwargs)


def log_performance(logger: logging.Logger, operation: str):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance
        operation: Operation name for logging

    Example:
        >>> @log_performance(logger, "model_training")
        >>> def train_model():
        ...     pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"{operation} completed",
                    extra={"operation": operation, "elapsed_time": elapsed},
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(
                    f"{operation} failed",
                    extra={
                        "operation": operation,
                        "elapsed_time": elapsed,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    """
    Example usage and testing of logging system.
    """
    # Basic logger setup
    logger = setup_logger(
        name="churn_engine",
        level=logging.DEBUG,
        log_to_console=True,
        log_to_file=True,
        log_file_path=Path("data/logs/test.log"),
    )

    # Basic logging
    logger.debug("Debug message")
    logger.info("Application started successfully")
    logger.warning("This is a warning")

    # Structured logging with extra fields
    logger.info(
        "User action",
        extra={"user_id": "user123", "action": "login", "ip_address": "192.168.1.1"},
    )

    # Context-aware logging
    with LogContext(correlation_id="req-12345", user_id="user456", request_path="/api/predict"):
        logger.info("Processing prediction request")
        logger.info("Model loaded successfully", extra={"model_version": "v2.1.0"})

    # Exception logging
    try:
        raise ValueError("Example error for testing")
    except Exception:
        logger.error("An error occurred", exc_info=True)

    print("\n✓ Logging examples complete! Check data/logs/test.log")
