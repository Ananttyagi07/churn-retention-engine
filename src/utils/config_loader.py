"""
Configuration Loader Module

This module provides a robust configuration loading system that:
1. Loads YAML configuration files (base + environment-specific)
2. Merges configurations with proper precedence
3. Validates required configuration keys
4. Supports environment variable substitution
5. Provides type-safe configuration access with Pydantic

Author: Customer Churn & Retention Engine Team
Date: October 2025
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================
# Configuration Models (Pydantic)
# ============================================


class DatabaseConfig(BaseModel):
    """Database connection configuration"""

    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=100)
    pool_timeout: int = Field(default=30, ge=5, le=300)
    echo: bool = Field(default=False, description="Enable SQL query logging")


class RedisConfig(BaseModel):
    """Redis cache configuration"""

    url: str = Field(..., description="Redis connection URL")
    password: Optional[str] = Field(default=None)
    db: int = Field(default=0, ge=0, le=15)
    default_timeout: int = Field(default=300, ge=0)
    key_prefix: str = Field(default="churn_")


class APIKeysConfig(BaseModel):
    """External API keys configuration"""

    alphavantage: Optional[str] = Field(default=None, alias="alphavantage_api_key")
    finnhub: Optional[str] = Field(default=None, alias="finnhub_api_key")
    polygon: Optional[str] = Field(default=None, alias="polygon_api_key")


class ModelConfig(BaseModel):
    """Machine learning model configuration"""

    version: str = Field(default="latest")
    registry_path: Path = Field(default=Path("data/models"))
    enable_caching: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, ge=0)
    enable_shap: bool = Field(default=True)
    shap_cache_path: Path = Field(default=Path(".shap_cache"))


class TrainingConfig(BaseModel):
    """Model training configuration"""

    test_size: float = Field(default=0.2, ge=0.0, le=0.5)
    validation_size: float = Field(default=0.1, ge=0.0, le=0.5)
    random_state: int = Field(default=42)
    cross_validation_folds: int = Field(default=5, ge=2, le=10)
    primary_metric: str = Field(default="roc_auc")
    minimum_recall: float = Field(default=0.75, ge=0.0, le=1.0)
    enable_hyperparameter_tuning: bool = Field(default=True)


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = Field(default="INFO")
    format: str = Field(default="json")
    file_path: Optional[Path] = Field(default=Path("data/logs/app.log"))
    max_bytes: int = Field(default=10485760)  # 10MB
    backup_count: int = Field(default=5)

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration"""

    opensearch_enabled: bool = Field(default=False)
    opensearch_host: Optional[str] = Field(default=None)
    opensearch_port: int = Field(default=443)
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    sentry_enabled: bool = Field(default=False)
    sentry_dsn: Optional[str] = Field(default=None)


class AppConfig(BaseSettings):
    """
    Main application configuration.
    Loads from YAML files and environment variables.
    Environment variables take precedence over YAML values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Settings
    app_name: str = Field(default="Customer Churn & Retention Engine")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    secret_key: str = Field(..., description="Flask secret key")

    # Server Settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=5000, ge=1024, le=65535)
    workers: int = Field(default=4, ge=1, le=16)

    # Configuration Sections
    database: DatabaseConfig
    redis: RedisConfig
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Feature Flags
    enable_market_enrichment: bool = Field(default=True)
    enable_batch_predictions: bool = Field(default=True)
    enable_retention_strategies: bool = Field(default=True)
    enable_drift_detection: bool = Field(default=True)


# ============================================
# Configuration Loader Class
# ============================================


class ConfigLoader:
    """
    Comprehensive configuration loader with YAML merging and validation.

    Features:
    - Load base + environment-specific YAML files
    - Deep merge configurations with proper precedence
    - Environment variable substitution in YAML values
    - Pydantic validation for type safety
    - Required key validation
    """

    # Environment variable pattern: ${VAR_NAME} or ${VAR_NAME:default_value}
    ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

    def __init__(self, config_dir: str = "config", environment: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (dev, staging, production).
                        If None, reads from FLASK_ENV or defaults to 'development'
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("FLASK_ENV", "development")
        self._config_cache: Optional[Dict[str, Any]] = None

    def load(self, use_cache: bool = True) -> AppConfig:
        """
        Load and validate complete configuration.

        Args:
            use_cache: Use cached configuration if available

        Returns:
            Validated AppConfig instance

        Raises:
            FileNotFoundError: If configuration files not found
            ValidationError: If configuration validation fails
            ValueError: If required keys are missing
        """
        if use_cache and self._config_cache is not None:
            return self._create_app_config(self._config_cache)

        # Step 1: Load base configuration
        base_config = self._load_yaml_file("base.yaml")

        # Step 2: Load environment-specific configuration
        env_config = self._load_yaml_file(f"{self.environment}.yaml", required=False)

        # Step 3: Merge configurations (env overrides base)
        merged_config = self._deep_merge(base_config, env_config or {})

        # Step 4: Substitute environment variables in values
        resolved_config = self._resolve_env_vars(merged_config)

        # Step 5: Validate required keys
        self._validate_required_keys(resolved_config)

        # Step 6: Cache the configuration
        self._config_cache = resolved_config

        # Step 7: Create and validate Pydantic model
        return self._create_app_config(resolved_config)

    def _load_yaml_file(self, filename: str, required: bool = True) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            filename: Name of YAML file
            required: Whether file is required to exist

        Returns:
            Dictionary containing YAML data

        Raises:
            FileNotFoundError: If required file doesn't exist
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            if required:
                raise FileNotFoundError(
                    f"Required configuration file not found: {filepath}"
                )
            return {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filepath}: {e}") from e

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries with override taking precedence.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged dictionary
        """
        merged = base.copy()

        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                merged[key] = self._deep_merge(merged[key], value)
            else:
                # Override value
                merged[key] = value

        return merged

    def _resolve_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve environment variables in configuration values.

        Supports syntax: ${VAR_NAME} or ${VAR_NAME:default_value}

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with resolved environment variables
        """
        resolved = {}

        for key, value in config.items():
            if isinstance(value, dict):
                # Recursively resolve nested dictionaries
                resolved[key] = self._resolve_env_vars(value)
            elif isinstance(value, str):
                # Resolve environment variables in string values
                resolved[key] = self._substitute_env_var(value)
            elif isinstance(value, list):
                # Resolve environment variables in list items
                resolved[key] = [
                    self._substitute_env_var(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                # Keep other types as-is
                resolved[key] = value

        return resolved

    def _substitute_env_var(self, value: str) -> str:
        """
        Substitute environment variables in a string value.

        Args:
            value: String that may contain ${VAR} patterns

        Returns:
            String with environment variables substituted
        """

        def replacer(match: re.Match) -> str:
            """Replace matched environment variable"""
            var_expr = match.group(1)

            # Check for default value syntax: VAR_NAME:default
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                var_name = var_expr.strip()
                env_value = os.getenv(var_name)
                if env_value is None:
                    raise ValueError(
                        f"Environment variable '{var_name}' not found and no default provided"
                    )
                return env_value

        return self.ENV_VAR_PATTERN.sub(replacer, value)

    def _validate_required_keys(self, config: Dict[str, Any]) -> None:
        """
        Validate that required configuration keys are present.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If required keys are missing
        """
        required_keys = [
            ("secret_key", "Application secret key is required"),
            ("database", "Database configuration is required"),
            ("database.url", "Database URL is required"),
        ]

        missing_keys = []

        for key_path, error_message in required_keys:
            keys = key_path.split(".")
            current = config

            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    missing_keys.append(error_message)
                    break
                current = current[key]

        if missing_keys:
            raise ValueError(
                f"Configuration validation failed:\n" + "\n".join(missing_keys)
            )

    def _create_app_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """
        Create and validate AppConfig from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Validated AppConfig instance

        Raises:
            ValidationError: If configuration validation fails
        """
        try:
            return AppConfig(**config_dict)
        except ValidationError as e:
            raise ValueError(
                f"Configuration validation failed:\n{self._format_validation_error(e)}"
            ) from e

    def _format_validation_error(self, error: ValidationError) -> str:
        """
        Format Pydantic validation error for better readability.

        Args:
            error: Pydantic ValidationError

        Returns:
            Formatted error message
        """
        lines = []
        for err in error.errors():
            location = " -> ".join(str(loc) for loc in err["loc"])
            message = err["msg"]
            lines.append(f"  • {location}: {message}")
        return "\n".join(lines)

    def reload(self) -> AppConfig:
        """
        Force reload configuration from files (bypasses cache).

        Returns:
            Freshly loaded AppConfig instance
        """
        self._config_cache = None
        return self.load(use_cache=False)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.

        Args:
            section: Name of configuration section

        Returns:
            Configuration section dictionary

        Raises:
            KeyError: If section doesn't exist
        """
        config = self.load()
        if hasattr(config, section):
            section_config = getattr(config, section)
            if isinstance(section_config, BaseModel):
                return section_config.model_dump()
            return section_config
        raise KeyError(f"Configuration section '{section}' not found")


# ============================================
# Convenience Functions
# ============================================


def load_config(
    config_dir: str = "config", environment: Optional[str] = None
) -> AppConfig:
    """
    Convenience function to load configuration.

    Args:
        config_dir: Directory containing configuration files
        environment: Environment name (dev, staging, production)

    Returns:
        Validated AppConfig instance

    Example:
        >>> config = load_config()
        >>> print(config.database.url)
        >>> print(config.model.version)
    """
    loader = ConfigLoader(config_dir=config_dir, environment=environment)
    return loader.load()


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a specific configuration value by dot-notation path.

    Args:
        key_path: Dot-separated path to configuration value (e.g., 'database.pool_size')
        default: Default value if key doesn't exist

    Returns:
        Configuration value or default

    Example:
        >>> pool_size = get_config_value('database.pool_size', default=10)
        >>> api_key = get_config_value('api_keys.alphavantage')
    """
    try:
        config = load_config()
        keys = key_path.split(".")
        current = config

        for key in keys:
            if isinstance(current, BaseModel):
                current = getattr(current, key)
            elif isinstance(current, dict):
                current = current[key]
            else:
                return default

        return current
    except (KeyError, AttributeError):
        return default


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    """
    Example usage and testing of configuration loader.
    """
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()

        # Display configuration
        print(f"\n✓ Configuration loaded successfully!")
        print(f"  Environment: {config.environment}")
        print(f"  App Name: {config.app_name}")
        print(f"  Debug Mode: {config.debug}")
        print(f"  Database URL: {config.database.url}")
        print(f"  Redis URL: {config.redis.url}")
        print(f"  Model Version: {config.model.version}")
        print(f"  Log Level: {config.logging.level}")

        # Test section access
        print("\n✓ Database configuration:")
        db_config = config.database.model_dump()
        for key, value in db_config.items():
            print(f"  {key}: {value}")

        # Test convenience function
        pool_size = get_config_value("database.pool_size", default=5)
        print(f"\n✓ Database pool size: {pool_size}")

    except Exception as e:
        print(f"\n✗ Configuration loading failed:")
        print(f"  {str(e)}")
