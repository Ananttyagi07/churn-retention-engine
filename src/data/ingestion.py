"""
Data Ingestion and Integration Module

This module handles all raw data ingestion workflows, including:
1. Loading the IBM Telco Customer Churn dataset (base seed dataset)
2. Loading external data sources such as CSV, SQL, or third-party APIs
3. Integrating external market and enrichment features (Alpha Vantage, Finnhub, etc.)
4. Persisting raw and integrated datasets into standardized formats

The module is designed for extensibility and compliance with reproducible ETL best practices.

References:
- IBM Telco dataset schema [IBM Docs, 2024][web:172]
- IBM open dataset GitHub [IBM Cloud Pak][web:12]
- Market enrichment APIs [Persana AI, 2025][web:175]
- ETL pipeline practices [Rivery, 2024][web:176][web:179][web:182]
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

# Local imports
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.market.alphavantage_client import AlphaVantageClient
from src.market.finnhub_client import FinnhubClient

# ======================================================================
# Class: DataIngestor
# ======================================================================


class DataIngestor:
    """
    Data ingestion manager responsible for reading and storing datasets.

    Functionalities:
    - Load IBM Telco Customer Churn dataset for compatibility with IBM schema
    - Ingest external datasets (CSV, SQL, API)
    - Fetch market-level enrichment data via external APIs
    - Persist raw files for version-controlled reproducibility
    """

    def __init__(self, data_dir: str = "data/raw", external_dir: str = "data/external"):
        """
        Initialize the DataIngestor and its configurations.
        Args:
            data_dir: Path to raw data storage folder.
            external_dir: Path to external API/market features folder.
        """
        self.config = load_config()
        self.data_dir = Path(data_dir)
        self.external_dir = Path(external_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

        # Initialize API clients if keys are available
        self.alpha_client = None
        self.finnhub_client = None

        api_keys = self.config.api_keys
        if api_keys.alphavantage:
            self.alpha_client = AlphaVantageClient(api_keys.alphavantage)
        if api_keys.finnhub:
            self.finnhub_client = FinnhubClient(api_keys.finnhub)

        self.logger = get_logger(__name__)
        self.logger.info("DataIngestor initialized successfully.")

    # ------------------------------------------------------------------
    # Function: load_ibm_telco
    # ------------------------------------------------------------------
    def load_ibm_telco(self, source_url: Optional[str] = None) -> pd.DataFrame:
        """
        Load the official IBM Telco Customer Churn dataset.
        Compatible with IBM Watson schema [web:12][web:171][web:172][web:177].

        Args:
            source_url: Optional link to dataset CSV (fallback to Kaggle IBM version).

        Returns:
            A Pandas DataFrame containing IBM Telco churn data.
        """
        self.logger.info("Loading IBM Telco Customer Churn dataset...")

        # Default to reliable public dataset if not provided
        default_url = (
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        )
        file_url = source_url or default_url

        try:
            df = pd.read_csv(file_url)
            self.logger.info(f"Dataset loaded successfully from {file_url}")
            self.logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

            # Normalize column names for consistency
            df.columns = [c.strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]

            # Compatibility: Ensure required IBM schema columns exist
            required_cols = [
                "customerid",
                "gender",
                "seniorcitizen",
                "partner",
                "dependents",
                "tenure",
                "phoneservice",
                "multiplelines",
                "internetservice",
                "onlinesecurity",
                "onlinebackup",
                "deviceprotection",
                "techsupport",
                "streamingtv",
                "streamingmovies",
                "contract",
                "paperlessbilling",
                "paymentmethod",
                "monthlycharges",
                "totalcharges",
                "churn",
            ]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Save to standardized path
            self.persist_raw(df, name="ibm_telco_churn")

            return df

        except Exception as e:
            self.logger.error(f"Failed to load Telco dataset: {e}")
            raise

    # ------------------------------------------------------------------
    # Function: load_external_sources
    # ------------------------------------------------------------------
    def load_external_sources(self) -> Dict[str, pd.DataFrame]:
        """
        Load and integrate external data sources such as CSVs, databases,
        or financial market APIs [web:175][web:184][web:186].

        Returns:
            Dictionary of loaded DataFrames:
                {
                    "economic_indicators": <df>,
                    "market_trends": <df>
                }
        """
        self.logger.info("Loading external and market sources...")
        data_sources = {}

        # Fetch economic indicators from Alpha Vantage
        if self.alpha_client:
            try:
                market_data = self.alpha_client.get_macro_indicators()
                data_sources["economic_indicators"] = pd.DataFrame(market_data)
                self.logger.info("Economic indicators loaded via Alpha Vantage.")
            except Exception as e:
                self.logger.warning(f"Failed to fetch Alpha Vantage data: {e}")

        # Fetch market trends from Finnhub
        if self.finnhub_client:
            try:
                trend_data = self.finnhub_client.get_market_trends()
                data_sources["market_trends"] = pd.DataFrame(trend_data)
                self.logger.info("Market trends loaded via Finnhub.")
            except Exception as e:
                self.logger.warning(f"Failed to fetch Finnhub data: {e}")

        # Load additional CSVs or SQL sources (optional)
        for file in self.external_dir.glob("*.csv"):
            self.logger.debug(f"Loading external CSV: {file.name}")
            try:
                df = pd.read_csv(file)
                data_sources[file.stem] = df
                self.logger.info(f"Loaded {file.name} ({df.shape[0]} rows).")
            except Exception as e:
                self.logger.warning(f"Skipping {file.name} due to error: {e}")

        self.logger.info(f"External sources loaded: {list(data_sources.keys())}")
        return data_sources

    # ------------------------------------------------------------------
    # Function: persist_raw
    # ------------------------------------------------------------------
    def persist_raw(
        self,
        df: pd.DataFrame,
        name: str,
        format: str = "csv",
        include_timestamp: bool = True,
        subfolder: Optional[str] = None,
    ) -> Path:
        """
        Persist raw dataset to the data/raw/ directory with timestamp versioning.

        Args:
            df: DataFrame to persist.
            name: Dataset name (used for filename).
            format: File format ('csv', 'parquet').
            include_timestamp: Whether to append UTC timestamp in filename.
            subfolder: Optional subdirectory (e.g., "ibm_telco", "external").

        Returns:
            Path to saved file.

        Raises:
            ValueError: If unsupported format specified.
        """
        self.logger.info(f"Persisting dataset: {name}")

        # Determine output path
        folder = self.data_dir / (subfolder or "")
        folder.mkdir(parents=True, exist_ok=True)

        # Construct filename
        timestamp = datetime_utc() if include_timestamp else "latest"
        filename = f"{name}_{timestamp}.{format}" if include_timestamp else f"{name}.{format}"
        file_path = folder / filename

        try:
            if format == "csv":
                df.to_csv(file_path, index=False)
            elif format == "parquet":
                df.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Dataset persisted successfully: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Failed to persist dataset: {e}")
            raise


# ======================================================================
# Utility Functions
# ======================================================================


def datetime_utc() -> str:
    """Return current UTC datetime string suitable for filenames."""
    from datetime import datetime

    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


# ======================================================================
# Example Usage (Manual Testing)
# ======================================================================

if __name__ == "__main__":
    ingestor = DataIngestor()

    # Load IBM dataset
    telco_df = ingestor.load_ibm_telco()

    # Load market and external features
    ext_data = ingestor.load_external_sources()

    # Persist datasets
    for name, df in ext_data.items():
        ingestor.persist_raw(df, name=name, subfolder="external")

    print("\n✓ Ingestion pipeline executed successfully.")
