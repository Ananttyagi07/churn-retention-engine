"""
Alpha Vantage Market Data Client

Provides access to free macroeconomic indicators and market volatility metrics from Alpha Vantage API
to enrich customer churn features with external economic context.

Uses free endpoints with rate limiting considerations.

API Key: AUD8X9P81MS6Z35Z (public key provided, use own key for production)

References:
- Alpha Vantage API documentation: https://www.alphavantage.co/documentation/
- Economic indicators: GDP, CPI, Unemployment, VIX, etc.
"""

import requests
import time
from typing import Dict, Any, Optional
import pandas as pd
from loguru import logger


class MarketClient:
    """
    Alpha Vantage API Client for macroeconomic and volatility data.
    """

    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT_SECONDS = 15  # Alpha Vantage free tier limits: max 5 requests/min

    def __init__(self, api_key: str):
        """
        Initialize the client with an API key.

        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self._last_call_time = 0.0
        self.logger = logger

    def _rate_limit(self):
        """
        Ensures minimum time interval between API calls to respect free tier limits.
        """
        elapsed = time.time() - self._last_call_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            wait_time = self.RATE_LIMIT_SECONDS - elapsed
            self.logger.debug(f"Rate limiting: sleeping for {wait_time:.1f} seconds")
            time.sleep(wait_time)
        self._last_call_time = time.time()

    def _make_request(self, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make a GET request to Alpha Vantage with error handling and rate limiting.

        Args:
            params: Query parameters for the API call

        Returns:
            JSON response as dict, or None on failure
        """
        self._rate_limit()
        params['apikey'] = self.api_key
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if "Note" in data or "Error Message" in data:
                # API limit reached or invalid query
                self.logger.warning(f"Alpha Vantage API limit or error: {data.get('Note') or data.get('Error Message')}")
                return None
            return data
        except Exception as e:
            self.logger.error(f"Alpha Vantage API request failed: {e}")
            return None

    def get_macro_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Retrieve a set of macroeconomic indicators.

        Returns:
            Dictionary of DataFrames keyed by indicator name
        """
        macros = {}

        # 1. GDP (Quarterly)
        gdp_data = self._make_request({
            "function": "REAL_GDP",
            "interval": "quarterly",
            "datatype": "json",
        })
        if gdp_data and "data" in gdp_data:
            frames = pd.DataFrame(gdp_data["data"])
            macros["GDP"] = frames

        # 2. Consumer Price Index (Monthly)
        cpi_data = self._make_request({
            "function": "CPI",
            "datatype": "json",
        })
        if cpi_data and "data" in cpi_data:
            macros["CPI"] = pd.DataFrame(cpi_data["data"])

        # 3. Unemployment Rate (Monthly)
        unemp_data = self._make_request({
            "function": "UNEMPLOYMENT",
            "datatype": "json",
        })
        if unemp_data and "data" in unemp_data:
            macros["Unemployment"] = pd.DataFrame(unemp_data["data"])

        return macros

    def get_volatility(self) -> Optional[pd.DataFrame]:
        """
        Retrieve VIX volatility index data (daily).

        Returns:
            VIX timeseries DataFrame or None if unavailable
        """
        vix_data = self._make_request({
            "function": "VIX",
            "datatype": "json",
        })
        if vix_data and "data" in vix_data:
            df = pd.DataFrame(vix_data["data"])
            return df
        else:
            self.logger.warning("VIX data not available from Alpha Vantage")
            return None

    def enrich_feature_frame(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        indicator: str = "CPI"
    ) -> pd.DataFrame:
        """
        Enrich an existing customer feature DataFrame with macro indicators by joining on the date.

        Args:
            df: Customer features DataFrame with a datetime column
            date_column: Name of the datetime column in df
            indicator: Macro indicator key from get_macro_indicators()

        Returns:
            DataFrame enriched with macroeconomic indicator features
        """
        self.logger.info(f"Enriching feature frame with {indicator} data...")
        macros = self.get_macro_indicators()

        if indicator not in macros:
            self.logger.warning(f"{indicator} data not found, skipping enrichment.")
            return df

        macro_df = macros[indicator]
        if date_column not in df.columns:
            self.logger.warning(f"{date_column} not present in df; skipping enrichment.")
            return df

        macro_df[date_column] = pd.to_datetime(macro_df[date_column], errors='coerce')
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Join macro indicators by date (nearest previous date if needed)
        enriched_df = pd.merge_asof(
            df.sort_values(date_column),
            macro_df.sort_values(date_column),
            on=date_column,
            direction="backward",
            tolerance=pd.Timedelta("30D"),
        )

        self.logger.info(f"Feature frame enriched with {indicator}.")
        return enriched_df

# ============= Example Usage =============

if __name__ == "__main__":
    client = MarketClient(api_key="AUD8X9P81MS6Z35Z")
    macros = client.get_macro_indicators()
    print("Available macro indicators:", list(macros.keys()))

    vix = client.get_volatility()
    if vix is not None:
        print("VIX data sample:")
        print(vix.head())

    # Example enrichment (assuming df with a valid date column)
    import pandas as pd
    sample_df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5)})
    enriched = client.enrich_feature_frame(sample_df, "date")
    print("Enriched DataFrame:")
    print(enriched.head())
