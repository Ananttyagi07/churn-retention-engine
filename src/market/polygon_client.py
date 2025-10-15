"""
Polygon.io Market Data Client

Integrates with Polygon.io to fetch real-time and historical financial market data,
including macroeconomic indicators and volatility metrics, for enriching the churn prediction model's features.

API Key: uh9wDxcrulMpMX5nb0fW6NajM4CC3wGg (replace with your key for production)

API docs: https://polygon.io/docs/
"""

import requests
import time
import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger


class MarketClient:
    """
    Polygon.io API Client for market and macroeconomic data.
    """

    BASE_URL = "https://api.polygon.io/v2"
    RATE_LIMIT_SECONDS = 1.0  # Approximate free tier limit (5 requests per second)

    def __init__(self, api_key: str):
        """
        Initialize the Polygon client.

        Args:
            api_key: Polygon.io API key string
        """
        self.api_key = api_key
        self._last_call_time = 0.0
        self.logger = logger

    def _rate_limit(self):
        """
        Enforce rate limit delays between API requests.
        """
        elapsed = time.time() - self._last_call_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            wait_time = self.RATE_LIMIT_SECONDS - elapsed
            self.logger.debug(f"Rate limiting: sleeping for {wait_time:.2f} seconds")
            time.sleep(wait_time)
        self._last_call_time = time.time()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Perform a GET request to the Polygon.io API.

        Args:
            path: Endpoint path appended to BASE_URL
            params: Query parameters dictionary

        Returns:
            JSON response as dict or None on error
        """
        self._rate_limit()
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        url = f"{self.BASE_URL}/{path}"

        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if "status" in data and data["status"] != "OK":
                self.logger.warning(f"Polygon API error or status: {data.get('status')}")
                return None

            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch data from Polygon API: {e}")
            return None

    def get_macro_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch selected macroeconomic indicators from Polygon.io.

        Returns:
            Dictionary of indicator name -> DataFrame of time series
        """
        indicators = {}

        # Example: Fetch US CPI monthly data
        cpi_data = self._get("aggs/ticker/CPI/prev")
        if cpi_data and "results" in cpi_data:
            df_cpi = pd.DataFrame(cpi_data["results"])
            if not df_cpi.empty:
                df_cpi["timestamp"] = pd.to_datetime(df_cpi["t"], unit="ms")
                indicators["CPI"] = df_cpi

        # Example: Fetch US unemployment monthly data
        unemp_data = self._get("aggs/ticker/UNEMP/prev")
        if unemp_data and "results" in unemp_data:
            df_unemp = pd.DataFrame(unemp_data["results"])
            if not df_unemp.empty:
                df_unemp["timestamp"] = pd.to_datetime(df_unemp["t"], unit="ms")
                indicators["Unemployment"] = df_unemp

        # Additional macros can be added similarly

        self.logger.info(f"Retrieved {len(indicators)} macro indicators from Polygon.io")
        return indicators

    def get_volatility(self) -> Optional[pd.DataFrame]:
        """
        Fetch market volatility index (e.g., VIX) from Polygon.io.

        Returns:
            DataFrame of volatility series or None on failure
        """
        # Example symbol for VIX
        vix_symbol = "VIX"

        # Fetch daily aggregates for last 1 year
        params = {
            "adjusted": False,
            "sort": "desc",
            "limit": 365,
        }
        vix_data = self._get(f"aggs/ticker/{vix_symbol}/range/1/day/2023-01-01/2024-01-01", params)
        if vix_data and "results" in vix_data:
            df_vix = pd.DataFrame(vix_data["results"])
            if not df_vix.empty:
                df_vix["timestamp"] = pd.to_datetime(df_vix["t"], unit="ms")
                return df_vix
        self.logger.warning("Volatility data (VIX) not available from Polygon.io")
        return None

    def enrich_feature_frame(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        indicator: str = "CPI"
    ) -> pd.DataFrame:
        """
        Enrich DataFrame with specified macroeconomic indicator retrieved from Polygon.io.

        Args:
            df: Input DataFrame with date column
            date_column: Name of column containing date
            indicator: Macro indicator key to join on

        Returns:
            DataFrame enriched with macroeconomic features
        """
        self.logger.info(f"Enriching feature frame with Polygon macro indicator '{indicator}'")
        macros = self.get_macro_indicators()

        if indicator not in macros:
            self.logger.warning(f"{indicator} data not found in Polygon response")
            return df

        macro_df = macros[indicator]

        if date_column not in df.columns:
            self.logger.warning(f"Date column '{date_column}' not found in input DataFrame")
            return df

        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        macro_df[date_column] = macro_df.get("timestamp")
        if macro_df[date_column].isnull().all():
            macro_df[date_column] = pd.to_datetime(macro_df["t"], unit="ms", errors='coerce')

        enriched_df = pd.merge_asof(
            df.sort_values(date_column),
            macro_df.sort_values(date_column),
            on=date_column,
            direction="backward",
            tolerance=pd.Timedelta("30D")
        )

        self.logger.info(f"Feature frame enriched with {indicator}")
        return enriched_df

# Example Usage
if __name__ == "__main__":
    client = MarketClient(api_key="uh9wDxcrulMpMX5nb0fW6NajM4CC3wGg")

    macros = client.get_macro_indicators()
    print("Polygon macro indicators:", list(macros.keys()))

    vix = client.get_volatility()
    if vix is not None:
        print("Volatility data sample:")
        print(vix.head())

    import pandas as pd
    df_sample = pd.DataFrame({"date": pd.date_range(start="2023-01-01", periods=5)})
    enriched = client.enrich_feature_frame(df_sample, date_column="date", indicator="CPI")
    print("Enriched DataFrame:")
    print(enriched.head())
