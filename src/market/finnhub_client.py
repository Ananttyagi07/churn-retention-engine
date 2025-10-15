"""
Finnhub Financial Metrics Client

Provides access to a variety of financial market data via Finnhub free APIs,
including macroeconomic indicators and market volatility metrics.

API Key: d2nka89r01qvm112dep0d2nka89r01qvm112depg (publicly supplied; replace with your own key)

References:
- Finnhub API docs: https://finnhub.io/docs/api
- Common endpoints: economic indicators, market indexes, volatility
"""

import requests
from typing import Dict, Any, Optional
import pandas as pd
import time
from loguru import logger


class MarketClient:
    """
    Finnhub API Client for financial and economic data enrichment.
    """

    BASE_URL = "https://finnhub.io/api/v1"
    RATE_LIMIT_SECONDS = 1.0  # Free-tier approx 60 calls per minute

    def __init__(self, api_key: str):
        """
        Initialize the Finnhub client with API key.
        
        Args:
            api_key: Finnhub API key string
        """
        self.api_key = api_key
        self._last_call_time = 0.0
        self.logger = logger

    def _rate_limit(self):
        """
        Rate limiting to avoid exceeding free tier API limits.
        """
        elapsed = time.time() - self._last_call_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            wait_time = self.RATE_LIMIT_SECONDS - elapsed
            self.logger.debug(f"Rate limiting: sleeping for {wait_time:.2f} seconds")
            time.sleep(wait_time)
        self._last_call_time = time.time()

    def _get(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Make a GET request to Finnhub API with error handling and rate limiting.

        Args:
            endpoint: API endpoint path
            params: Dictionary of query parameters

        Returns:
            JSON dict response or None on failure
        """
        self._rate_limit()
        if params is None:
            params = {}
        params['token'] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                self.logger.warning(f"Finnhub API error response: {data['error']}")
                return None

            return data

        except Exception as e:
            self.logger.error(f"Finnhub API request failed: {e}")
            return None

    def get_macro_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Retrieve common macroeconomic indicators from Finnhub.

        Returns:
            Dictionary mapping indicator name to DataFrame with time series data
        """
        macros = {}

        # US Unemployment Rate (monthly)
        unemp = self._get("economic/us/unemployment_rate")
        if unemp and "data" in unemp:
            df_unemp = pd.DataFrame(unemp["data"])
            # Convert timestamp to datetime
            df_unemp["date"] = pd.to_datetime(df_unemp["date"], unit="s")
            macros["unemployment_rate"] = df_unemp

        # US Consumer Price Index (monthly)
        cpi = self._get("economic/us/cpi")
        if cpi and "data" in cpi:
            df_cpi = pd.DataFrame(cpi["data"])
            df_cpi["date"] = pd.to_datetime(df_cpi["date"], unit="s")
            macros["cpi"] = df_cpi

        # US GDP (quarterly)
        gdp = self._get("economic/us/gdp")
        if gdp and "data" in gdp:
            df_gdp = pd.DataFrame(gdp["data"])
            df_gdp["date"] = pd.to_datetime(df_gdp["date"], unit="s")
            macros["gdp"] = df_gdp

        # Add additional macro indicators as needed

        self.logger.info(f"Retrieved {len(macros)} macroeconomic indicators from Finnhub.")
        return macros

    def get_volatility(self) -> Optional[pd.DataFrame]:
        """
        Retrieve market volatility index (VIX) time series.

        Returns:
            DataFrame of VIX daily quotes or None if unavailable
        """
        # VIX US Volatility Index symbol
        symbol = "VIX"

        # Endpoint for historical quote
        params = {"symbol": symbol, "resolution": "D", "from": "1262304000", "to": str(int(time.time()))}

        data = self._get("index/quote", params={"symbol": symbol})
        if data is None:
            self.logger.warning("Finnhub VIX data unavailable.")
            return None

        # For more detailed historical data, use /index/series endpoint if authorized
        return pd.DataFrame([data])

    def enrich_feature_frame(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        indicator_name: str = "unemployment_rate"
    ) -> pd.DataFrame:
        """
        Enrich input features DataFrame with macroeconomic data by date.

        Args:
            df: Input DataFrame with a date column
            date_column: Name of the date column in `df`
            indicator_name: Macro indicator key to join on (from get_macro_indicators)

        Returns:
            DataFrame enriched with macroeconomic features.
        """
        self.logger.info(f"Enriching features with Finnhub macro indicator '{indicator_name}'")
        macros = self.get_macro_indicators()

        if indicator_name not in macros:
            self.logger.warning(f"Macro indicator '{indicator_name}' unavailable. Skipping enrichment.")
            return df

        macro_df = macros[indicator_name].copy()
        macro_df[date_column] = pd.to_datetime(macro_df[date_column], errors='coerce')

        if date_column not in df.columns:
            self.logger.warning(f"Date column '{date_column}' missing from input DataFrame. Skipping enrichment.")
            return df

        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        enriched_df = pd.merge_asof(
            df.sort_values(date_column),
            macro_df.sort_values(date_column),
            on=date_column,
            direction="backward",
            tolerance=pd.Timedelta("30D")
        )
        self.logger.info(f"Feature frame enriched with '{indicator_name}'.")
        return enriched_df


# Example usage
if __name__ == "__main__":
    client = MarketClient(api_key="d2nka89r01qvm112dep0d2nka89r01qvm112depg")

    # Get macro indicators
    macros = client.get_macro_indicators()
    print("Available macro indicators:", list(macros.keys()))

    # Get volatility data (expect basic summary)
    vix = client.get_volatility()
    if vix is not None:
        print("VIX data sample:")
        print(vix.head())

    # Feature enrichment demonstration
    import pandas as pd
    sample_df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5)})
    enriched = client.enrich_feature_frame(sample_df, "date", indicator_name="unemployment_rate")
    print("Enriched DataFrame:")
    print(enriched.head())
