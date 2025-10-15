"""
Business Metrics Module

Computes essential Key Performance Indicators (KPIs) related to customer churn,
retention strategies, and customer lifetime value (LTV) for business impact analysis.

Key metrics:
- churn_rate: fraction of customers lost in a period
- retention_roi: return-on-investment estimate for retention campaigns
- lifetime_value: approximation of customer revenue over lifespan

These metrics enable data-driven business decisions and model evaluation.

References:
- Customer churn KPIs: IBM Telco churn analytics[web:215]
- Retention ROI calculation principles[web:224]
- LTV approximation methods[web:220][web:224]
"""

import numpy as np
import pandas as pd
from typing import Optional


def churn_rate(labels: pd.Series) -> float:
    """
    Calculate churn rate: fraction of customers who churned.

    Args:
        labels: Binary series where 1 indicates churned, 0 retained.

    Returns:
        Churn rate as float between 0 and 1.
    """
    if labels.empty:
        return 0.0

    churned = np.sum(labels == 1)
    total = len(labels)
    rate = churned / total
    return rate


def retention_roi(
    current_customers: int,
    retained_customers: int,
    avg_customer_ltv: float,
    retention_cost_per_customer: float,
    campaign_cost: float,
) -> float:
    """
    Estimate retention ROI of a campaign.

    Args:
        current_customers: Total customers targeted
        retained_customers: Customers retained due to intervention
        avg_customer_ltv: Average lifetime value (e.g., revenue) per customer
        retention_cost_per_customer: Cost to retain one customer (discounts, support)
        campaign_cost: Fixed campaign costs (marketing, operations)

    Returns:
        ROI as ratio: (Gain - Cost) / Cost
    """
    total_gain = retained_customers * avg_customer_ltv
    total_cost = (current_customers * retention_cost_per_customer) + campaign_cost
    if total_cost == 0:
        return float("inf")  # Avoid divide-by-zero; interpret as infinite ROI
    roi = (total_gain - total_cost) / total_cost
    return roi


def lifetime_value(
    avg_monthly_revenue: float,
    avg_tenure_months: float,
    gross_margin: float = 0.7,
) -> float:
    """
    Approximate Customer Lifetime Value (LTV).

    Args:
        avg_monthly_revenue: Average monthly revenue per customer
        avg_tenure_months: Average customer tenure in months
        gross_margin: Proportion of revenue retained after COGS (default 70%)

    Returns:
        LTV estimate in currency units
    """
    ltv = avg_monthly_revenue * avg_tenure_months * gross_margin
    return ltv


# =======================
# Example Usage
# =======================

if __name__ == "__main__":
    # Sample churn labels: 1 = churned, 0 = retained
    labels = pd.Series([0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    rate = churn_rate(labels)
    print(f"Churn rate: {rate:.2%}")

    # Retention ROI calculation example
    roi = retention_roi(
        current_customers=1000,
        retained_customers=120,
        avg_customer_ltv=500.0,
        retention_cost_per_customer=30.0,
        campaign_cost=10000,
    )
    print(f"Retention ROI: {roi:.2f}")

    # LTV approximation
    ltv_val = lifetime_value(avg_monthly_revenue=50.0, avg_tenure_months=24)
    print(f"Estimated Lifetime Value: ${ltv_val:.2f}")

    print("\n✓ Business metrics module demo complete.")
