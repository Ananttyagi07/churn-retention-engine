"""
Date and Time Utility Functions

Provides helper functions for date/time windows, scheduling, and common temporal calculations
used in temporal feature engineering, data slicing, and batch scheduling.

Designed to ensure consistency and clarity in time-based data processing.

Based on best practices for time series ML and production scheduling [web:219][web:225][web:241].
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple


def utcnow() -> datetime:
    """
    Return the current UTC datetime with tzinfo.

    Returns:
        Current UTC datetime
    """
    return datetime.now(tz=timezone.utc)


def datetime_to_str(dt: datetime, fmt: str = "%Y-%m-%dT%H:%M:%SZ") -> str:
    """
    Convert datetime to formatted string in UTC ISO8601 format by default.

    Args:
        dt: datetime object
        fmt: Format string

    Returns:
        Formatted datetime string
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime(fmt)


def parse_datetime(s: str, fmt: Optional[str] = None) -> datetime:
    """
    Parse string to datetime object.

    Args:
        s: datetime string
        fmt: Optional format specifier (default uses fromisoformat/time parsing)

    Returns:
        Parsed datetime object with UTC timezone if missing
    """
    if fmt:
        dt = datetime.strptime(s, fmt)
    else:
        dt = datetime.fromisoformat(s)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def add_days(dt: datetime, days: int) -> datetime:
    """
    Add given number of days to a datetime object.

    Args:
        dt: datetime object
        days: Number of days to add (+ or -)

    Returns:
        New datetime object
    """
    return dt + timedelta(days=days)


def get_time_window(
    reference: Optional[datetime] = None,
    days_back: int = 30,
) -> Tuple[datetime, datetime]:
    """
    Returns a tuple representing a time window going back `days_back`
    days from a reference date (UTC now by default).

    Args:
        reference: Reference datetime (defaults to utcnow)
        days_back: Number of days in the past

    Returns:
        (start_datetime, end_datetime) tuple
    """
    end = reference or utcnow()
    start = end - timedelta(days=days_back)
    return start, end


def is_within_window(
    timestamp: datetime, window_start: datetime, window_end: datetime
) -> bool:
    """
    Check if a timestamp lies within a (start, end) window (inclusive).

    Args:
        timestamp: datetime to check
        window_start: start of window
        window_end: end of window

    Returns:
        True if within window else False
    """
    return window_start <= timestamp <= window_end


def round_to_nearest_hour(dt: datetime) -> datetime:
    """
    Round a datetime up or down to the nearest hour.

    Args:
        dt: datetime to round

    Returns:
        Rounded datetime
    """
    if dt.minute >= 30:
        dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt


def get_next_scheduled_run(
    now: Optional[datetime] = None,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    """
    Return the next scheduled run datetime after `now`
    for daily jobs at given hour/minute/second UTC.

    Args:
        now: Current datetime (UTC by default)
        hour: Hour of day 0-23
        minute: Minute 0-59
        second: Second 0-59

    Returns:
        Datetime of next scheduled run
    """
    now = now or utcnow()
    scheduled = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if scheduled <= now:
        scheduled += timedelta(days=1)
    return scheduled


def timedelta_in_minutes(td: timedelta) -> int:
    """
    Utility to convert timedelta to total minutes integer.

    Args:
        td: timedelta object

    Returns:
        Total minutes as int
    """
    return int(td.total_seconds() // 60)


# ============= Example Usage ===============
if __name__ == "__main__":
    now = utcnow()
    print("Current UTC time:", now)
    start, end = get_time_window(days_back=7)
    print(f"7-day window: {start} to {end}")

    test_dt = now - timedelta(days=3)
    print(f"Is {test_dt} within window? {is_within_window(test_dt, start, end)}")

    next_run = get_next_scheduled_run(hour=1, minute=0)
    print("Next scheduled run:", next_run)

    rounded = round_to_nearest_hour(now)
    print("Rounded time:", rounded)
    print("Timedelta in minutes:", timedelta_in_minutes(timedelta(hours=2, minutes=30)))