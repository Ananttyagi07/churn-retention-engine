"""
File I/O Utilities Module

Provides robust, flexible file input/output utilities supporting:
- Reading and writing CSV and Parquet files
- Transparent support for local disk and S3-compatible object storage
- Automatic compression handling (gzip, snappy, etc.)
- Efficient file operations with logging and error handling

Designed for data persistence and sharing in ML pipelines or production[web:185][web:210][web:215].
"""

import os
from typing import Optional, Union
from pathlib import Path
import logging

import pandas as pd
import s3fs

logger = logging.getLogger(__name__)

# ================================
# Utility Functions
# ================================

def _is_s3_path(path: Union[str, Path]) -> bool:
    """
    Checks if given path is an S3 URI.

    Args:
        path: File path or URI string

    Returns:
        True if S3 URI, else False
    """
    if isinstance(path, Path):
        path = str(path)
    return path.lower().startswith("s3://")

def _get_s3_filesystem() -> s3fs.S3FileSystem:
    """
    Get or create an S3FileSystem instance using environment-based credentials.

    Returns:
        s3fs.S3FileSystem instance
    """
    return s3fs.S3FileSystem(anon=False)

# ================================
# CSV Operations
# ================================

def read_csv(
    filepath: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Read CSV file from local or S3 with optional arguments passed to pandas.

    Args:
        filepath: Path to CSV file or S3 URI
        kwargs: Additional kwargs to pd.read_csv (chunksize, dtype, parse_dates, etc.)

    Returns:
        Loaded pandas DataFrame

    Raises:
        FileNotFoundError or IOError on failure
    """
    if _is_s3_path(filepath):
        fs = _get_s3_filesystem()
        with fs.open(filepath, "rb") as f:
            df = pd.read_csv(f, **kwargs)
    else:
        df = pd.read_csv(filepath, **kwargs)

    logger.info(f"CSV loaded: {filepath} (shape={df.shape})")
    return df

def write_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
    compression: Optional[str] = None,
    **kwargs
) -> None:
    """
    Write DataFrame to CSV on local or S3 with optional compression.

    Args:
        df: DataFrame to write
        filepath: Local or S3 path
        index: Whether to write the index
        compression: Compression to use ('gzip', 'xz', 'zip', 'bz2', None)
        kwargs: Additional args to pd.to_csv

    Raises:
        IOError on failure
    """
    if _is_s3_path(filepath):
        fs = _get_s3_filesystem()
        with fs.open(filepath, "wb") as f:
            df.to_csv(f, index=index, compression=compression, **kwargs)
    else:
        df.to_csv(filepath, index=index, compression=compression, **kwargs)

    logger.info(f"CSV written: {filepath} (rows={len(df)})")

# ================================
# Parquet Operations
# ================================

def read_parquet(
    filepath: Union[str, Path],
    engine: str = "pyarrow",
    **kwargs,
) -> pd.DataFrame:
    """
    Read Parquet file from local or S3.

    Args:
        filepath: Path or URI
        engine: Parquet engine ('pyarrow' or 'fastparquet')
        kwargs: Additional kwargs passed to pd.read_parquet

    Returns:
        DataFrame loaded from Parquet
    """
    if _is_s3_path(filepath):
        fs = _get_s3_filesystem()
        with fs.open(filepath, "rb") as f:
            df = pd.read_parquet(f, engine=engine, **kwargs)
    else:
        df = pd.read_parquet(filepath, engine=engine, **kwargs)

    logger.info(f"Parquet loaded: {filepath} (shape={df.shape})")
    return df

def write_parquet(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    compression: str = "snappy",
    engine: str = "pyarrow",
    index: bool = False,
    **kwargs,
) -> None:
    """
    Write DataFrame to Parquet file on local or S3.

    Args:
        df: DataFrame to save
        filepath: Destination path/URI
        compression: Compression codec ('snappy', 'gzip', 'brotli', None)
        engine: Parquet engine to use
        index: Whether to write index
        kwargs: Additional arguments to to_parquet function
    """
    if _is_s3_path(filepath):
        fs = _get_s3_filesystem()
        with fs.open(filepath, "wb") as f:
            df.to_parquet(f, compression=compression, engine=engine, index=index, **kwargs)
    else:
        df.to_parquet(filepath, compression=compression, engine=engine, index=index, **kwargs)

    logger.info(f"Parquet written: {filepath} (rows={len(df)})")

# ================================
# File Existence and Removal Utilities
# ================================

def exists(filepath: Union[str, Path]) -> bool:
    """
    Check if file exists (local or S3).

    Args:
        filepath: Path or URI

    Returns:
        True if exists, False otherwise
    """
    if _is_s3_path(filepath):
        fs = _get_s3_filesystem()
        return fs.exists(filepath)
    else:
        return os.path.exists(filepath)

def remove_file(filepath: Union[str, Path]) -> None:
    """
    Remove file (local or S3).

    Args:
        filepath: Path or URI to remove

    Raises:
        FileNotFoundError if file does not exist
    """
    if _is_s3_path(filepath):
        fs = _get_s3_filesystem()
        if fs.exists(filepath):
            fs.rm(filepath)
            logger.info(f"Removed S3 file: {filepath}")
        else:
            raise FileNotFoundError(f"S3 file not found: {filepath}")
    else:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Removed local file: {filepath}")
        else:
            raise FileNotFoundError(f"File not found: {filepath}")

# ================================
# Example Usage
# ================================

if __name__ == "__main__":
    # Local CSV read/write test
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    write_csv(df, "data/test.csv")
    df_loaded = read_csv("data/test.csv")
    print(f"Loaded CSV: {df_loaded}")

    # Local Parquet read/write test
    write_parquet(df, "data/test.parquet")
    df_loaded_pq = read_parquet("data/test.parquet")
    print(f"Loaded Parquet: {df_loaded_pq}")

    # S3 read/write samples (requires credentials)
    # s3_path = "s3://your-bucket-name/path/test.csv"
    # write_csv(df, s3_path)
    # loaded_s3_df = read_csv(s3_path)
    # print(f"Loaded CSV from S3: {loaded_s3_df}")
