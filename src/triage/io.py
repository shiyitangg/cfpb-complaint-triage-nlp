"""Chunked reading of the raw CFPB complaints export."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

COL_DATE = "Date received"
COL_RESPONSE = "Company response to consumer"
COL_NARRATIVE = "Consumer complaint narrative"
COL_ID = "Complaint ID"


def read_header(path: str | Path) -> list[str]:
    path = Path(path)
    compression = "zip" if path.suffix == ".zip" else "infer"
    return list(pd.read_csv(path, nrows=0, compression=compression).columns)


def iter_chunks(path: str | Path, usecols: Sequence[str], chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Yield chunks of the raw export. Accepts the .csv or the downloaded .csv.zip directly."""
    path = Path(path)
    compression = "zip" if path.suffix == ".zip" else "infer"
    available = read_header(path)
    missing = [c for c in usecols if c not in available]
    if missing:
        raise ValueError(f"columns not found in {path.name}: {missing}. Available columns: {available}")
    reader = pd.read_csv(
        path,
        usecols=list(usecols),
        dtype=str,
        chunksize=chunksize,
        compression=compression,
        keep_default_na=False,
        na_values=[""],
    )
    for chunk in reader:
        yield chunk


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse Date received. Exports have used both YYYY-MM-DD and MM/DD/YY, so allow mixed formats."""
    return pd.to_datetime(s, errors="coerce", format="mixed")
