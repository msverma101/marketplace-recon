"""Replay observable events in processing time (docs §8.4).

Reads ONLY <run>/observable/. Serves, for each window ending at `clock`, the events with
clock - window < available_at <= clock. Raises if any served event is from the future.
"""
from __future__ import annotations

import pathlib
import time
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
import polars as pl


class FutureLeak(AssertionError):
    pass


class Replayer:
    def __init__(self, run: pathlib.Path | str):
        self.path = pathlib.Path(run) / "observable" / "events.parquet"
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.df = pl.read_parquet(self.path).sort("available_at")
        self._t = self.df["available_at"].cast(pl.Int64).to_numpy()
        self.timings: list[float] = []

    def windows(self, window: timedelta = timedelta(hours=1), start: datetime | None = None,
                end: datetime | None = None) -> Iterator[tuple[datetime, pl.DataFrame]]:
        w = int(window.total_seconds() * 1e6)
        lo = int(np.datetime64(start, "us").astype(np.int64)) if start else (self._t[0] // w) * w
        hi = int(np.datetime64(end, "us").astype(np.int64)) if end else self._t[-1] + w
        clock = lo + w
        while clock - w < hi:
            t0 = time.perf_counter()
            a, b = np.searchsorted(self._t, [clock - w, clock], side="right")
            frame = self.df.slice(int(a), int(b - a))
            if b > a and self._t[a:b].max() > clock:
                raise FutureLeak("replayer served an event with available_at > clock")
            self.timings.append(time.perf_counter() - t0)
            yield _dt(clock), frame
            clock += w


def _dt(us: int) -> datetime:
    return datetime(1970, 1, 1) + timedelta(microseconds=int(us))
