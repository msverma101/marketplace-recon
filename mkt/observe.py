"""Observation layer: world events -> what a monitor can see (docs §7, decision L3).

Per stream: a logging delay (event_time -> available_at), a small duplication rate, and
optional measurement interventions (drop / delay events in a window). Entity IDs are
re-keyed with a keyed hash so observable IDs cannot be joined to the truth store without
the world key (leakage guard, docs §9).
"""
from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

from . import rng
from .contracts import Intervention, conform
from .interventions import selector_mask

# delay model per stream: (median seconds, log-sigma). Assumption A-11.
DEFAULT_DELAYS = {
    "web": (20, 1.0), "payments": (60, 1.0), "finance": (6 * 3600, 0.5), "logistics": (2 * 3600, 1.0),
    "reviews": (3600, 1.0), "catalog": (300, 1.0), "telemetry": (30, 0.5), "ops": (0, 0.0),
}
DUP_RATE = 0.001   # A-11: at-least-once delivery duplicates
ID_COLS = ["buyer_id", "session_id", "order_id"]   # sellers/products stay readable (public catalog entities)


def rekey(col: pl.Expr, key: str) -> pl.Expr:
    k = hashlib.sha256(key.encode()).hexdigest()[:16]
    return pl.when(col.is_null()).then(None).otherwise(
        pl.concat_str([pl.lit("h"), (pl.concat_str([pl.lit(k), col]).hash(seed=7) % (1 << 60)).cast(pl.Utf8)]))


def observe(world_events: pl.DataFrame, seed: int, interventions: list[Intervention] | None = None,
            world_key: str = "", delays: dict | None = None) -> pl.DataFrame:
    delays = delays or DEFAULT_DELAYS
    ivs = [i for i in (interventions or []) if i.layer == "observation"]
    df = world_events
    n = df.height
    g = rng.stream(seed, "observe")
    streams = df["stream"].to_numpy()
    med = np.array([delays.get(s, (60, 1.0))[0] for s in streams], float)
    sig = np.array([delays.get(s, (60, 1.0))[1] for s in streams], float)
    delay_us = (med * np.exp(sig * g.standard_normal(n)) * 1e6).astype(np.int64)
    keep = np.ones(n, bool)
    t_us = df["event_time"].cast(pl.Int64).to_numpy()
    attrs = {"event_type": df["event_type"].to_numpy(), "stream": streams,
             "device": df["device"].fill_null("").to_numpy(), "region": df["region"].fill_null("").to_numpy()}
    for iv in ivs:
        m = selector_mask(iv.selector, attrs, n, {})
        if iv.window:
            lo, hi = (int(np.datetime64(iv.window[0], "us").astype(np.int64)), int(np.datetime64(iv.window[1], "us").astype(np.int64)))
            m &= (t_us >= lo) & (t_us < hi)
        if iv.target == "logging.drop_rate":       # value = fraction dropped
            keep &= ~(m & (g.random(n) < iv.value))
        elif iv.target == "logging.extra_delay_s":  # value = seconds added
            delay_us = np.where(m, delay_us + int(iv.value * 1e6), delay_us)
        else:
            raise ValueError(f"unknown observation target {iv.target}")
    out = df.with_columns(pl.from_epoch(pl.Series(t_us + delay_us), time_unit="us").alias("available_at")).filter(pl.Series(keep))
    dup = out.filter(pl.Series(rng.stream(seed, "dup").random(out.height) < DUP_RATE)).with_columns(
        pl.col("available_at") + pl.duration(seconds=30))
    out = pl.concat([out, dup])
    out = out.with_columns([rekey(pl.col(c), world_key or str(seed)).alias(c) for c in ID_COLS])
    return conform(out).sort("available_at")
