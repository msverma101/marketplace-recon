"""Baseline monitor: deliberately simple anomaly detection on replayed observables.

It is a reference consumer that proves the replay -> monitor -> evaluator loop. It detects
symptoms but never attributes causes (cause = 'unknown'), so it should score on detection
and fail attribution: the gap a causal engine is meant to close.
Buckets by available_at (processing time), like a naive streaming dashboard.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import polars as pl

from .replay import Replayer

HOURLY = [("order_created", None), ("order_created", "device"), ("checkout_error", None), ("checkout_error", "device"),
          ("session_start", None), ("payment_approved", None)]
Z = 4.0


def run(replayer: Replayer, warmup_days: int = 4) -> list[dict]:
    hist: dict = defaultdict(lambda: deque(maxlen=7))      # (metric, seg, hour-of-day) -> last 7 days
    daily_hist: dict = defaultdict(lambda: deque(maxlen=14))
    day_acc: dict = defaultdict(list)
    alerts, n_hours, cur_day = [], 0, None
    for clock, f in replayer.windows():
        n_hours += 1
        hod = (clock.hour - 1) % 24
        counts = {}
        for metric, dim in HOURLY:
            sub = f.filter(pl.col("event_type") == metric)
            if dim is None:
                counts[(metric, "all")] = sub.height
            else:
                for v, n in sub.group_by(dim).len().iter_rows():
                    counts[(metric, f"{dim}={v}")] = n
                for v in ("mobile", "desktop"):
                    counts.setdefault((metric, f"{dim}={v}"), 0)
        rate = f.filter((pl.col("event_type") == "metric_sample") & (pl.col("metric") == "checkout_error_rate"))
        for dev, val in rate.select("device", "value").iter_rows():
            if val is not None and not np.isnan(val):
                counts[("checkout_error_rate", f"device={dev}")] = val
        for key, x in counts.items():
            h = hist[(key, hod)]
            if len(h) >= warmup_days:
                exp = float(np.mean(h))
                if key[0] == "checkout_error_rate":
                    sd = max(float(np.std(h)), 0.01)
                else:
                    sd = np.sqrt(exp + 1.0)
                z = (x - exp) / sd
                if abs(z) >= Z:
                    alerts.append({"time": clock.isoformat(), "metric": key[0], "segment": key[1], "z": round(z, 2),
                                   "observed": x, "expected": round(exp, 3), "cause": "unknown"})
            h.append(x)
        # daily quality metrics
        d = f.filter(pl.col("event_type") == "delivered").select("region", "value")
        day_acc["late"].append(d)
        day_acc["score"].append(f.filter(pl.col("event_type") == "review_submitted").select("region", "value"))
        day_acc["promise"].append(f.filter(pl.col("event_type") == "order_created").select("value"))
        if cur_day is None:
            cur_day = clock.date()
        if clock.hour == 0 and clock.date() != cur_day:
            cur_day = clock.date()
            alerts += _daily(clock, day_acc, daily_hist, warmup_days)
            day_acc.clear()
    return alerts


def _daily(clock: datetime, acc: dict, hist: dict, warm: int) -> list[dict]:
    out = []
    late = pl.concat(acc.get("late", [])) if acc.get("late") else pl.DataFrame()
    score = pl.concat(acc.get("score", [])) if acc.get("score") else pl.DataFrame()
    prom = pl.concat(acc.get("promise", [])) if acc.get("promise") else pl.DataFrame()
    vals = {}
    if late.height:
        vals[("late_delivered", "all")] = (late["value"] > 0).mean()
        for r, g in late.group_by("region"):
            if g.height >= 5:
                vals[("late_delivered", f"region={r[0]}")] = (g["value"] > 0).mean()
    if score.height:
        vals[("review_score", "all")] = score["value"].mean()
        for r, g in score.group_by("region"):
            if g.height >= 5:
                vals[("review_score", f"region={r[0]}")] = g["value"].mean()
    if prom.height and prom["value"].null_count() < prom.height:
        vals[("promised_days", "all")] = prom["value"].mean()
    for key, x in vals.items():
        h = hist[key]
        if len(h) >= warm:
            m, sd = float(np.mean(h)), max(float(np.std(h)), 0.02 * max(abs(float(np.mean(h))), 1e-3), 0.01)
            z = (x - m) / sd
            if abs(z) >= Z:
                out.append({"time": clock.isoformat(), "metric": key[0], "segment": key[1], "z": round(z, 2),
                            "observed": round(float(x), 4), "expected": round(m, 4), "cause": "unknown"})
        h.append(x)
    return out
