"""Integrity and leakage checks on a run (docs §12.1). Each returns (ok, detail)."""
from __future__ import annotations

import json
import pathlib

import polars as pl

from .contracts import EVENT_SCHEMA, TRUTH_ONLY_COLUMNS


def check_run(run: pathlib.Path) -> dict[str, tuple[bool, str]]:
    run = pathlib.Path(run)
    obs = pl.read_parquet(run / "observable" / "events.parquet")
    out: dict[str, tuple[bool, str]] = {}

    extra = set(obs.columns) - set(EVENT_SCHEMA)
    bad = {c for c in obs.columns if any(t in c for t in TRUTH_ONLY_COLUMNS)}
    out["observable_schema"] = (not extra and not bad, f"extra={sorted(extra)} truth_like={sorted(bad)}")
    neg = obs.filter(pl.col("available_at") < pl.col("event_time")).height
    out["available_after_event_time"] = (neg == 0, f"{neg} events visible before they happened")

    wpath = run / "truth" / "world_events.parquet"
    if not wpath.exists():                      # real replay track: world = source data
        return out
    w = pl.read_parquet(wpath)
    shared = set(obs["buyer_id"].drop_nulls().unique().to_list()) & set(w["buyer_id"].drop_nulls().unique().to_list())
    out["ids_rekeyed"] = (not shared, f"{len(shared)} buyer ids appear in both observable and truth")

    o = w.filter(pl.col("order_id").is_not_null() & pl.col("event_type").is_in(
        ["order_created", "payment_approved", "handed_to_carrier", "delivered"])).pivot(
        on="event_type", index="order_id", values="event_time", aggregate_function="first")
    cols = [c for c in ["order_created", "payment_approved", "handed_to_carrier", "delivered"] if c in o.columns]
    viol = 0
    for a, b in zip(cols, cols[1:]):
        viol += o.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null() & (pl.col(b) < pl.col(a))).height
    out["order_lifecycle_monotone"] = (viol == 0, f"{viol} order-step inversions")

    amt = w.filter(pl.col("event_type").is_in(["order_created", "payment_approved"])).pivot(
        on="event_type", index="order_id", values="amount", aggregate_function="first").drop_nulls()
    mism = amt.filter((pl.col("order_created") - pl.col("payment_approved")).abs() > 0.01).height if amt.height else 0
    out["payments_reconcile"] = (mism == 0, f"{mism} of {amt.height} orders: payment != price + freight")

    dup = w.group_by("event_id").len().filter(pl.col("len") > 1).height
    out["world_event_ids_unique"] = (dup == 0, f"{dup} duplicate world event ids")
    man = json.loads((run / "manifest.json").read_text())
    out["manifest_has_repro_key"] = (all(k in man for k in ("seed", "code", "scenario_hash")), "seed/code/scenario_hash")
    return out
