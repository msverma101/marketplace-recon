"""Phase 5: action gateway, one intervention, counterfactual twin, evaluator-only oracle.

The oracle (true per-order and aggregate effects) is returned in a separate `Truth` object and is
never given to the EmulatorPort; the reader only ever sees the port (gate S6).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import polars as pl

from ..interventions import parse_selector
from ..v3 import FailureCode, InterventionEvent, InterventionSpec
from .chain import ChainParams, simulate
from .observe_chain import EmulatorPort, ObservationModel, observe

ALLOWED = {
    "carrier.transit_multiplier": {"operators": {"scale"}, "bounds": (0.5, 5.0), "selector_keys": {"region"}},
}


def gateway(spec: InterventionSpec, regions: set[str], horizon: tuple[datetime, datetime]):
    """Validate against the allowlist; return (InterventionEvent, transit_multiplier callable | None)."""
    def reject(code: FailureCode, why: str):
        return InterventionEvent(spec_id=spec.id, status="rejected", at=spec.window_start, failure_code=code.value, reason=why), None

    rule = ALLOWED.get(spec.target)
    if rule is None:
        return reject(FailureCode.UNSUPPORTED_REQUEST, f"target {spec.target!r} not on the allowlist {sorted(ALLOWED)}")
    if spec.operator not in rule["operators"]:
        return reject(FailureCode.UNSUPPORTED_REQUEST, f"operator {spec.operator!r} not allowed for {spec.target}")
    lo, hi = rule["bounds"]
    if not (isinstance(spec.value, (int, float)) and lo <= float(spec.value) <= hi):
        return reject(FailureCode.UNSUPPORTED_REQUEST, f"value {spec.value!r} outside safety bounds [{lo}, {hi}]")
    try:
        clauses = parse_selector(spec.scope)
    except ValueError as e:
        return reject(FailureCode.UNSUPPORTED_REQUEST, str(e))
    for key, _, vals in clauses:
        if key not in rule["selector_keys"]:
            return reject(FailureCode.UNSUPPORTED_REQUEST, f"selector key {key!r} not allowed")
        if not set(vals) <= regions:
            return reject(FailureCode.UNSUPPORTED_REQUEST, f"unknown region(s) {sorted(set(vals) - regions)}")
    if not (horizon[0] <= spec.window_start < spec.window_end <= horizon[1]):
        return reject(FailureCode.UNSUPPORTED_REQUEST, "window outside the simulation horizon")

    w0 = int(np.datetime64(spec.window_start, "us").astype(np.int64))
    w1 = int(np.datetime64(spec.window_end, "us").astype(np.int64))
    factor = float(spec.value)

    def multiplier(df: pl.DataFrame) -> np.ndarray:
        shipped_us = df["shipped_us"].to_numpy()                       # absolute time: chunk-invariant
        m = (shipped_us >= w0) & (shipped_us < w1)
        for key, op, vals in clauses:
            hit = np.isin(df["buyer_region"].to_numpy(), vals)
            m &= ~hit if op == "!=" else hit
        return np.where(m, factor, 1.0)

    return InterventionEvent(spec_id=spec.id, status="executed", at=spec.window_start), multiplier


@dataclass
class Truth:
    """Evaluator-only. Never passed to the port or the reader."""
    spec: InterventionSpec
    per_order: pl.DataFrame          # order_idx, affected, late_fact, late_twin, score_fact, score_twin
    effect_late_share_affected: float
    effect_late_share_region_weeks: dict[str, float]   # week -> factual - twin, in the scoped region (cohort by purchase week)


def run(params: ChainParams, start: date, end: date, seed: int, spec: InterventionSpec | None,
        om: ObservationModel = ObservationModel()):
    horizon = (datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()))
    regions = set(params.buyer_regions)
    event, mult = (None, None)
    if spec is not None:
        event, mult = gateway(spec, regions, horizon)
    factual = simulate(params, start, end, seed=seed, transit_multiplier=mult)
    twin = simulate(params, start, end, seed=seed)
    port = EmulatorPort(observe(factual, om, seed=seed), params, om, horizon[0], seed)
    truth = None
    if mult is not None:
        late = lambda w: ((w["delivered_at"].dt.truncate("1d") - w["promised_at"]).dt.total_days() > 0)
        po = pl.DataFrame({"order_idx": factual["order_idx"], "region": factual["buyer_region"],
                           "week": factual["purchased_at"].dt.truncate("1w").dt.date().cast(pl.Utf8),
                           "affected": factual["delivered_at"] != twin["delivered_at"],
                           "late_fact": late(factual), "late_twin": late(twin),
                           "score_fact": factual["score"], "score_twin": twin["score"]})
        aff = po.filter(pl.col("affected"))
        scoped = [v for _, _, vals in parse_selector(spec.scope) for v in vals]
        reg = po.filter(pl.col("region").is_in(scoped) & pl.col("late_fact").is_not_null() & pl.col("late_twin").is_not_null())
        weeks = reg.group_by("week").agg((pl.col("late_fact").mean() - pl.col("late_twin").mean()).alias("d")).sort("week")
        truth = Truth(spec=spec, per_order=po,
                      effect_late_share_affected=float(aff["late_fact"].mean() - aff["late_twin"].mean()) if aff.height else 0.0,
                      effect_late_share_region_weeks=dict(zip(weeks["week"].to_list(), weeks["d"].to_list())))
    return event, factual, twin, port, truth
