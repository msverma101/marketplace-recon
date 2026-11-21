"""EvidenceAccessPort over REAL Olist data (phase 2: real-only integration, no simulator).

Serves v3 Observations from the canonical Olist events (phase-1 semantics). Read-only and as-of:
nothing whose available_at is later than the request's as_of is ever returned.

Assumption (recorded in the EnvironmentScope): Olist has no ingestion/arrival times, so in this
real-replay environment available_at = event_time. Phase 4 replaces this with an observation model.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Union

import polars as pl

from . import pipeline, registry
from .config import Config
from .v3 import (CapabilitySpec, EnvironmentKind, EnvironmentScope, EvidenceRecord, EvidenceRequest, EvidenceResult,
                 FailureCode, InterventionEvent, InterventionSpec, LineageRoot, MetricDefinition, Observation, OriginClass,
                 ProvenanceLineage, Refusal, VariableDefinition, VariableKind)

EVENT_VARS = {   # variable id -> (canonical event_type, kind, value schema, missingness note)
    "order_placed": ("order_created", VariableKind.EVENT, {"amount": "float[BRL]", "promised_days": "float[days]"}, "none known"),
    "item_sold": ("order_item", VariableKind.EVENT, {"seller_id": "id", "product_id": "id", "price": "float[BRL]", "freight": "float[BRL]"}, "none known"),
    "payment_approved": ("payment_approved", VariableKind.EVENT, {"amount": "float[BRL]"}, "absent for some canceled orders"),
    "handed_to_carrier": ("handed_to_carrier", VariableKind.EVENT, {}, "absent for undelivered/canceled orders; 1,359 precede approval"),
    "delivered": ("delivered", VariableKind.EVENT, {"days_late": "float[days]"}, "censored: orders not yet delivered at the cut"),
    "survey_sent": ("survey_sent", VariableKind.EVENT, {}, "date only"),
    "review_score": ("review_submitted", VariableKind.CATEGORICAL, {"score": "int[1-5]"}, "~1% of orders have no answered survey"),
}
METRICS = [
    MetricDefinition(id="orders_count", value_schema={"value": "int"}, unit="orders", scope="orders", formula="count(order_placed)",
                     grain=["day", "week", "month"], population="all orders"),
    MetricDefinition(id="gmv", value_schema={"value": "float"}, unit="BRL", scope="orders", formula="sum(order_placed.amount)",
                     grain=["day", "week", "month"], population="all orders", empirical_identities=["gmv = sum(item_sold.price + item_sold.freight)"]),
    MetricDefinition(id="late_share", value_schema={"value": "float"}, unit="share", scope="delivered orders",
                     formula="count(delivered[days_late>0]) / count(delivered)", grain=["week", "month"],
                     population="orders delivered within the window (by delivery time)"),
    MetricDefinition(id="late_share_cohort", value_schema={"value": "float"}, unit="share", scope="orders",
                     formula="count(order_placed with delivered.days_late>0) / count(order_placed delivered by as_of)",
                     grain=["week", "month"], population="orders PLACED within the window and delivered by as_of (purchase cohort)"),
    MetricDefinition(id="mean_promised_days", value_schema={"value": "float"}, unit="days", scope="orders",
                     formula="mean(order_placed.promised_days)", grain=["week", "month"], population="orders placed within the window"),
    MetricDefinition(id="mean_delivery_days", value_schema={"value": "float"}, unit="days", scope="delivered orders",
                     formula="mean(delivered.t - order_placed.t)", grain=["week", "month"],
                     population="orders placed within the window that were delivered by as_of"),
    MetricDefinition(id="mean_review_score", value_schema={"value": "float"}, unit="1-5", scope="answered surveys",
                     formula="mean(review_score)", grain=["week", "month"], population="surveys answered within the window"),
]
METRIC_BY_ID = {m.id: m for m in METRICS}
TRUNC = {"day": "1d", "week": "1w", "month": "1mo"}


class EventFramePort:
    """Shared query engine over a canonical event frame (real replay or emulator).

    The frame needs: event_id, event_type, event_time, available_at, order_id, seller_id, product_id, region,
    amount, value, metric; optional record_key + revision for late corrections (supersession).
    """
    event_origin = OriginClass.REAL
    metric_origin = OriginClass.DERIVED
    source_id = "olist"
    observation_mechanism = "Olist order-system export (Kaggle 2018 release)"

    def __init__(self, events: pl.DataFrame, env: EnvironmentScope, root: LineageRoot, event_rows_max: int = 200_000,
                 trained_through: datetime | None = None):
        self.events = events
        self.env = env
        self.root = root
        self.event_rows_max = event_rows_max
        self.cut = env.data_cut or events["event_time"].max()
        self.trained_through = trained_through

    # -------------------------------------------------------------------------------------------
    def variables(self) -> list[VariableDefinition]:
        out = [VariableDefinition(id=k, kind=v[1], value_schema=v[2], scope="orders", description=f"Olist {v[0]} event")
               for k, v in EVENT_VARS.items()]
        return out + METRICS

    def capabilities(self) -> CapabilitySpec:
        shapes = {k: ["event_sequence"] for k in EVENT_VARS} | {m.id: ["series", "aggregate"] for m in METRICS}
        return CapabilitySpec(
            source_id=self.source_id, environment_id=self.env.id, variables=list(shapes), accepted_shapes=shapes,
            supported_requests=["query"], required_metadata=["as_of"],
            assumptions=self.env.assumptions + ["observational data only: no randomized studies in this source"],
            cost={"event_rows_per_query_max": float(self.event_rows_max)},
            failure_codes=[FailureCode.UNSUPPORTED_SHAPE, FailureCode.UNKNOWN_VARIABLE, FailureCode.NO_STUDIES_IN_SOURCE,
                           FailureCode.NO_INTERVENTIONS_IN_ENVIRONMENT, FailureCode.BUDGET_EXCEEDED])

    def propose_intervention(self, spec: InterventionSpec) -> InterventionEvent:
        return InterventionEvent(spec_id=spec.id, status="rejected", at=self.cut, failure_code=FailureCode.NO_INTERVENTIONS_IN_ENVIRONMENT.value,
                                 reason="this port is read-only; interventions are executed only by the emulator's action gateway")

    # -------------------------------------------------------------------------------------------
    def _evidence(self, origin: OriginClass, transforms: list[str]) -> EvidenceRecord:
        eid = "ev:" + hashlib.sha1("|".join([origin.value, self.env.id, *transforms]).encode()).hexdigest()[:16]
        return EvidenceRecord(id=eid, origin=origin, environment_id=self.env.id,
                              lineage=ProvenanceLineage(roots=[self.root], transformations=transforms, trained_through=self.trained_through))

    def _visible(self, as_of: datetime) -> pl.DataFrame:
        """Everything knowable at as_of: arrived by then, and only the latest revision of each record."""
        v = self.events.filter(pl.col("available_at") <= as_of)
        if "revision" in v.columns:
            v = v.sort("revision").unique(subset=["record_key"], keep="last", maintain_order=True).sort("event_time")
        return v

    def query(self, req: EvidenceRequest) -> Union[EvidenceResult, Refusal]:
        cap = self.capabilities()
        if req.variable_id not in cap.accepted_shapes:
            return Refusal(code=FailureCode.UNKNOWN_VARIABLE, reason=f"{req.variable_id!r} is not served by this source")
        if req.shape == "assignment":
            return Refusal(code=FailureCode.NO_STUDIES_IN_SOURCE, reason="Olist contains no randomized assignments")
        if req.shape not in cap.accepted_shapes[req.variable_id]:
            return Refusal(code=FailureCode.UNSUPPORTED_SHAPE,
                           reason=f"{req.variable_id} is served as {cap.accepted_shapes[req.variable_id]}, not {req.shape}")
        vis = self._visible(req.as_of)
        flags = []
        if req.window_end and req.window_end > req.as_of:
            flags.append("window extends past as_of: values after as_of are not included (incomplete, not zero)")
        if req.variable_id in EVENT_VARS:
            return self._events(req, vis, flags)
        return self._metric(req, vis, flags)

    def _window(self, df: pl.DataFrame, req: EvidenceRequest) -> pl.DataFrame:
        if req.window_start:
            df = df.filter(pl.col("event_time") >= req.window_start)
        if req.window_end:
            df = df.filter(pl.col("event_time") < req.window_end)
        for k, v in req.filters.items():
            df = df.filter(pl.col(k) == v)
        return df

    def _events(self, req, vis, flags):
        etype, _, schema, missing = EVENT_VARS[req.variable_id]
        df = self._window(vis.filter(pl.col("event_type") == etype), req)
        if df.height > self.event_rows_max:
            return Refusal(code=FailureCode.BUDGET_EXCEEDED, reason=f"{df.height:,} rows; narrow the window or use a metric")
        cols = ["event_id", "event_time", "order_id", "seller_id", "product_id", "region", "amount", "value", "metric"]
        ev = self._evidence(self.event_origin, [f"{self.source_id} {self.env.mechanism_versions}", f"select {etype}"])
        obs = Observation(id=f"obs:{req.variable_id}:{ev.id}:{req.as_of.isoformat()}", variable_id=req.variable_id, shape="event_sequence",
                          payload=df.select(cols).to_dicts(), window_start=req.window_start, window_end=req.window_end,
                          available_at=df["available_at"].max() if df.height else req.as_of, population="orders",
                          observation_mechanism=self.observation_mechanism, missingness=missing,
                          quality_flags=flags, evidence_id=ev.id)
        return EvidenceResult(observations=[obs], evidence=[ev])

    def _metric(self, req, vis, flags):
        m = METRIC_BY_ID[req.variable_id]
        grain = req.grain or "week"
        if req.shape == "series" and grain not in m.grain:
            return Refusal(code=FailureCode.UNSUPPORTED_SHAPE, reason=f"{m.id} grains are {m.grain}")
        placed = self._window(vis.filter(pl.col("event_type") == "order_created"), req)
        if m.id == "orders_count":
            base, val = placed, pl.len()
        elif m.id == "gmv":
            base, val = placed, pl.col("amount").sum()
        elif m.id == "mean_promised_days":
            base, val = placed, pl.col("value").mean()
        elif m.id == "late_share":
            base, val = self._window(vis.filter(pl.col("event_type") == "delivered"), req), (pl.col("value") > 0).mean()
            flags = flags + ["denominator = deliveries completed by as_of; undelivered orders are censored, not on time"]
        elif m.id == "late_share_cohort":
            d = vis.filter(pl.col("event_type") == "delivered").select("order_id", pl.col("value").alias("days_late"))
            base = placed.join(d, on="order_id", how="inner")
            val = (pl.col("days_late") > 0).mean()
            flags = flags + ["purchase cohort: only orders delivered by as_of; recent cohorts are biased toward fast (on-time) deliveries"]
        elif m.id == "mean_review_score":
            base, val = self._window(vis.filter(pl.col("event_type") == "review_submitted"), req), pl.col("value").mean()
        else:  # mean_delivery_days: by purchase week, only orders delivered by as_of
            d = vis.filter(pl.col("event_type") == "delivered").select("order_id", pl.col("event_time").alias("t_del"))
            base = placed.join(d, on="order_id", how="inner")
            val = ((pl.col("t_del") - pl.col("event_time")).dt.total_hours() / 24).mean()
            flags = flags + ["only orders delivered by as_of: recent weeks are biased toward fast deliveries"]
        ev = self._evidence(self.metric_origin, [f"{self.source_id} {self.env.mechanism_versions}", f"metric {m.id}: {m.formula}"])
        if req.shape == "aggregate":
            payload = {"value": base.select(val).item() if base.height else None, "n": base.height}
        else:
            s = (base.with_columns(pl.col("event_time").dt.truncate(TRUNC[grain]).alias("t"))
                 .group_by("t").agg(val.alias("value"), pl.len().alias("n")).sort("t"))
            payload = [{"t": r["t"].isoformat(), "value": r["value"], "n": r["n"]} for r in s.to_dicts()]
        obs = Observation(id=f"obs:{m.id}:{grain}:{ev.id}:{req.as_of.isoformat()}", variable_id=m.id, shape=req.shape, payload=payload,
                          window_start=req.window_start, window_end=req.window_end,
                          available_at=base["available_at"].max() if base.height else req.as_of, population=m.population,
                          observation_mechanism=f"derived from {self.source_id} events by {m.formula}", missingness="see source variables",
                          quality_flags=flags, evidence_id=ev.id)
        return EvidenceResult(observations=[obs], evidence=[ev])


class OlistReplayPort(EventFramePort):
    """Real Olist data. Olist has no arrival times, so available_at = event_time (recorded assumption)."""

    def __init__(self, cfg: Config | None = None, event_rows_max: int = 200_000):
        cfg = cfg or Config.load()
        canon = pipeline.canonical_olist(cfg)
        self.manifest = registry.load("olist", cfg.artifacts)
        ev = pl.read_parquet(canon / "events.parquet").filter(pl.col("event_time").is_not_null())
        events = ev.with_columns(pl.col("event_time").alias("available_at"))
        env = EnvironmentScope(
            id="olist-real-replay", kind=EnvironmentKind.REAL_REPLAY, data_cut=events["event_time"].max(),
            mechanism_versions={"adapter": canon.name},
            assumptions=["available_at = event_time (Olist exports have no arrival times; phase 4 adds an observation model)",
                         "timestamps are naive local time (assumed America/Sao_Paulo)"])
        root = LineageRoot(source="olist", source_version=self.manifest.version, files=self.manifest.files)
        super().__init__(events, env, root, event_rows_max)
