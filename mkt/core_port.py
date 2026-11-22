"""The port the causal core sees, conforming to `Contracts/` (schema 2.0).

Wraps the internal EventFramePort engine (real Olist replay or emulator) and speaks only
Contracts types: manifest() -> ProviderManifest, query(CapabilityRequest) -> CapabilityResponse.
Read-only by construction (I15): there is no act/write/intervene method here.
Design: docs/p0/CONFORMANCE_AND_M1_M5_DESIGN.md.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter

from Contracts.capabilities import CapabilityRequest, CapabilityResponse, ProviderManifest
from Contracts.common import SCHEMA_VERSION, Clock, EnvironmentScope, ProvenanceLineage, SourceAttestation
from Contracts.evidence import EvidenceRecord
from Contracts.observations import (AggregateStats, DataQuality, EventBatch, EventRecord, Observation,
                                    ObservationClocks, TimePoint, TimeSeries)
from Contracts.variables import (EventPayloadSchema, EventVariableDefinition, FormulaBinOp, FormulaRef,
                                 MetricDefinition, MetricTimeSemantics, NumericValueSchema, PopulationCoverage)

from .replay_port import EVENT_VARS, SESSION_EVENT
from .v3 import EvidenceRequest as _InternalRequest  # internal engine type only, never exposed
from .v3 import Refusal as _InternalRefusal

LOCAL = ZoneInfo("America/Sao_Paulo")      # Olist and the simulated clock are local Brazil time
GRAINS = {"1d": "day", "1w": "week"}
SERVED = ("get_series", "get_aggregate", "get_event_batch", "get_source_metadata")
_REQ = TypeAdapter(CapabilityRequest)

# metric id -> (aggregation_kind, dtype, unit)
_KIND = {"orders_count": ("count", "count", "orders"), "payments_count": ("count", "count", "orders"),
         "gmv": ("sum", "real", "BRL"), "late_share": ("mean", "proportion", None), "late_share_cohort": ("mean", "proportion", None),
         "mean_promised_days": ("mean", "real", "days"), "mean_delivery_days": ("mean", "real", "days"),
         "mean_review_score": ("mean", "real", "1-5"), "sessions_count": ("count", "count", "sessions"),
         "cart_adds_count": ("count", "count", "sessions"), "checkouts_count": ("count", "count", "sessions")}
# ratio metrics: id -> (numerator, denominator, identity_tolerance or None)
RATIOS = {"order_payment_ratio": ("payments_count", "orders_count", 0.02),     # identity: each order is paid once
          "cart_rate": ("cart_adds_count", "sessions_count", None),             # funnel stages
          "checkout_rate": ("checkouts_count", "cart_adds_count", None),
          "completion_rate": ("orders_count", "checkouts_count", None)}


def to_utc(t: datetime) -> datetime:
    return t.replace(tzinfo=LOCAL).astimezone(timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def to_local(t: datetime) -> datetime:
    return t.astimezone(LOCAL).replace(tzinfo=None)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class ContractPort:
    """Contracts.ports.EvidenceAccessPort over an internal EventFramePort."""

    def __init__(self, engine, *, provider_id: str, origin_kind: str, role: str, scope: EnvironmentScope,
                 available_clock: str, lineage_root: str, generator_version: str | None = None,
                 hidden_variables: frozenset[str] = frozenset(), extensions: dict | None = None):
        self._e = engine
        self.provider_id, self.origin_kind, self.role, self.scope = provider_id, origin_kind, role, scope
        self.available_clock, self.lineage_root, self.generator_version = available_clock, lineage_root, generator_version
        self.hidden = set(hidden_variables)
        self._ext = extensions or {}
        self._metric_ids = [m.id for m in engine.metrics]

    # -------------------------------------------------------------- catalogue
    def expose(self, variable_id: str) -> None:
        """Make a hidden variable visible (used only by the emulator's measurement policy, never by the core)."""
        self.hidden.discard(variable_id)

    def _served_metrics(self) -> list[str]:
        ms = [m for m in self._metric_ids if m not in self.hidden]
        ms += [r for r, (a, b, _) in RATIOS.items() if a in ms and b in ms and r not in self.hidden]
        return ms

    def _served_events(self) -> list[str]:
        types = set(self._e.events["event_type"].unique().to_list())
        out = [k for k, v in EVENT_VARS.items() if v[0] in types]
        out += [v for v in SESSION_EVENT.values() if v in types]
        return [x for x in out if x not in self.hidden]

    def definitions(self) -> tuple:
        """Read-only variable catalogue (gap G1: ProviderManifest has no catalogue field)."""
        att = SourceAttestation(source_id=self.provider_id, method="adapter_declared")
        base = dict(schema_version=SCHEMA_VERSION, scope=self.scope, provenance=att)
        out = []
        internal = {m.id: m for m in self._e.metrics}
        for vid in self._served_metrics():
            if vid in RATIOS:
                a, b, tol = RATIOS[vid]
                out.append(MetricDefinition(**base, kind="metric", variable_id=vid,
                                            value_schema=NumericValueSchema(type="numeric", dtype="real", minimum=0.0),
                                            aggregation_kind="ratio", time_semantics=MetricTimeSemantics(grain="1w", window="tumbling"),
                                            formula_ast=FormulaBinOp(node="binop", op="div", left=FormulaRef(node="ref", variable_id=a),
                                                                     right=FormulaRef(node="ref", variable_id=b)),
                                            identity_tolerance=tol, description=f"{a} / {b}", extensions=self._ext.get(vid, {})))
                continue
            agg, dtype, unit = _KIND[vid]
            m = internal[vid]
            ext = self._ext.get(vid, {})
            out.append(MetricDefinition(**base, kind="metric", variable_id=vid,
                                        value_schema=NumericValueSchema(type="numeric", dtype=dtype, minimum=0.0,
                                                                        maximum=1.0 if dtype == "proportion" else None),
                                        unit=unit, aggregation_kind=agg, description=m.formula,
                                        population_coverage=PopulationCoverage(population_id=m.scope.replace(" ", "_"), coverage="full"),
                                        time_semantics=MetricTimeSemantics(grain="1w", window="tumbling"), extensions=ext))
        for vid in self._served_events():
            fields = {"entity_id": "text", "region": "category", "value": "real"}
            out.append(EventVariableDefinition(**base, kind="event", variable_id=vid,
                                               value_schema=EventPayloadSchema(type="event", fields=fields, identity_fields=("entity_id",)),
                                               occurrence_time="event_time"))
        return tuple(out)

    def manifest(self) -> ProviderManifest:
        return ProviderManifest(provider_id=self.provider_id, manifest_version="mkt-core-port-1", supported_requests=SERVED,
                                max_rows=self._e.event_rows_max, scopes=(self.scope,))

    # -------------------------------------------------------------- queries
    def query(self, request) -> CapabilityResponse:
        req = _REQ.validate_python(request) if isinstance(request, dict) else request
        fp = _sha(req.model_dump(mode="json"))

        def resp(status, reason=None, obs=(), truncated=False):
            return CapabilityResponse(request_id=req.request_id, request_fingerprint=fp, status=status, observations=tuple(obs),
                                      reason=reason, truncated=truncated, available_at=req.data_cut)
        if req.request not in SERVED:
            return resp("UNSUPPORTED", f"{req.request} is not served by {self.provider_id}: no studies, panels, strata or "
                                       "intervention history in this source")
        if req.scope.environment_id != self.scope.environment_id:
            return resp("UNSUPPORTED", f"scope {req.scope.environment_id} is not served here ({self.scope.environment_id})")
        if req.request == "get_source_metadata":
            return resp("UNSUPPORTED", "source metadata is served through manifest() and definitions()")
        known = set(self._served_metrics()) | set(self._served_events())
        unknown = [v for v in req.variables if v not in known]
        if unknown:
            return resp("UNSUPPORTED", f"variables not served in this environment: {unknown}")
        obs = []
        for vid in req.variables:
            if req.request == "get_event_batch":
                if vid not in self._served_events():
                    return resp("UNSUPPORTED", f"{vid} is a metric; ask get_series or get_aggregate")
                o = self._events(req, vid)
            else:
                if vid not in self._served_metrics():
                    return resp("UNSUPPORTED", f"{vid} is an event variable; ask get_event_batch")
                o = self._metric(req, vid)
            if isinstance(o, str):
                return resp("BUDGET_EXCEEDED" if o.startswith("budget") else "UNSUPPORTED", o)
            obs.append(o)
        return resp("OK", None, obs)

    def _internal(self, req, vid, shape):
        w = req.window
        return _InternalRequest(variable_id=vid, shape=shape, as_of=to_local(req.data_cut),
                                window_start=to_local(w.start) if w else None, window_end=to_local(w.end) if w else None,
                                grain=GRAINS.get(getattr(req, "grain", None) or "1w", "week"))

    def _metric(self, req, vid):
        if getattr(req, "grain", None) and req.grain not in GRAINS:
            return f"grain {req.grain} not served (served: {sorted(GRAINS)})"
        shape = "series" if req.request == "get_series" else "aggregate"
        if vid in RATIOS:
            a, b, _ = RATIOS[vid]
            ra, rb = (self._e.query(self._internal(req, x, shape)) for x in (a, b))
            if isinstance(ra, _InternalRefusal) or isinstance(rb, _InternalRefusal):
                return "component metric refused"
            pa, pb = ra.observations[0].payload, rb.observations[0].payload
            if shape == "series":
                db = {p["t"]: p["value"] for p in pb}
                pts = [(p["t"], p["value"] / db[p["t"]] if db.get(p["t"]) else None) for p in pa if p["t"] in db]
                payload = self._series(pts, req)
            else:
                payload = AggregateStats(shape="aggregate_stats", n=int(pb["n"] or 0), numerator=float(pa["value"] or 0),
                                         denominator=float(pb["value"] or 0), window=req.window)
        else:
            r = self._e.query(self._internal(req, vid, shape))
            if isinstance(r, _InternalRefusal):
                return r.reason
            p = r.observations[0].payload
            if shape == "series":
                payload = self._series([(x["t"], x["value"]) for x in p], req)
            else:
                agg = _KIND[vid][0]
                n = int(p["n"] or 0)
                v = p["value"]
                payload = (AggregateStats(shape="aggregate_stats", n=n, total=float(v or 0), window=req.window) if agg in ("count", "sum")
                           else AggregateStats(shape="aggregate_stats", n=n, mean=float(v), window=req.window) if n and v is not None
                           else AggregateStats(shape="aggregate_stats", n=0, total=0.0, window=req.window))
        if isinstance(payload, str):
            return payload
        return self._observation(req, vid, payload, f"{vid}:{req.request}:{getattr(req, 'grain', '')}:"
                                 f"{req.window.start.isoformat() if req.window else ''}")

    def _series(self, pts, req):
        if not pts:
            return "no data in the window"
        if len(pts) > req.max_buckets:
            return f"budget: {len(pts)} buckets > max_buckets {req.max_buckets}"
        return TimeSeries(shape="time_series", points=tuple(
            TimePoint(t=to_utc(datetime.fromisoformat(t)), value=None if v is None else float(v)) for t, v in sorted(pts)))

    def _events(self, req, vid):
        etype = EVENT_VARS[vid][0] if vid in EVENT_VARS else vid
        vis = self._e._visible(to_local(req.data_cut))
        import polars as pl
        df = vis.filter(pl.col("event_type") == etype)
        if req.window:
            df = df.filter((pl.col("event_time") >= to_local(req.window.start)) & (pl.col("event_time") < to_local(req.window.end)))
        cap = min(req.max_rows, req.max_events)
        if df.height > cap:
            return f"budget: {df.height:,} events > limit {cap:,}; narrow the window or ask a metric"
        if df.height == 0:
            return "no events in the window"
        q = "KNOWN"
        evs = tuple(EventRecord(event_id=r["event_id"].replace(" ", "_"), occurred_at=Clock(quality=q, value=to_utc(r["event_time"])),
                                attributes={"entity_id": r["order_id"], "region": r.get("region"),
                                            "value": None if r.get("value") is None else float(r["value"])})
                    for r in df.select("event_id", "event_time", "order_id", "region", "value").to_dicts())
        return self._observation(req, vid, EventBatch(shape="event_batch", events=evs), f"{vid}:events:{req.window.start.isoformat() if req.window else ''}")

    def _observation(self, req, vid, payload, record_key):
        body = payload.model_dump(mode="json")
        lineage = ProvenanceLineage(source_id=self.provider_id, origin_kind=self.origin_kind, lineage_root_id=self.lineage_root,
                                    generator_version=self.generator_version, content_hash=_sha(body),
                                    transformation_hash=_sha({"variable": vid, "request": req.request}))
        cut = to_utc(self._e.cut) if self._e.cut else req.data_cut
        av = min(req.data_cut, cut)
        observed = min(req.window.end, av) if req.window else av
        clocks = ObservationClocks(observed_at=Clock(quality="KNOWN", value=observed),
                                   available_at=Clock(quality=self.available_clock, value=av),
                                   as_of=req.data_cut, source_timezone="America/Sao_Paulo")
        return Observation(schema_version=SCHEMA_VERSION, observation_id=f"obs:{_sha([record_key, req.data_cut.isoformat(), body])[:24]}",
                           variable_id=vid, payload=payload, source_id=self.provider_id, record_key=_safe(record_key), revision=1,
                           lineage=lineage, scope=self.scope, clocks=clocks,
                           dq=DataQuality(maturity="IMMATURE" if req.window and req.window.end > av else "UNKNOWN",
                                          missingness="declared", selection="unknown"))

    def evidence_record(self, obs: Observation) -> EvidenceRecord:
        return EvidenceRecord(evidence_id=f"ev:{obs.observation_id}", lineage=obs.lineage, scope=obs.scope,
                              provenance_role=self.role, content_hash=obs.lineage.content_hash)


def _safe(s: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9_.:\-/]", "_", s)
    return s if s[:1].isalnum() else "k" + s


# ------------------------------------------------------------------ constructors
def olist_port(cfg=None) -> ContractPort:
    """Real Olist data. available_at is ESTIMATED: Olist has no arrival times (assumption: available at event time)."""
    from .replay_port import OlistReplayPort
    e = OlistReplayPort(cfg)
    files = e.root.files
    scope = EnvironmentScope(environment_id="olist-production-2018", environment_kind="production", data_cut=to_utc(e.cut),
                             dataset_version=f"olist-kaggle-{e.root.source_version}")
    return ContractPort(e, provider_id="olist", origin_kind="original", role="REAL_OBSERVED", scope=scope,
                        available_clock="ESTIMATED", lineage_root=f"olist:{_sha(files)[:16]}")


def emulator_port(engine, session_params=None, hidden_variables=frozenset()) -> ContractPort:
    """Emulator (synthetic). generator_version pins the chain params, the observation model, the seed and the cards."""
    mv = engine.env.mechanism_versions
    gen = "chain-" + _sha(mv)[:16]
    scope = EnvironmentScope(environment_id=_safe(engine.env.id), environment_kind="emulator", data_cut=to_utc(engine.cut),
                             generator_version=gen)
    ext = {}
    if session_params is not None:
        card_for = {"sessions_count": "TC-RR1", "cart_adds_count": "TC-RR1", "checkouts_count": "TC-TB1"}
        for vid, card in card_for.items():
            ext[vid] = {"mkt:transport_card": card, "mkt:transport_status": session_params.cards[card][:60]}
    return ContractPort(engine, provider_id="emulator-chain", origin_kind="synthetic_generated", role="SYNTHETIC_GENERATED",
                        scope=scope, available_clock="KNOWN", lineage_root=f"emulator:{gen}", generator_version=gen,
                        hidden_variables=frozenset(hidden_variables), extensions=ext)
