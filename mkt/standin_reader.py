"""Stand-in reader: a deliberately small consumer of the EvidenceAccessPort.

It is NOT the causal core v3 (that lives elsewhere). It exists to prove the input layer can serve
a question-first investigation with no simulator, no anomaly feed, no experiment fields and no
domain-specific code in the reader (gate S8). It follows the v3 workflow names:
FRAME -> AUDIT -> ACQUIRE -> EVALUATE -> CHALLENGE -> REPORT.

Output types Finding / EvidenceGap are provisional stand-ins for the v3 outbound contracts.
"""
from __future__ import annotations

from datetime import datetime
from statistics import median

from pydantic import BaseModel, ConfigDict

from .v3 import CausalQuestion, EvidenceAccessPort, EvidenceRequest, Refusal


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    statement: str
    certainty: str                    # "observational association" | "identified" ...
    cited_observations: list[str]
    rivals_considered: list[str]


class EvidenceGap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    statement: str
    would_resolve: str


class Report(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    question: CausalQuestion
    findings: list[Finding]
    gaps: list[EvidenceGap]
    refusals: list[str]


def _series(port, var, start, end, as_of, refusals):
    r = port.query(EvidenceRequest(variable_id=var, shape="series", grain="week", as_of=as_of, window_start=start, window_end=end))
    if isinstance(r, Refusal):
        refusals.append(f"{var}: {r.code.value}")
        return None, None
    o = r.observations[0]
    return o, [(datetime.fromisoformat(p["t"]), p["value"]) for p in o.payload if p["value"] is not None]


def _shift(points, at, before_weeks=6):
    """Largest |robust z| in the 3 weeks from `at`, vs the preceding weeks.
    Baseline = median, scale = 1.4826 * MAD (robust to a contaminated pre-period, e.g. a spike just before `at`)."""
    pre = [v for t, v in points if t < at][-before_weeks:]
    post = [(t, v) for t, v in points if t >= at][:3]
    if len(pre) < 3 or not post:
        return None
    mu = median(pre)
    sd = max(1.4826 * median([abs(v - mu) for v in pre]), 1e-3 * max(abs(mu), 1e-9), 1e-9)
    t, v = max(post, key=lambda tv: abs((tv[1] - mu) / sd))
    return {"week": t.date().isoformat(), "value": v, "baseline": mu, "z": (v - mu) / sd}


def investigate_change(port: EvidenceAccessPort, metrics: list[str], at: datetime, start: datetime, end: datetime,
                       as_of: datetime, yardstick_checks: list[tuple[str, str, str | None]] = ()) -> Report:
    """yardstick_checks: (rate_metric, yardstick_metric, outcome_metric) triples supplied by the task profile.
    The reader itself knows no metric names: it only tests 'rate fell while its yardstick moved and the
    underlying outcome did not', a generic definition-change rival."""
    q = CausalQuestion(target=",".join(metrics), contrast=f"weeks from {at.date()} vs the 6 weeks before",
                       estimand="change in weekly level (descriptive)", population="as defined by each metric",
                       horizon=f"{start.date()}..{end.date()}", constraints=[f"as_of {as_of.isoformat()}"])
    cap = port.capabilities()                                     # AUDIT: what can this source answer?
    refusals: list[str] = []
    found, cited = {}, {}
    for m in metrics:                                             # ACQUIRE + EVALUATE
        obs, pts = _series(port, m, start, end, as_of, refusals)
        if obs is None:
            continue
        cited[m] = obs.id
        found[m] = _shift(pts, at)
    findings, gaps = [], []
    for m, s in found.items():
        if s and abs(s["z"]) >= 3:
            findings.append(Finding(statement=f"{m} moved from {s['baseline']:.3g} to {s['value']:.3g} in week {s['week']} (z={s['z']:.1f})",
                                    certainty="observational association", cited_observations=[cited[m]],
                                    rivals_considered=["seasonality", "measurement or definition change"]))
    # CHALLENGE: a falling rate can be a moved yardstick (changed denominator), not a real improvement
    for rate, yard, outcome in yardstick_checks:
        r_, y_ = found.get(rate), found.get(yard)
        o_ = found.get(outcome) if outcome else None
        if r_ and y_ and r_["z"] <= -2 and abs(y_["z"]) >= 2 and (o_ is None or abs(o_["z"]) < 2):
            findings.append(Finding(
                statement=f"{rate} fell while its yardstick {yard} moved (z={y_['z']:.1f})"
                          + (f" and the outcome {outcome} did not change (z={o_['z']:.1f})" if o_ else "")
                          + ": the drop reflects a changed yardstick, not a real improvement",
                certainty="observational association", cited_observations=[cited[k] for k in (rate, yard, outcome) if k and k in cited],
                rivals_considered=[f"genuine improvement in {outcome or rate}", "volume change in the window"]))
    if any("observational data only" in a for a in cap.assumptions):
        gaps.append(EvidenceGap(statement="cause of the changes is not identified: the source is observational only",
                                would_resolve="a natural-experiment design (e.g. region-level difference-in-differences) or an external record of the shock"))
    return Report(question=q, findings=findings, gaps=gaps, refusals=refusals)


def localise_change(port: EvidenceAccessPort, metric: str, filters: dict[str, str], start: datetime, end: datetime,
                    as_of: datetime, z_threshold: float = 3.0) -> dict:
    """Which periods of a metric deviate from its own robust baseline, and by how much (descriptive only).
    Baseline = median of all periods, scale = 1.4826 * MAD; excess = value - baseline."""
    r = port.query(EvidenceRequest(variable_id=metric, shape="series", grain="week", as_of=as_of,
                                   window_start=start, window_end=end, filters=filters))
    if isinstance(r, Refusal):
        return {"refused": r.code.value, "flagged": {}, "cited": []}
    pts = [(p["t"][:10], p["value"]) for p in r.observations[0].payload if p["value"] is not None]
    vals = [v for _, v in pts]
    mu = median(vals)
    sd = max(1.4826 * median([abs(v - mu) for v in vals]), 1e-9)
    flagged = {t: v - mu for t, v in pts if (v - mu) / sd >= z_threshold}
    return {"flagged": flagged, "baseline": mu, "cited": [r.observations[0].id], "certainty": "observational association"}


class Diagnosis(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    metric: str
    classification: str              # "measurement_suspected" | "real_change" | "no_change" | "undetermined"
    periods: list[str]
    rationale: str
    cited_observations: list[str]


def _weekly(port, var, start, end, as_of):
    r = port.query(EvidenceRequest(variable_id=var, shape="series", grain="week", as_of=as_of, window_start=start, window_end=end))
    if isinstance(r, Refusal):
        return None, {}
    return r.observations[0].id, {p["t"][:10]: p["value"] for p in r.observations[0].payload if p["value"] is not None}


def _robust_z(series: dict) -> dict:
    vals = list(series.values())
    mu = median(vals)
    sd = max(1.4826 * median([abs(v - mu) for v in vals]), 1e-9)
    return {t: (v - mu) / sd for t, v in series.items()}


def diagnose_change(port: EvidenceAccessPort, metric: str, start: datetime, end: datetime, as_of: datetime,
                    z_flag: float = 3.0, z_quiet: float = 2.0) -> Diagnosis:
    """Is a change in `metric` real, or a measurement problem? Uses only declared empirical identities
    (MetricDefinition.empirical_identities, 'a ~ b'): if the metric moves but its identity partners do not,
    the identity is broken -> measurement suspected; if partners move the same way -> real change; else abstain."""
    defs = {v.id: v for v in port.variables()}
    oid, series = _weekly(port, metric, start, end, as_of)
    if not series:
        return Diagnosis(metric=metric, classification="undetermined", periods=[], rationale="no data", cited_observations=[])
    z = _robust_z(series)
    flagged = [t for t, v in z.items() if abs(v) >= z_flag]
    if not flagged:
        return Diagnosis(metric=metric, classification="no_change", periods=[], rationale="no period beyond the robust threshold",
                         cited_observations=[oid])
    partners = []
    for ident in getattr(defs.get(metric), "empirical_identities", []) or []:
        lhs, _, rest = ident.partition("~")
        cand = rest.strip().split(" ")[0]
        if lhs.strip() == metric and cand in defs:
            partners.append(cand)
    if not partners:
        return Diagnosis(metric=metric, classification="undetermined", periods=flagged,
                         rationale="change found, but no declared identity to cross-check it against", cited_observations=[oid])
    cited, votes = [oid], []
    for pt in partners:
        poid, ps = _weekly(port, pt, start, end, as_of)
        cited.append(poid)
        pz = _robust_z(ps) if ps else {}
        for t in flagged:
            same_dir = t in pz and abs(pz[t]) >= z_flag and (pz[t] > 0) == (z[t] > 0)
            quiet = t in pz and abs(pz[t]) < z_quiet
            votes.append("real" if same_dir else "measurement" if quiet else "unclear")
    if votes and all(v == "measurement" for v in votes):
        cls, why = "measurement_suspected", f"{metric} moved but its identity partner(s) {partners} did not: the identity is broken"
    elif votes and all(v == "real" for v in votes):
        cls, why = "real_change", f"{metric} and its identity partner(s) {partners} moved together"
    else:
        cls, why = "undetermined", f"identity partners gave mixed evidence ({votes}): abstaining"
    return Diagnosis(metric=metric, classification=cls, periods=flagged, rationale=why, cited_observations=[c for c in cited if c])


class FunnelLocalisation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    metric: str
    classification: str        # "no_change" | "localised" | "undetermined" | "cannot_localise"
    stage: str | None          # "<prev> -> <stage>" for a broken transition, or "upstream of <root>"
    periods: list[str]
    rationale: str
    cited_observations: list[str]


def localise_funnel(port: EvidenceAccessPort, metric: str, start: datetime, end: datetime, as_of: datetime,
                    z_flag: float = 3.0, z_quiet: float = 2.0) -> FunnelLocalisation:
    """Walk the declared funnel ("funnel_after: <previous stage>" identities) back from `metric`.
    For each period where the metric moved, report the first transition whose ratio broke while the others
    stayed quiet; if no ratio broke but the root level moved the same way, the change is upstream of the root."""
    defs = {v.id: v for v in port.variables()}
    chain = [metric]
    while True:
        prev = [i.split(":", 1)[1].strip() for i in (getattr(defs.get(chain[0]), "empirical_identities", []) or [])
                if i.startswith("funnel_after:")]
        if not prev or prev[0] not in defs or prev[0] in chain:
            break
        chain.insert(0, prev[0])
    oid, series = _weekly(port, metric, start, end, as_of)
    if not series:
        return FunnelLocalisation(metric=metric, classification="undetermined", stage=None, periods=[], rationale="no data", cited_observations=[])
    z = _robust_z(series)
    flagged = [t for t, v in z.items() if abs(v) >= z_flag]
    if not flagged:
        return FunnelLocalisation(metric=metric, classification="no_change", stage=None, periods=[], rationale="no period beyond the threshold",
                                  cited_observations=[oid])
    if len(chain) == 1:
        return FunnelLocalisation(metric=metric, classification="cannot_localise", stage=None, periods=flagged,
                                  rationale="the metric moved, but no funnel stages are declared for it", cited_observations=[oid])
    cited, lv = [], {}
    for st in chain:
        o, sr = _weekly(port, st, start, end, as_of)
        cited.append(o); lv[st] = sr
    weeks = sorted(set.intersection(*(set(s) for s in lv.values())))
    ratio_z = {}
    for a, b in zip(chain, chain[1:]):
        ratio_z[f"{a} -> {b}"] = _robust_z({w: lv[b][w] / lv[a][w] for w in weeks if lv[a][w]})
    root_z = _robust_z({w: lv[chain[0]][w] for w in weeks})
    verdicts = []
    for t in flagged:
        broken = [k for k, rz in ratio_z.items() if abs(rz.get(t, 0)) >= z_flag]
        quiet = [k for k, rz in ratio_z.items() if abs(rz.get(t, 0)) < z_quiet]
        if len(broken) == 1 and len(quiet) == len(ratio_z) - 1:
            verdicts.append(broken[0])
        elif not broken and len(quiet) == len(ratio_z) and abs(root_z.get(t, 0)) >= z_flag and (root_z[t] > 0) == (z[t] > 0):
            verdicts.append(f"upstream of {chain[0]}")
        else:
            verdicts.append(None)
    if verdicts and all(v is not None and v == verdicts[0] for v in verdicts):
        return FunnelLocalisation(metric=metric, classification="localised", stage=verdicts[0], periods=flagged,
                                  rationale=f"funnel {' -> '.join(chain)}: only '{verdicts[0]}' moved in the flagged period(s)",
                                  cited_observations=[c for c in [oid, *cited] if c])
    return FunnelLocalisation(metric=metric, classification="undetermined", stage=None, periods=flagged,
                              rationale=f"funnel evidence mixed across flagged periods ({verdicts}): abstaining",
                              cited_observations=[c for c in [oid, *cited] if c])
