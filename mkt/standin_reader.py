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
