"""Evaluator: score monitor alerts against the privileged truth (the only module that reads truth/)."""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timedelta


def _episodes(alerts: list[dict], gap_h: float = 3) -> list[dict]:
    """Merge consecutive alerts on the same (metric, segment) within gap_h hours."""
    eps, last = [], {}
    for a in sorted(alerts, key=lambda a: a["time"]):
        k = (a["metric"], a["segment"])
        t = datetime.fromisoformat(a["time"])
        if k in last and t - last[k]["end"] <= timedelta(hours=gap_h):
            last[k]["end"] = t
            last[k]["n"] += 1
            continue
        ep = {**a, "start": t, "end": t, "n": 1}
        eps.append(ep)
        last[k] = ep
    return eps


def evaluate(run: pathlib.Path, alerts: list[dict]) -> dict:
    run = pathlib.Path(run)
    truth = json.loads((run / "truth" / "ground_truth.json").read_text())
    man = json.loads((run / "observable" / "manifest.json").read_text())
    eps = _episodes(alerts)
    matched_eps = set()
    causes = []
    for c in [t for t in truth if t.get("kind") == "cause"]:
        lo = datetime.fromisoformat(c["window"][0])
        hi = datetime.fromisoformat(c["window"][1]) + timedelta(hours=c.get("max_lag_h", 0))
        sym = {tuple(s) for s in c.get("symptoms", [])}
        hits = [i for i, e in enumerate(eps) if (e["metric"], e["segment"]) in sym and lo - timedelta(hours=1) <= e["start"] <= hi]
        matched_eps.update(hits)
        first = min((eps[i]["start"] for i in hits), default=None)
        attributed = [eps[i] for i in hits if eps[i].get("cause") == c["target"]]
        causes.append({
            "target": c["target"], "identifiable": c.get("identifiable"),
            "detected": bool(hits), "detection_delay_h": round((first - lo).total_seconds() / 3600, 2) if first else None,
            "symptoms_hit": sorted({f"{eps[i]['metric']}|{eps[i]['segment']}" for i in hits}),
            "attributed_correctly": bool(attributed),
            "abstained": bool(hits) and all(eps[i].get("cause") in ("unknown", "abstain") for i in hits),
        })
    span_days = max((datetime.fromisoformat(man["available_to"][:19]) - datetime.fromisoformat(man["available_from"][:19])).days, 1)
    fa = [e for i, e in enumerate(eps) if i not in matched_eps]
    return {
        "scenario": man["scenario"], "alerts": len(alerts), "episodes": len(eps),
        "causes": causes,
        "false_alarm_episodes": len(fa), "false_alarms_per_day": round(len(fa) / span_days, 3),
        "false_alarm_examples": [{k: str(v) for k, v in e.items() if k in ("start", "metric", "segment", "z")} for e in fa[:5]],
    }
