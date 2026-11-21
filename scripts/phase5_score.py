"""Phase 5: score the stand-in reader against the evaluator-only oracle (first independent score; not a gate)."""
import json
import pathlib
import sys
from datetime import date, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mkt import pipeline, standin_reader  # noqa: E402
from mkt.config import Config  # noqa: E402
from mkt.sim import chain, twin  # noqa: E402
from mkt.v3 import InterventionSpec  # noqa: E402

p = chain.fit(pipeline.canonical_olist(Config.load()))
spec = InterventionSpec(id="iv-rj-carrier", target="carrier.transit_multiplier", scope="region == 'RJ'", operator="scale",
                        value=2.5, window_start=datetime(2018, 2, 5), window_end=datetime(2018, 2, 12), requested_by="scenario")
args = dict(metric="late_share_cohort", filters={"region": "RJ"}, start=datetime(2018, 1, 1), end=datetime(2018, 4, 1),
            as_of=datetime(2018, 5, 15))
rows = []
for seed in (21, 22, 23, 24, 25):
    _, _, _, port, truth = twin.run(p, date(2018, 1, 1), date(2018, 4, 1), seed, spec)
    _, _, _, ctrl_port, _ = twin.run(p, date(2018, 1, 1), date(2018, 4, 1), seed, None)       # negative control
    est = standin_reader.localise_change(port, **args)
    ctrl = standin_reader.localise_change(ctrl_port, **args)
    true_weeks = {w for w, d in truth.effect_late_share_region_weeks.items() if d >= 0.05}
    got = set(est["flagged"])
    tp = got & true_weeks
    err = [abs(est["flagged"][w] - truth.effect_late_share_region_weeks[w]) for w in tp]
    rows.append({"seed": seed, "true_weeks": sorted(true_weeks), "flagged": sorted(got),
                 "precision": len(tp) / len(got) if got else None, "recall": len(tp) / len(true_weeks) if true_weeks else None,
                 "effect_abs_error": sum(err) / len(err) if err else None,
                 "true_effects": {w: round(truth.effect_late_share_region_weeks[w], 3) for w in sorted(true_weeks)},
                 "estimated": {w: round(v, 3) for w, v in sorted(est["flagged"].items())},
                 "control_false_alarms": len(ctrl["flagged"])})
out = pathlib.Path("docs/p0/phase5_score.json")
out.write_text(json.dumps(rows, indent=1))
for r in rows:
    print(r["seed"], "true", r["true_weeks"], "flagged", r["flagged"], "P", r["precision"], "R", r["recall"],
          "abs err", None if r["effect_abs_error"] is None else round(r["effect_abs_error"], 3), "control FA", r["control_false_alarms"])
