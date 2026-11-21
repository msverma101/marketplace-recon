"""P1 look-alike pair (scenario 8): logging failure vs real demand drop. Score the stand-in reader's
diagnosis against the oracle's cause layer, with a no-intervention control. Not a gate."""
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
W0, W1 = datetime(2018, 2, 12), datetime(2018, 2, 19)
SCEN = {
    "8a_logging_drop": InterventionSpec(id="8a", target="logging.drop_rate", scope="event_type == 'order_created'", operator="set",
                                        value=0.4, window_start=W0, window_end=W1, requested_by="scenario"),
    "8b_demand_drop": InterventionSpec(id="8b", target="demand.multiplier", scope="", operator="scale",
                                       value=0.6, window_start=W0, window_end=W1, requested_by="scenario"),
    "control": None,
}
EXPECT = {"observation": "measurement_suspected", "world": "real_change", None: "no_change"}
rows = []
for name, sp in SCEN.items():
    for seed in (31, 32, 33, 34, 35):
        _, f, t, port, truth = twin.run(p, date(2018, 1, 1), date(2018, 4, 1), seed, sp)
        d = standin_reader.diagnose_change(port, "orders_count", datetime(2018, 1, 1), datetime(2018, 4, 1), datetime(2018, 5, 1))
        layer = truth.causes[0]["layer"] if truth else None
        rows.append({"scenario": name, "seed": seed, "truth_layer": layer, "world_changed": truth.world_changed if truth else False,
                     "classification": d.classification, "periods": d.periods, "correct": d.classification == EXPECT[layer],
                     "rationale": d.rationale})
pathlib.Path("docs/p0/p1_lookalike_score.json").write_text(json.dumps(rows, indent=1))
for name in SCEN:
    rs = [r for r in rows if r["scenario"] == name]
    print(f"{name:16} correct {sum(r['correct'] for r in rs)}/5   " + " | ".join(f"{r['classification']} {r['periods']}" for r in rs))
