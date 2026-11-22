"""P1-2 gate S10 (preregistered): scenarios 9a checkout failure vs 9b demand drop, with and without the session layer."""
import json
import pathlib
import sys
from datetime import date, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mkt import pipeline, standin_reader  # noqa: E402
from mkt.config import Config  # noqa: E402
from mkt.sim import chain, twin  # noqa: E402
from mkt.sim.sessions import SessionParams  # noqa: E402
from mkt.v3 import InterventionSpec  # noqa: E402

p = chain.fit(pipeline.canonical_olist(Config.load()))
sp = SessionParams.from_cards()
W0, W1 = datetime(2018, 2, 12), datetime(2018, 2, 19)
SCEN = {"9a_checkout_failure": InterventionSpec(id="9a", target="checkout.failure_rate", scope="", operator="set", value=0.4,
                                                window_start=W0, window_end=W1, requested_by="scenario"),
        "9b_demand_drop": InterventionSpec(id="9b", target="demand.multiplier", scope="", operator="scale", value=0.6,
                                           window_start=W0, window_end=W1, requested_by="scenario"),
        "control": None}
EXPECT = {"9a_checkout_failure": ("localised", "checkouts_count -> orders_count"),
          "9b_demand_drop": ("localised", "upstream of sessions_count"), "control": ("no_change", None)}
POSTHOC = "--posthoc" in sys.argv          # count-floor reader on NEW seeds; supplementary, not the gate
SEEDS = (71, 72, 73, 74, 75) if POSTHOC else (61, 62, 63, 64, 65)
rows = []
for layer in ((True,) if POSTHOC else (True, False)):
    for name, spec in SCEN.items():
        for seed in SEEDS:
            _, _, _, port, _ = twin.run(p, date(2018, 1, 1), date(2018, 4, 2), seed, spec, session_params=sp if layer else None)
            r = standin_reader.localise_funnel(port, "orders_count", datetime(2018, 1, 1), datetime(2018, 4, 2), datetime(2018, 5, 1),
                                               count_floor=POSTHOC)
            rows.append({"layer": layer, "scenario": name, "seed": seed, "classification": r.classification, "stage": r.stage,
                         "periods": r.periods, "correct": (r.classification, r.stage) == EXPECT[name], "rationale": r.rationale})
            print(layer, name, seed, r.classification, r.stage, r.periods, flush=True)
pathlib.Path("docs/p0/p1_2_s10_posthoc.json" if POSTHOC else "docs/p0/p1_2_s10_score.json").write_text(json.dumps(rows, indent=1))
for layer in ((True,) if POSTHOC else (True, False)):
    print("with layer" if layer else "without layer",
          {n: sum(r["correct"] for r in rows if r["layer"] == layer and r["scenario"] == n) for n in SCEN})
