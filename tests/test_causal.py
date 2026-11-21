"""Causal-fidelity tests on a small simulated world (docs §12.1 'Causal')."""
from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from mkt.contracts import Intervention
from mkt.observe import observe
from mkt.sim.engine import simulate

START = datetime(2018, 3, 1)
DAYS = 6
OUT = (datetime(2018, 3, 3, 10), datetime(2018, 3, 3, 18))


def outage(value=0.6, op="set"):
    return Intervention(id="o", layer="world", target="infra.checkout.error_rate", operator=op, value=value,
                        selector="device == 'mobile'", window=OUT)


def _cnt(ev, etype, device=None, window=OUT):
    f = ev.filter((pl.col("event_type") == etype) & pl.col("event_time").is_between(*window, closed="left"))
    if device:
        f = f.filter(pl.col("device") == device)
    return f.height


def test_deterministic(world):
    a = simulate(world, START, 2).events
    b = simulate(world, START, 2).events
    assert_frame_equal(a, b)


def test_placebo_equals_twin(world):
    twin = simulate(world, START, DAYS).events
    placebo = simulate(world, START, DAYS, [outage(1.0, "scale")]).events
    assert_frame_equal(twin, placebo)


def test_do_changes_only_descendants(world):
    twin = simulate(world, START, DAYS).events
    fact = simulate(world, START, DAYS, [outage()]).events
    # sessions are not descendants of checkout errors: identical, including timing
    assert_frame_equal(twin.filter(pl.col("event_type") == "session_start").drop("event_id"),
                       fact.filter(pl.col("event_type") == "session_start").drop("event_id"))
    # before the intervention, everything is identical
    pre = lambda e: e.filter(pl.col("event_time") < OUT[0]).drop("event_id")
    assert_frame_equal(pre(twin), pre(fact))
    # descendants move in the right direction, only for mobile
    assert _cnt(fact, "checkout_error", "mobile") > 5 * max(_cnt(twin, "checkout_error", "mobile"), 1)
    assert _cnt(fact, "order_created", "mobile") < _cnt(twin, "order_created", "mobile")
    assert _cnt(fact, "checkout_error", "desktop") == _cnt(twin, "checkout_error", "desktop")


def test_observation_intervention_leaves_world_untouched(world):
    drop = Intervention(id="d", layer="observation", target="logging.drop_rate", operator="set", value=0.5,
                        selector="event_type == 'order_created'", window=(datetime(2018, 3, 2), datetime(2018, 3, 5)))
    plain = simulate(world, START, DAYS).events
    with_drop = simulate(world, START, DAYS, [drop]).events          # world layer ignores observation ivs
    assert_frame_equal(plain, with_drop)
    obs = observe(plain, seed=1, interventions=[drop])
    w = (datetime(2018, 3, 2), datetime(2018, 3, 5))
    seen = _cnt(obs.unique("event_id"), "order_created", window=w)
    real = _cnt(plain, "order_created", window=w)
    assert 0.35 * real < seen < 0.65 * real
    # finance stream is unaffected: reconciliation evidence survives
    assert _cnt(obs.unique("event_id"), "payment_approved", window=w) == _cnt(plain, "payment_approved", window=w)


def test_carrier_slowdown_raises_lateness_and_lowers_scores(world):
    iv = Intervention(id="c", layer="world", target="carrier.transit_multiplier", operator="scale", value=3.0,
                      selector="region in ['SP', 'RJ', 'MG']", window=(datetime(2018, 3, 1), datetime(2018, 3, 8)))
    twin = simulate(world, START, 40)
    fact = simulate(world, START, 40, [iv])
    # measure where the cause acts: orders whose shipment got the slower carrier (diluted otherwise)
    hit = fact.orders.filter(pl.col("transit_multiplier") > 1)
    assert hit.height > 50
    assert (hit["late_days_true"] > 0).mean() > (twin.orders["late_days_true"] > 0).mean() + 0.15
    assert twin.orders["transit_multiplier"].max() == 1.0      # the twin never sees the intervention
    score = lambda r: r.events.filter(pl.col("event_type") == "review_submitted")["value"].mean()
    assert score(fact) < score(twin)
