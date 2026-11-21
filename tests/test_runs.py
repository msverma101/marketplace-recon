"""End-to-end: scenario run -> integrity/leakage checks -> replay -> evaluation; plus fidelity vs held-out Olist."""
from datetime import datetime, timedelta

import polars as pl
import pytest

from mkt import evaluate, monitor, pipeline, scenario, validate
from mkt.replay import FutureLeak, Replayer
from mkt.sim.engine import simulate


@pytest.fixture(scope="module")
def s01_run(cfg, params, tmp_path_factory):
    from dataclasses import replace
    c = replace(cfg, runs=tmp_path_factory.mktemp("runs"))
    return scenario.run(c, "s01")


def test_run_passes_integrity_and_leakage(s01_run):
    res = validate.check_run(s01_run)
    assert all(ok for ok, _ in res.values()), {k: d for k, (ok, d) in res.items() if not ok}
    obs_files = sorted(p.name for p in (s01_run / "observable").iterdir())
    assert obs_files == ["events.parquet", "manifest.json"]


def test_reruns_are_identical(s01_run, cfg, tmp_path):
    from dataclasses import replace
    again = scenario.run(replace(cfg, runs=tmp_path), "s01")
    a = pl.read_parquet(s01_run / "observable" / "events.parquet")
    b = pl.read_parquet(again / "observable" / "events.parquet")
    assert a.equals(b)


def test_replay_serves_everything_once_and_never_early(s01_run):
    r = Replayer(s01_run)
    n = 0
    for clock, f in r.windows(timedelta(minutes=30)):
        assert f.height == 0 or f["available_at"].max() <= clock
        n += f.height
    assert n == r.df.height


def test_replayer_detects_future_leak(s01_run, monkeypatch):
    r = Replayer(s01_run)
    r._t = r._t.copy()
    r._t[5] = r._t[-1] + 10**12          # corrupt: an event far in the future inside an early window
    with pytest.raises(FutureLeak):
        for _ in r.windows(timedelta(hours=1)):
            pass


def test_effects_are_factual_minus_twin(s01_run):
    eff = pl.read_parquet(s01_run / "truth" / "effects_hourly.parquet")
    m = eff.filter((pl.col("metric") == "order_created") & (pl.col("dim_value") == "mobile")
                   & pl.col("hour").is_between(datetime(2018, 3, 15, 19), datetime(2018, 3, 15, 21)))
    assert (m["factual"] - m["twin"]).sum() < 0
    s = eff.filter(pl.col("metric") == "session_start")
    assert (s["factual"] == s["twin"]).all()


def test_baseline_monitor_detects_outage(s01_run):
    rep = evaluate.evaluate(s01_run, monitor.run(Replayer(s01_run)))
    c = rep["causes"][0]
    assert c["detected"] and c["detection_delay_h"] <= 4
    assert c["abstained"] and not c["attributed_correctly"]      # baseline never attributes


def test_fidelity_against_heldout_olist(cfg, params, world):
    """Generated fulfilment & reviews vs Olist 2018 hold-out (fit used only 2017)."""
    hold = pipeline.holdout(cfg)
    r = simulate(world, datetime(2018, 3, 1), 60)
    o = r.events.filter(pl.col("event_type") == "order_created").select("order_id", pl.col("event_time").alias("t0"))
    o = o.filter(pl.col("t0") < datetime(2018, 3, 21))
    d = r.events.filter(pl.col("event_type") == "delivered").join(o, on="order_id")
    days = ((d["event_time"] - d["t0"]).dt.total_hours() / 24)
    late = r.orders.join(o, left_on="order_id", right_on="order_id")["late_days_true"]
    # documented tolerance: the 2017 fit is slower than 2018 (README 'Known fidelity gaps')
    assert abs(days.median() - hold["delivery_days_q"][2]) / hold["delivery_days_q"][2] < 0.35
    assert 0.5 * hold["late_share"] < (late > 0).mean() < 2.0 * hold["late_share"]
    rv = r.events.filter(pl.col("event_type") == "review_submitted")["value"].mean()
    real = sum(v["mean"] * v["n"] for v in hold["score_by_bucket"].values()) / sum(v["n"] for v in hold["score_by_bucket"].values())
    assert abs(rv - real) < 0.3
