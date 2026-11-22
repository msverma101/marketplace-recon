"""M4/M5 without a running shop (fake HTTP), plus an opt-in live run (MKT_STOREFRONT=1, shop on :8080)."""
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from causal_core.contracts.capabilities import CapabilityResponse, GetAggregate, GetSeries
from causal_core.contracts.common import TimeWindow
from mkt.adapters.astronomy import storefront as sf
from mkt.adapters.astronomy.telemetry import TelemetryPort, WindowMap

U = timezone.utc
CAT = pl.DataFrame({"public_id": ["AAAAAAAAAA", "BBBBBBBBBB"], "weight": [0.7, 0.3]})


def fake_shop(outcomes):
    calls = []

    def call(method, path, body=None, baggage="", timeout=20):
        calls.append((method, path.split("?")[0], baggage))
        if path.startswith("/api/checkout"):
            code = outcomes.pop(0)
            return code, ({"orderId": f"o-{len(calls)}"} if code == 200 else None), 0.01
        return 200, {}, 0.01
    return call, calls


def plans(n):
    return [{"command_id": f"c{i}", "order_idx": i, "session_alias": f"s{i}"} for i in range(n)]


def test_driver_outcomes_baggage_and_idempotency(tmp_path, monkeypatch):
    call, calls = fake_shop([200, 422, 500, -1])
    monkeypatch.setattr(sf, "_call", call)
    d = sf.SessionDriver(CAT, tmp_path / "r.jsonl", {"email": "x@example.com"}, seed=1)
    res = d(plans(4))
    assert [res[f"c{i}"]["outcome"] for i in range(4)] == ["ok", "payment_failed", "checkout_failed", "timeout"]
    assert all("synthetic_request=true" in b and "mkt.command_id=" in b for _, _, b in calls)
    n = len(calls)
    again = d(plans(4))                                              # replay: nothing re-executed
    assert len(calls) == n and again == res
    d2 = sf.SessionDriver(CAT, tmp_path / "r.jsonl", {"email": "x@example.com"}, seed=1)   # durable across restarts
    assert d2(plans(4)) == res and len(calls) == n


def test_product_choice_is_keyed():
    a = sf.SessionDriver(CAT, pathlib.Path("/dev/null"), {}, seed=3)
    assert [a.product_for(i) for i in range(50)] == [a.product_for(i) for i in range(50)]


def _spans_file(tmp_path, t0):
    def span(svc, k, err):
        s = int((t0 + timedelta(seconds=k)).timestamp() * 1e9)
        return {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": svc}}]},
                                   "scopeSpans": [{"spans": [{"kind": 2, "startTimeUnixNano": str(s), "endTimeUnixNano": str(s + 5_000_000),
                                                              "status": {"code": 2 if err else 0}}]}]}]}
    f = tmp_path / "spans.jsonl"
    f.write_text("\n".join(json.dumps(span("payment", k, k % 4 == 0)) for k in range(40)) + "\n")
    return f


def test_telemetry_port_conforms(tmp_path):
    from mkt.adapters.astronomy.telemetry import load_spans
    t0 = datetime(2026, 11, 22, 8, 0, tzinfo=U)
    tp = TelemetryPort(load_spans(_spans_file(tmp_path, t0)),
                       [WindowMap(t0, t0 + timedelta(seconds=40), datetime(2018, 2, 21, tzinfo=U), datetime(2018, 2, 21, 0, 40, tzinfo=U))],
                       seed=1, generator_version="test")
    w = TimeWindow(start=datetime(2018, 2, 21, tzinfo=U), end=datetime(2018, 2, 22, tzinfo=U))
    base = dict(schema_version="2.0", request_id="t", scope=tp.scope, data_cut=datetime(2018, 2, 22, tzinfo=U), window=w, purpose="t")
    r = tp.query(GetAggregate(**base, variables=("payment.error_rate",), request="get_aggregate"))
    assert r.status == "OK" and r.observations[0].payload.numerator == 10 and r.observations[0].payload.denominator == 40
    s = tp.query(GetSeries(**base, variables=("payment.requests",), request="get_series", grain="1m"))
    assert s.status == "OK" and sum(p.value for p in s.observations[0].payload.points) == 40
    for x in (r, s):
        assert CapabilityResponse.model_validate_json(x.model_dump_json()) == x
    assert tp.evidence_record(r.observations[0]).provenance_role == "SANDBOX_OBSERVED"


@pytest.mark.skipif(os.environ.get("MKT_STOREFRONT") != "1", reason="live shop run: set MKT_STOREFRONT=1 with the stack up")
def test_live_scenario5():
    root = pathlib.Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-I", "scripts/storefront_s5.py"], cwd=root, capture_output=True, text=True, timeout=1800)
    rep = json.loads((root / "docs/experiments/storefront_s5.json").read_text())
    assert rep["all_pass"], rep["checks"]
