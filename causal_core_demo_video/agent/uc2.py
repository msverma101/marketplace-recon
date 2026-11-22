"""Video 2 (demo-only): an autonomous agent watches a marketplace that runs through the real OpenTelemetry Astronomy
Shop, and acts only through a sandbox.

World: marketplace-recon's emulator (order chain fitted on real Olist 2017 orders, checked on 2018) decides who buys
what and when; its buying sessions are played through the real shop's HTTP API (product → cart → checkout → payment),
so the real microservices produce real, redacted OpenTelemetry spans. The scenario (not the agent) injects a payment
fault through the emulator's gateway and the shop's flag service. The agent sees only public telemetry through
marketplace-recon's read-only telemetry port, runs the causal core after every 3-hour block, and proposes; a scripted
on-call approver approves; the approved change runs in the local shop's flag file (the sandbox), and is read back.

Every core task is traced (stages and their timings, capability requests, methods chosen or refused, results,
report revisions, policy verdicts, ledger rows written) for the "inside the engine" panel.

    cd causal_core_demo_video && ~/.venvs/mkt-emulator/bin/python -m agent.uc2      # shop must be running (README)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CORE = Path(os.environ.get("UC2_CORE") or HERE.parent / "causal_core")            # the causal core (read-only use)
MKT = Path(os.environ.get("UC2_MKT") or next((p for p in (HERE.parent.parent / "marketplace-recon", HERE.parent / "marketplace-recon")
                                              if p.exists()), HERE.parent / "marketplace-recon"))   # the marketplace emulator
sys.path[:0] = [str(CORE), str(MKT)]

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from adapters.outputs.sqlite_sinks import IdempotentSink  # noqa: E402
from adapters.sources.canonical_port import CanonicalEvidencePort  # noqa: E402
from causal_core import api  # noqa: E402
from causal_core.contracts.capabilities import GetAggregate  # noqa: E402
from causal_core.contracts.common import TimeWindow  # noqa: E402
from causal_core.orchestration.coordinator import TaskCoordinator  # noqa: E402
from causal_core.orchestration.planner import MonitorSpec, TaskContext  # noqa: E402
from causal_core.storage.ledger import Ledger  # noqa: E402
from mkt import pipeline  # noqa: E402
from mkt.adapters.astronomy import storefront as sf  # noqa: E402
from mkt.adapters.astronomy.telemetry import TelemetryPort, WindowMap, load_spans  # noqa: E402
from mkt.config import Config  # noqa: E402
from mkt.emulator.api import Emulator, EmulatorConfig  # noqa: E402
from mkt.engine_types import InterventionSpec  # noqa: E402
from mkt.sim import chain  # noqa: E402

from .sandbox import ActionProposal, Approval, LocalShopFlags  # noqa: E402

U = timezone.utc
HOME = Path.home()
OUT = HERE / "outputs" / "uc2"
RUN = HOME / "data/mkt/runs/demo-uc2"
SPANS = HOME / "data/mkt/astronomy/live/spans.jsonl"
FLAGS = HOME / "data/mkt/astronomy/flags/demo.flagd.json"
PEOPLE = HOME / ".cache/otel-demo-3.1.0/src/load-generator/people.json"
SEED = 21
NORMAL_DAYS = [date(2018, 2, 12) + timedelta(days=i) for i in range(4)]
FAULT_DAY, RECOVERY_DAY = date(2018, 2, 16), date(2018, 2, 17)
BLOCK_H = 3
FLUSH_S = float(os.environ.get("UC2_FLUSH_S", "6.5"))  # services batch spans (Go SDK: 5 s) before the collector's file export
P0 = 0.01                                     # declared error budget for payments (the null's noise scale)
APPROVER, APPROVAL_DELAY = "on-call marketplace lead", timedelta(minutes=15)
SCOPE = {"environment_id": "demo-shop", "environment_kind": "emulator", "generator_version": "demo-uc2/0.1"}
STAGES = ("FRAME", "AUDIT_INPUTS", "SELECT_CAPABILITIES", "ACQUIRE", "EVALUATE", "CHALLENGE", "REPORT")


def iso(t: datetime) -> str:
    return t.astimezone(U).isoformat().replace("+00:00", "Z")


class Console:
    def __init__(self):
        self.lines = []
        self.hook = None

    def __call__(self, sim: datetime, who: str, text: str):
        line = {"sim": iso(sim), "wall": iso(datetime.now(U)), "who": who, "text": text}
        self.lines.append(line)
        print(f"{line['sim'][:16].replace('T', ' ')}  {who:<8} {text}", flush=True)
        if self.hook:
            self.hook(line)


class Live:
    """outputs/uc2/live.json for the live dashboard (scenes/uc2_live.html), plus a snapshot at every key event."""

    def __init__(self, out: Path):
        self.out, self.n, self.state, self._t = out, 0, {}, 0.0
        (out / "snapshots").mkdir(parents=True, exist_ok=True)

    def write(self, force: bool = False, snapshot: str | None = None):
        now = time.monotonic()
        if not force and snapshot is None and now - self._t < 0.25:
            return
        self._t = now
        self.state["wall"] = iso(datetime.now(U))
        body = json.dumps(self.state, default=str)
        tmp = self.out / "live.tmp"
        tmp.write_text(body)
        os.replace(tmp, self.out / "live.json")
        if snapshot:
            self.n += 1
            (self.out / "snapshots" / f"{self.n:03d}_{snapshot}.json").write_text(body)


class SpanTail:
    """Incremental reader of the collector's span export: parses only lines appended since this run started (same
    parsing as marketplace-recon's load_spans: SERVER spans → service, start, duration, error)."""

    def __init__(self, path: Path):
        self.path, self.off = path, path.stat().st_size
        self.rows: list[tuple] = []

    def read(self) -> pl.DataFrame:
        with open(self.path, "rb") as f:
            f.seek(self.off)
            data = f.read()
        end = data.rfind(b"\n")
        if end >= 0:
            self.off += end + 1
            for line in data[:end + 1].splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                for rs in rec.get("resourceSpans", []):
                    svc = next((a["value"].get("stringValue") for a in rs.get("resource", {}).get("attributes", [])
                                if a["key"] == "service.name"), "?")
                    for ss in rs.get("scopeSpans", []):
                        for sp in ss.get("spans", []):
                            if sp.get("kind") != 2:
                                continue
                            t0, t1 = int(sp["startTimeUnixNano"]), int(sp["endTimeUnixNano"])
                            self.rows.append((svc, t0, (t1 - t0) / 1e6, sp.get("status", {}).get("code", 0) == 2))
        return pl.DataFrame(self.rows, schema=["service", "t_ns", "dur_ms", "error"], orient="row")


class TracingPort(CanonicalEvidencePort):
    """The core's evidence port for this run, with every capability request logged for the engine panel."""

    def __init__(self, observations, provider_id):
        super().__init__(observations, provider_id)
        self.log = []

    def query(self, request):
        t0 = time.perf_counter()
        r = super().query(request)
        self.log.append({"request": request.request, "variables": list(getattr(request, "variables", ()) or ()),
                         "status": r.status, "observations": len(r.observations), "ms": round((time.perf_counter() - t0) * 1000, 3)})
        return r


def obs_point(var: str, key: str, value: float, t_end: datetime, extra: dict) -> dict:
    payload = {"shape": "time_series", "points": [{"t": iso(t_end), "value": value}]}
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"schema_version": "2.0", "observation_id": f"o:{key}", "variable_id": var, "payload": payload, "source_id": "storefront-telemetry",
            "record_key": key, "revision": 1, "scope": SCOPE,
            "lineage": {"source_id": "storefront-telemetry", "origin_kind": "original", "lineage_root_id": f"otel:{key}",
                        "content_hash": h, **extra},
            "clocks": {"observed_at": {"quality": "KNOWN", "value": iso(t_end)},
                       "available_at": {"quality": "KNOWN", "value": iso(t_end + timedelta(minutes=2))}}}


def task(tid: str, question: dict, cut: datetime, trigger: str = "monitor"):
    raw = {"schema_version": "2.0", "task_id": tid, "trigger": trigger, "source_scope": SCOPE, "priority": 5,
           "data_cut": {"as_of": iso(cut)}, "budget": {"max_wall_ms": 20_000}, "created_at": iso(cut), "question": question}
    if trigger == "monitor":
        raw["incident_id"] = f"inc-{tid}"
    return api.parse_reasoning_task(raw)


def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def run(pace_block: float = 0.0, event_pause: float = 0.0, paced_days: tuple[date, ...] = ()) -> int:
    assert SPANS.exists() and FLAGS.exists() and PEOPLE.exists(), "start the shop first (README: video 2)"
    for d in (OUT, RUN):
        shutil.rmtree(d, ignore_errors=True)
    OUT.mkdir(parents=True)
    say = Console()
    traces, blocks, events, actions = [], [], [], []
    live = Live(OUT)
    timeline = []
    cur = {"day": None}
    L = live.state
    L.update({"console": [], "sessions": [], "blocks": blocks, "trace": None, "tasks": 0, "ledger": {}, "actions": actions,
              "proposal": None, "alarm": False, "fixed": False, "phase": "setup", "paced": False, "sim": None,
              "flag": "off", "world": {}})
    KEY = {"engine": "engine", "policy": "policy", "human": "approval", "sandbox": "execution"}

    def on_line(line):
        L["console"] = (L["console"] + [line])[-16:]
        L["sim"] = line["sim"]
        kind = KEY.get(line["who"]) or ("proposal" if line["text"].startswith("PROPOSAL") else None)
        if kind:
            timeline.append({"wall": line["wall"], "sim": line["sim"], "kind": kind, "text": line["text"]})
        live.write(force=True, snapshot=kind)
        if kind and cur["day"] in paced_days:
            time.sleep(event_pause)
    say.hook = on_line

    # ── world: emulator fitted on Olist, catalog synced into the real shop ──
    t_setup = time.perf_counter()
    cfg = Config.load(MKT / "mkt.toml")
    canon = pipeline.canonical_olist(cfg)
    params = chain.fit(canon)
    cat = sf.build_catalog(canon, HOME / "data/mkt/raw", n=200)
    synced = sf.catalog_sync(cat)
    people = json.loads(PEOPLE.read_text())
    driver = sf.SessionDriver(cat, RUN / "privileged/driver_results.jsonl", people[0], SEED)
    flags = sf.FlagAdapter(FLAGS)
    assert flags.set_variant("paymentFailure", "off"), "flagd read-back failed"
    sandbox = LocalShopFlags(flags)
    emu = Emulator(params, RUN)
    emu.initialize(EmulatorConfig(start=date(2018, 1, 1), horizon_end=date(2018, 3, 15)), seed=SEED)
    emu.step(NORMAL_DAYS[0])
    world_meta = {"canonical": canon.name, "catalog_products_synced": synced, "orders_before_window": int(emu._world().height),
                  "setup_s": round(time.perf_counter() - t_setup, 1)}
    names = {r["public_id"]: (f"{r['category_en'].replace('_', ' ').title()} {r['public_id'][:4]}", round(r["price_usd"], 2))
             for r in cat.to_dicts()}
    L["world"] = world_meta | {"normal_days": [str(d) for d in NORMAL_DAYS], "fault_day": str(FAULT_DAY)}
    say(datetime.combine(NORMAL_DAYS[0], datetime.min.time(), U), "world",
        f"emulator ready: {world_meta['orders_before_window']:,} orders simulated since 1 Jan 2018; {synced} Olist products synced into the shop")

    # ── engine ──
    ledger = Ledger(RUN / "ledger.db")
    sink = IdempotentSink(RUN / "sink.db")
    observations: list[dict] = []
    att_null: dict[int, float] = {}                 # block-of-day → usual checkout attempts (frozen after the normal days)
    windows: list[WindowMap] = []
    tail = SpanTail(SPANS)                         # this run's spans only, read incrementally
    state = {"alarm": False, "fixed_at": None, "pending": []}

    def counts() -> dict:
        out = {}
        for db, label in ((ledger, "ledger"),):
            for (name,) in db.read("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
                out[f"{name}"] = db.read(f"SELECT COUNT(*) FROM {name}")[0][0]
        return out

    def core(tid: str, question: dict, cut: datetime, specs: dict, label: str, requested_action: str | None = None):
        port = TracingPort(observations, "demo-shop-telemetry")
        co = TaskCoordinator(ledger, port, sink, context=TaskContext(monitors=specs))
        before = counts()
        t0 = time.perf_counter()
        r = co.run(task(tid, question, cut), requested_action=requested_action) if requested_action else co.run(task(tid, question, cut))
        wall_ms = (time.perf_counter() - t0) * 1000
        after = counts()
        res = {}
        for m, x in r.state.results:
            v = getattr(x, "values", None) or {}
            res[m] = {"status": x.status, "refusal": getattr(x, "refusal_code", None),
                      "values": {k: v[k] for k in ("alarm", "direction", "log_wealth", "threshold", "points", "e_value") if k in v}}
        tr = {"task_id": tid, "label": label, "sim": iso(cut), "question": question["question_kind"],
              "variables": question.get("variables"), "outcome": r.state.outcome, "wall_ms": round(wall_ms, 2),
              "timings_ms": dict(r.timings_ms), "requests": port.log,
              "methods": [{"method": m, "status": st, "reason": (list(rs) or [""])[0][:160]} for m, st, rs in r.state.applicability],
              "results": res,
              "revisions": [{"revision": rep.get("revision"), "provisional": rep.get("provisional"),
                             "status": rep.get("status") or rep.get("outcome")} for rep in r.reports],
              "policy": [{"verdict": d.verdict, "reason": (list(d.reasons) or [""])[0][:200]} for d in r.decisions],
              "ledger_delta": {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)},
              "findings": [f.statement[:200] for f in r.state.findings][:3]}
        traces.append(tr)
        L["trace"], L["tasks"], L["ledger"] = tr, len(traces), after
        live.write(force=True)
        return r, tr

    def apply_due(sim_now: datetime):
        for item in list(state["pending"]):
            due, p = item
            if due <= sim_now:
                a = Approval.grant(p, APPROVER, due)
                say(due, "human", f"APPROVED {p.kind} {p.params['flag']}={p.params['variant']} ({APPROVER})")
                rec = sandbox.execute(p, a, due + timedelta(minutes=1))
                actions.append({"proposal": asdict(p), "approval": asdict(a), "receipt": asdict(rec)})
                events.append({"sim": iso(due), "kind": "approval", "detail": f"approved by {APPROVER}"})
                events.append({"sim": iso(due + timedelta(minutes=1)), "kind": "execution",
                               "detail": f"{rec.result} in {rec.sandbox}: {rec.state_after}"})
                say(due + timedelta(minutes=1), "sandbox", f"{rec.result}: flag {rec.state_after.get('flag')} "
                    f"{rec.state_after.get('before')} → {rec.state_after.get('after')} (read back through flagd: {rec.state_after.get('read_back_ok')})")
                state["fixed_at"] = due + timedelta(minutes=1)
                state["pending"].remove(item)
                L["fixed"], L["flag"] = True, rec.state_after.get("after")
                L["proposal"] = {**L["proposal"], "approval": asdict(a), "receipt": asdict(rec)} if L["proposal"] else None
                live.write(force=True, snapshot="executed")

    def watch(day: date, b: int, a: datetime, z: datetime, wall: tuple):
        # 1. adapter: telemetry for this block through marketplace-recon's read-only telemetry port
        if wall[1] > wall[0]:
            windows.append(WindowMap(wall[0], wall[1], a, z))
        time.sleep(FLUSH_S)
        t0 = time.perf_counter()
        spans = tail.read()
        tp = TelemetryPort(spans, windows, SEED, generator_version="storefront-3.1.0+" + emu.run_id)
        vals = {}
        for vid in ("payment.error_rate", "checkout.error_rate"):
            resp = tp.query(GetAggregate(schema_version="2.0", request_id=f"{day}:{b}:{vid}", variables=(vid,), scope=tp.scope,
                                         data_cut=z, window=TimeWindow(start=a, end=z), purpose="watch", request="get_aggregate"))
            o = resp.observations[0].payload if resp.status == "OK" else None
            vals[vid] = (int(o.numerator), int(o.denominator)) if o else (0, 0)
        tele_ms = (time.perf_counter() - t0) * 1000
        (perr, pn), (cerr, cn) = vals["payment.error_rate"], vals["checkout.error_rate"]
        orders_ok = cn - cerr
        blk = {"day": str(day), "block": b, "sim_start": iso(a), "sim_end": iso(z), "payment_requests": pn, "payment_errors": perr,
               "checkout_attempts": cn, "checkout_errors": cerr, "orders": orders_ok, "telemetry_spans": int(spans.height),
               "telemetry_ms": round(tele_ms, 1)}
        # 2. derived signals (declared nulls): checkout errors scaled by the error budget (checkout spans arrive complete
        #    and promptly; payment spans are batched by their service and can land a block late); attempts vs usual
        zp = (cerr / cn) / math.sqrt(P0 * (1 - P0) / cn) if cn else 0.0
        observations.append(obs_point("checkout_errors_vs_budget", f"chk:{day}:{b}", zp, z, {}))
        blk["checkout_z"] = zp
        blocks.append(blk)                                       # on the chart before the engine runs
        if day in NORMAL_DAYS:
            att_null.setdefault(b, 0.0)
            att_null[b] += cn / len(NORMAL_DAYS)
        else:
            mu = max(att_null.get(b, 0.0), 1.0)
            za = (cn - mu) / math.sqrt(mu)
            observations.append(obs_point("checkout_attempts_vs_usual", f"att:{day}:{b}", za, z, {}))
            blk["attempts_usual"], blk["attempts_z"] = mu, za
        # 3. core: the payment monitor after every block
        say(z, "agent", f"block {b + 1}/8 · checkouts {cn} · failed {cerr} · telemetry {tele_ms:.0f} ms")
        specs = {"checkout_errors_vs_budget": MonitorSpec(0.0, 1.0, family_size=2),
                 "checkout_attempts_vs_usual": MonitorSpec(0.0, 1.0, family_size=2)}
        r, tr = core(f"chk-{day}-{b}", {"question_kind": "DETECT_CHANGE", "variables": ["checkout_errors_vs_budget"],
                                        "baseline_policy": "declared null: no checkout errors; noise scaled by a 1% error budget"},
                     z + timedelta(minutes=3), specs, "checkout monitor")
        mon = tr["results"].get("savi_eprocess_v1", {})
        blk["monitor"] = mon
        blk["task"] = tr["task_id"]
        alarm = bool(mon.get("values", {}).get("alarm")) and mon.get("status") != "REFUSED"
        if alarm and not state["alarm"]:
            state["alarm"] = True
            say(z + timedelta(minutes=3), "engine", f"ALARM checkout errors {mon['values'].get('direction', '')} "
                f"({cerr}/{cn} checkouts failed this block) · task {tr['task_id']} in {tr['wall_ms']:.0f} ms")
            events.append({"sim": iso(z + timedelta(minutes=3)), "kind": "incident", "detail": "checkout errors above the error budget"})
            # diagnosis: is demand down too?
            r2, tr2 = core(f"att-{day}-{b}", {"question_kind": "DETECT_CHANGE", "variables": ["checkout_attempts_vs_usual"],
                                              "baseline_policy": "frozen null: usual attempts per block from the 4 previous days (Poisson)"},
                           z + timedelta(minutes=4), specs, "demand check")
            dm = tr2["results"].get("savi_eprocess_v1", {})
            demand_alarm = bool(dm.get("values", {}).get("alarm"))
            today = [x for x in blocks if x["day"] == str(day)]
            k = sum(x["checkout_errors"] for x in today); n = sum(x["checkout_attempts"] for x in today)
            d0 = datetime.combine(day, datetime.min.time(), U)       # where do checkouts fail? re-read the day's payment spans
            pr = tp.query(GetAggregate(schema_version="2.0", request_id=f"{day}:{b}:payday", variables=("payment.error_rate",),
                                       scope=tp.scope, data_cut=z, window=TimeWindow(start=d0, end=z), purpose="locate",
                                       request="get_aggregate"))
            po = pr.observations[0].payload if pr.status == "OK" else None
            pk, pnn = (int(po.numerator), int(po.denominator)) if po else (0, 0)
            att_today = sum(x["checkout_attempts"] for x in today); usual_today = sum(att_null.get(x["block"], 0) for x in today)
            lo, hi = wilson(k, n)
            rest = sum(att_null.get(j, 0) for j in range(b + 1, 24 // BLOCK_H))
            expected = {"metric": "orders recovered for the rest of the day by turning the payment fault off",
                        "mean": rest * k / max(n, 1), "lower": rest * lo, "upper": rest * hi,
                        "basis": "identity: orders = checkout attempts × (1 − payment failure share); usual attempts from the "
                                 "4 previous days; 90% Wilson range on the failure share. Arithmetic, not a fitted model."}
            diag = (f"checkout attempts {att_today} vs usual {usual_today:.0f} → demand "
                    f"{'DOWN' if demand_alarm else 'unchanged'}; checkouts failing {k}/{n} = {k / max(n, 1):.0%} "
                    f"[{lo:.0%}, {hi:.0%}]; payment service errors {pk} of {pnn} → the failures are in the payment path")
            say(z + timedelta(minutes=4), "engine", f"diagnosis: {diag}")
            events.append({"sim": iso(z + timedelta(minutes=4)), "kind": "diagnosis", "detail": diag})
            # policy: the core itself may not actuate
            r3, tr3 = core(f"act-{day}-{b}", {"question_kind": "DETECT_CHANGE", "variables": ["checkout_errors_vs_budget"],
                                              "baseline_policy": "declared null: no checkout errors; noise scaled by a 1% error budget"},
                           z + timedelta(minutes=5), specs, "action request", requested_action="turn the payment fault flag off")
            pol = tr3["policy"][-1] if tr3["policy"] else {"verdict": "?", "reason": ""}
            say(z + timedelta(minutes=5), "policy", f"{pol['verdict']}: {pol['reason'][:110]}")
            events.append({"sim": iso(z + timedelta(minutes=5)), "kind": "policy", "detail": f"{pol['verdict']}: {pol['reason']}"})
            p = ActionProposal.make("uc2", sandbox.name, "payment", "set_flag", {"flag": "paymentFailure", "variant": "off"},
                                    "checkouts fail in the payment service while demand is unchanged: the fault is in the payment path, not the market",
                                    expected, "low", [tr["task_id"], tr2["task_id"], tr3["task_id"]], iso(z + timedelta(minutes=6)))
            say(z + timedelta(minutes=6), "agent", f"PROPOSAL set_flag paymentFailure=off · expected +{expected['mean']:.0f} orders "
                f"today [{expected['lower']:.0f}, {expected['upper']:.0f}] · sandbox {sandbox.name}")
            events.append({"sim": iso(z + timedelta(minutes=6)), "kind": "proposal", "detail": "turn the payment fault flag off",
                           "expected": expected})
            state["pending"].append((z + timedelta(minutes=6) + APPROVAL_DELAY, p))
            L["proposal"] = {"proposal": asdict(p), "approval": None, "receipt": None, "diagnosis": diag, "policy": pol}
            L["alarm"] = True
            live.write(force=True, snapshot="proposal_card")
        live.write(force=True)
        if day in paced_days:
            time.sleep(pace_block)

    def make_callback(day: date):
        def callback(plans: list[dict]) -> dict:
            plans = sorted(plans, key=lambda p: p["purchased_at"])
            out = {}
            d0 = datetime.combine(day, datetime.min.time(), U)
            for b in range(24 // BLOCK_H):
                a, z = d0 + timedelta(hours=BLOCK_H * b), d0 + timedelta(hours=BLOCK_H * (b + 1))
                mine = [p for p in plans if a <= datetime.fromisoformat(p["purchased_at"]).replace(tzinfo=U) < z]
                w0 = datetime.now(U)
                for p in mine:
                    apply_due(datetime.fromisoformat(p["purchased_at"]).replace(tzinfo=U))
                    res = driver([p])
                    out.update(res)
                    r1 = res.get(p["command_id"], {})
                    L["sessions"] = (L["sessions"] + [{"sim": p["purchased_at"], "product": names.get(r1.get("product"), (r1.get("product"), None)),
                                                       "outcome": r1.get("outcome")}])[-14:]
                    L["sim"] = iso(datetime.fromisoformat(p["purchased_at"]).replace(tzinfo=U))
                    live.write()
                apply_due(z)
                w1 = datetime.now(U)
                watch(day, b, a, z, (w0, w1))
            return out
        return callback

    # ── the replay: 4 normal days, the fault day, a recovery day ──
    truth = {}
    for day in NORMAL_DAYS + [FAULT_DAY, RECOVERY_DAY]:
        d0 = datetime.combine(day, datetime.min.time())
        if day == FAULT_DAY:                                    # scenario side, never visible to the agent
            ev = emu.intervene(InterventionSpec(id="uc2-pay", target="storefront.payment_failure", scope="", operator="set",
                                                value=0.25, window_start=d0, window_end=d0 + timedelta(days=1), requested_by="scenario"))
            assert ev.status == "accepted", ev.reason
            fl = emu.operational_flags()[0]
            assert flags.set_variant(fl["flag"], fl["variant"]), "flag read-back failed"
            truth["injected"] = {"what": "storefront.payment_failure = 25% (flagd paymentFailure)", "from": iso(d0.replace(tzinfo=U)),
                                 "scheduled_until": iso((d0 + timedelta(days=1)).replace(tzinfo=U)), "flag": fl}
        if day == RECOVERY_DAY and flags.read("paymentFailure") != "off":
            flags.set_variant("paymentFailure", "off")             # end of the scenario window (if the agent never acted)
        cur["day"] = day
        L["paced"] = day in paced_days
        L["phase"] = "fault day" if day == FAULT_DAY else "recovery" if day == RECOVERY_DAY else "learning usual levels"
        L["flag"] = flags.read("paymentFailure") if day == FAULT_DAY else L["flag"]
        timeline.append({"wall": iso(datetime.now(U)), "sim": iso(d0.replace(tzinfo=U)), "kind": "day", "text": str(day),
                         "paced": day in paced_days})
        say(d0.replace(tzinfo=U), "world", f"day {day}: shoppers from the emulator go through the shop")
        emu.start_storefront(d0, d0 + timedelta(days=1))
        emu.step(day + timedelta(days=1), storefront=make_callback(day))
        live.write(force=True, snapshot=f"day_end_{day}")
    flags.set_variant("paymentFailure", "off")

    # ── truth (privileged plane: driver results), for the end-of-video reveal only ──
    cmd = pl.DataFrame(emu.commands).drop("latency_s", strict=False)
    cmd = cmd.with_columns(pl.col("purchased_at").str.slice(0, 10).alias("day"))
    truth["outcomes_by_day"] = cmd.group_by("day", "outcome").len().sort("day", "outcome").to_dicts()
    fixed = state["fixed_at"]
    if fixed is not None:
        fd = cmd.filter(pl.col("day") == str(FAULT_DAY))
        t = fd["purchased_at"].str.to_datetime(time_zone="UTC")
        truth["failed_before_fix"] = int(((t < fixed) & (fd["outcome"] != "ok")).sum())
        truth["failed_after_fix"] = int(((t >= fixed) & (fd["outcome"] != "ok")).sum())
        truth["fixed_at"] = iso(fixed)
        planned = int((t >= fixed).sum())
        truth["orders_saved_vs_no_fix_expected"] = round(planned * 0.25, 1)
        bad = fd.filter(fd["outcome"] != "ok").sort("purchased_at")
        truth["first_failure_sim"] = bad["purchased_at"][0] + "Z" if bad.height else None
        truth["checkouts_after_fix"] = planned
    (OUT / "blocks.json").write_text(json.dumps(blocks, indent=1, default=str))
    (OUT / "traces.json").write_text(json.dumps(traces, indent=1, default=str))
    (OUT / "events.json").write_text(json.dumps(events, indent=1, default=str))
    (OUT / "actions.json").write_text(json.dumps(actions + [{"refused_example": asdict(sandbox.execute(
        ActionProposal.make("uc2", "production-payments", "payment", "set_flag", {"flag": "paymentFailure", "variant": "off"},
                            "outside the sandbox", None, "high", [], iso(datetime.combine(RECOVERY_DAY, datetime.min.time(), U))),
        None, datetime.combine(RECOVERY_DAY, datetime.min.time(), U)))}], indent=1, default=str))
    (OUT / "console.json").write_text(json.dumps(say.lines, indent=1))
    timeline.append({"wall": iso(datetime.now(U)), "sim": None, "kind": "end", "text": "run complete"})
    (OUT / "timeline.json").write_text(json.dumps(timeline, indent=1))
    L["phase"] = "done"
    live.write(force=True, snapshot="done")
    (OUT / "world.json").write_text(json.dumps(world_meta | {"normal_days": [str(d) for d in NORMAL_DAYS], "fault_day": str(FAULT_DAY),
                                                             "recovery_day": str(RECOVERY_DAY), "block_hours": BLOCK_H,
                                                             "error_budget": P0, "attempts_null": att_null}, indent=1))
    (OUT / "truth.json").write_text(json.dumps(truth, indent=1, default=str))
    tasks_ms = [t["wall_ms"] for t in traces]
    print(f"\nwrote {OUT}: {len(blocks)} blocks, {len(traces)} core tasks (median {np.median(tasks_ms):.0f} ms), "
          f"{len(actions)} sandbox actions; truth {json.dumps({k: v for k, v in truth.items() if k != 'outcomes_by_day'}, default=str)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pace", type=float, default=0.0, help="seconds to pause after each 3-hour block on paced days")
    ap.add_argument("--event-pause", type=float, default=0.0, help="seconds to pause after each key event on paced days")
    ap.add_argument("--paced-days", default="", help="comma-separated dates shown at watchable speed, e.g. 2018-02-15,2018-02-16")
    a = ap.parse_args()
    days = tuple(date.fromisoformat(x) for x in a.paced_days.split(",") if x)
    sys.exit(run(a.pace, a.event_pause, days))
