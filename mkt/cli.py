"""mkt command-line interface (docs §11)."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timedelta

from .config import Config


def _p(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


def cmd_source(a, cfg):
    from . import registry
    if a.action == "add":
        m = registry.register(a.ns, cfg.data, cfg.artifacts, pathlib.Path(a.path) if a.path else None)
        _p({"registered": m.ns, "version": m.version, "licence": m.licence, "path": m.path})
    else:
        _p([{"ns": m.ns, "version": m.version, "licence": m.licence, "path": m.path} for m in registry.list_sources(cfg.artifacts)])


def cmd_semantics(a, cfg):
    from . import registry, semantics
    src = pathlib.Path(registry.load("olist", cfg.artifacts).path)
    res = semantics.audit(src)
    out = pathlib.Path("docs/p0"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase1_semantics.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "PHASE1_SOURCE_SEMANTICS.md").write_text(semantics.report_md(res))
    _p({"gates": res["gates"], "report": str(out / "PHASE1_SOURCE_SEMANTICS.md")})
    return 0 if all(v == "PASS" for v in res["gates"].values()) else 1


def cmd_ingest(a, cfg):
    from . import pipeline
    p = pipeline.canonical_olist(cfg, force=a.force)
    _p({"canonical": str(p), **json.loads((p / "stats.json").read_text())})


def cmd_profile(a, cfg):
    import polars as pl
    from . import pipeline
    p = pipeline.canonical_olist(cfg)
    rows = []
    for f in sorted(p.glob("*.parquet")):
        df = pl.read_parquet(f)
        for c in df.columns:
            s = df[c]
            rows.append({"table": f.stem, "column": c, "dtype": str(s.dtype), "rows": df.height,
                         "null_pct": round(100 * s.null_count() / max(df.height, 1), 2),
                         "distinct": s.n_unique(), "min": str(s.min()) if s.dtype.is_numeric() or s.dtype.is_temporal() else "",
                         "max": str(s.max()) if s.dtype.is_numeric() or s.dtype.is_temporal() else ""})
    with pl.Config(tbl_rows=200, tbl_cols=10, fmt_str_lengths=30):
        print(pl.DataFrame(rows))


def cmd_fit(a, cfg):
    from . import pipeline
    P, key = pipeline.params(cfg, force=a.force)
    _p({"params_key": key, "provenance": P["_provenance"], "daily_orders": P["daily_orders"],
        "holdout": pipeline.holdout(cfg)})


def cmd_run(a, cfg):
    from . import scenario
    t = time.time()
    out = scenario.run(cfg, a.scenario, a.seed)
    man = json.loads((out / "manifest.json").read_text())
    _p({"run": str(out), "seconds": round(time.time() - t, 1), "stats": man["stats"]})


def cmd_replay(a, cfg):
    from .replay import Replayer
    r = Replayer(a.run)
    n_w = n_e = 0
    sink = open(a.sink, "w") if a.sink else None
    for clock, f in r.windows(timedelta(minutes=a.window_min)):
        n_w += 1
        n_e += f.height
        if sink and f.height:
            sink.write(f.with_columns([f[c].cast(str) for c in ("event_time", "available_at")]).write_ndjson())
        if a.max_windows and n_w >= a.max_windows:
            break
    if sink:
        sink.close()
    import numpy as np
    t = np.array(r.timings) * 1000
    _p({"windows": n_w, "events": n_e, "window_ms_p50": round(float(np.percentile(t, 50)), 3),
        "window_ms_p99": round(float(np.percentile(t, 99)), 3)})


def cmd_monitor(a, cfg):
    from . import monitor
    from .replay import Replayer
    alerts = monitor.run(Replayer(a.run))
    out = pathlib.Path(a.out or pathlib.Path(a.run) / "monitor_baseline_alerts.jsonl")
    out.write_text("".join(json.dumps(x) + "\n" for x in alerts))
    _p({"alerts": len(alerts), "file": str(out)})


def cmd_evaluate(a, cfg):
    from . import evaluate
    path = pathlib.Path(a.alerts or pathlib.Path(a.run) / "monitor_baseline_alerts.jsonl")
    alerts = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rep = evaluate.evaluate(pathlib.Path(a.run), alerts)
    (pathlib.Path(a.run) / f"evaluation_{path.stem}.json").write_text(json.dumps(rep, indent=1, default=str))
    _p(rep)


def cmd_validate(a, cfg):
    from . import validate
    res = validate.check_run(pathlib.Path(a.run))
    _p({k: {"ok": ok, "detail": d} for k, (ok, d) in res.items()})
    return 0 if all(ok for ok, _ in res.values()) else 1


def cmd_bench(a, cfg):
    from . import observe, pipeline
    from .replay import Replayer
    from .sim import engine
    from .sim.world import build_world
    import shutil, tempfile
    P, _ = pipeline.params(cfg)
    w = build_world(P, seed=a.seed, scale=a.scale)
    days = max(1, int(round(a.events / (150_000 * a.scale))))      # ~150k events/day at scale 1
    t = time.time(); r = engine.simulate(w, datetime(2018, 3, 1), days); t_sim = time.time() - t
    t = time.time(); o = observe.observe(r.events, a.seed); t_obs = time.time() - t
    d = pathlib.Path(tempfile.mkdtemp()); (d / "observable").mkdir()
    o.write_parquet(d / "observable" / "events.parquet")
    t = time.time(); rp = Replayer(d); n = sum(1 for _ in rp.windows(timedelta(minutes=15))); t_rep = time.time() - t
    import numpy as np
    ms = np.array(rp.timings) * 1000
    shutil.rmtree(d)
    _p({"days": days, "scale": a.scale, "world_events": r.events.height, "simulate_s": round(t_sim, 2),
        "events_per_s": int(r.events.height / t_sim), "observe_s": round(t_obs, 2),
        "replay_windows_15min": n, "replay_total_s": round(t_rep, 2),
        "window_ms_p50": round(float(np.percentile(ms, 50)), 3), "window_ms_p99": round(float(np.percentile(ms, 99)), 3)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mkt", description=__doc__)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("source", help="register or list raw sources")
    s.add_argument("action", choices=["add", "list"]); s.add_argument("ns", nargs="?"); s.add_argument("--path")
    s.set_defaults(fn=cmd_source)
    s = sub.add_parser("semantics", help="phase-1 Olist field dictionary + gates S1/S2"); s.set_defaults(fn=cmd_semantics)
    s = sub.add_parser("ingest", help="build canonical Olist tables + events"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("profile", help="profile canonical tables"); s.set_defaults(fn=cmd_profile)
    s = sub.add_parser("fit", help="calibrate mechanisms (with held-out stats)"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_fit)
    s = sub.add_parser("run", help="run a scenario"); s.add_argument("scenario"); s.add_argument("--seed", type=int); s.set_defaults(fn=cmd_run)
    s = sub.add_parser("replay", help="replay a run's observable events"); s.add_argument("run")
    s.add_argument("--window-min", type=int, default=60); s.add_argument("--sink"); s.add_argument("--max-windows", type=int)
    s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("monitor", help="run the baseline monitor over a replay"); s.add_argument("run"); s.add_argument("--out"); s.set_defaults(fn=cmd_monitor)
    s = sub.add_parser("evaluate", help="score alerts against hidden truth"); s.add_argument("run"); s.add_argument("--alerts"); s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("validate", help="integrity and leakage checks on a run"); s.add_argument("run"); s.set_defaults(fn=cmd_validate)
    s = sub.add_parser("bench", help="generation + replay benchmark"); s.add_argument("--events", type=int, default=1_000_000)
    s.add_argument("--scale", type=float, default=1.0); s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=cmd_bench)
    a = ap.parse_args(argv)
    cfg = Config.load(a.config)
    if a.cmd == "source" and a.action == "add" and not a.ns:
        ap.error("source add needs a namespace")
    rc = a.fn(a, cfg)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
