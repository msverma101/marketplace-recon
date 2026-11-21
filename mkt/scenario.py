"""Scenario runner: config -> run directory with a strict observable / truth split (docs §12.4).

runs/<scenario>-s<seed>/
    observable/events.parquet      what the monitor may see (replay reads only this)
    observable/manifest.json       schema version, time range, streams (no truth)
    truth/                          evaluator-only: interventions, ground truth, effects, latents
    manifest.json                  reproducibility key

Simulated scenarios run a factual world and a counterfactual twin (same seed, no world
interventions); effects are factual minus twin, so they are computed, not asserted.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tomllib
from datetime import datetime

import polars as pl

from . import SCHEMA_VERSION, observe, pipeline
from .config import ROOT, Config
from .contracts import Intervention
from .sim import engine
from .sim.world import build_world
from .store import code_hash, stable_hash

SCENARIO_DIR = ROOT / "scenarios"


def load(name_or_path: str) -> dict:
    p = pathlib.Path(name_or_path)
    if not p.exists():
        hits = sorted(SCENARIO_DIR.glob(f"{name_or_path}*.toml"))
        if not hits:
            raise FileNotFoundError(f"no scenario '{name_or_path}' in {SCENARIO_DIR}")
        p = hits[0]
    sc = tomllib.loads(p.read_text())
    sc["_path"] = str(p)
    return sc


def run_dir(cfg: Config, sc: dict, seed: int) -> pathlib.Path:
    return cfg.runs / f"{sc['id']}-s{seed}"


def run(cfg: Config, name_or_path: str, seed: int | None = None) -> pathlib.Path:
    sc = load(name_or_path)
    w = sc.get("world", {})
    seed = int(seed if seed is not None else w.get("seed", 0))
    ivs = [Intervention.from_dict(d) for d in sc.get("interventions", [])]
    out = run_dir(cfg, sc, seed)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "observable").mkdir(parents=True)
    (tmp / "truth").mkdir()
    world_key = stable_hash({"scenario": sc["id"], "seed": seed, "salt": "world-key"})

    if sc.get("kind", "simulated") == "simulated":
        P, pkey = pipeline.params(cfg)
        world = build_world(P, seed=seed, scale=float(w.get("scale", 0.5)))
        start, days = datetime.fromisoformat(w["start"]), int(w["days"])
        fact = engine.simulate(world, start, days, ivs, world_tag=f"w{seed}")
        twin = engine.simulate(world, start, days, [], world_tag=f"w{seed}")
        world_events = fact.events
        obs = observe.observe(world_events, seed, ivs, world_key=world_key)
        _series(fact.events).join(_series(twin.events), on=["hour", "metric", "dim", "dim_value"], how="full",
                                  suffix="_twin", coalesce=True).fill_null(0).rename({"n": "factual", "n_twin": "twin"}) \
            .write_parquet(tmp / "truth" / "effects_hourly.parquet")
        fact.orders.write_parquet(tmp / "truth" / "orders_latent.parquet")
        pl.DataFrame({"seller_id": world.seller_id(range(len(world.seller_weight))),
                      "handling_mu_log_hours": world.seller_handling_mu, "region": world.seller_region}) \
            .write_parquet(tmp / "truth" / "sellers_latent.parquet")
        world_events.write_parquet(tmp / "truth" / "world_events.parquet")
        stats = {"factual": fact.stats, "twin": twin.stats, "params_key": pkey, "aliases": world.aliases}
    else:  # real-only replay track
        canon = pipeline.canonical_olist(cfg)
        ev = pl.read_parquet(canon / "events.parquet")
        lo, hi = datetime.fromisoformat(w["start"]), datetime.fromisoformat(w["end"])
        world_events = ev.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))
        obs = observe.observe(world_events, seed, ivs, world_key=world_key)
        stats = {"events": world_events.height, "source": canon.name}

    obs.write_parquet(tmp / "observable" / "events.parquet")
    (tmp / "observable" / "manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "scenario": sc["id"], "events": obs.height,
        "available_from": str(obs["available_at"].min()), "available_to": str(obs["available_at"].max()),
        "streams": sorted(obs["stream"].unique().to_list()),
        "announcements": "event_type == 'announcement' rows are public release notes"}, indent=1))
    (tmp / "truth" / "interventions.json").write_text(json.dumps([i.to_dict() for i in ivs], indent=1))
    (tmp / "truth" / "ground_truth.json").write_text(json.dumps(sc.get("truth", []), indent=1, default=str))
    (tmp / "truth" / "limitations.txt").write_text(sc.get("limitations", "Ground truth is simulator-relative; "
                                                         "it is not a claim about Olist or any real marketplace.\n"))
    (tmp / "manifest.json").write_text(json.dumps({
        "scenario": sc["id"], "version": sc.get("version", 1), "seed": seed, "kind": sc.get("kind", "simulated"),
        "code": code_hash(engine, observe), "scenario_hash": stable_hash({k: v for k, v in sc.items() if k != "_path"}),
        "stats": stats}, indent=1, default=str))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


SERIES_TYPES = ["session_start", "checkout_start", "checkout_error", "order_created", "order_unavailable",
                "delivered", "review_created", "payment_settled"]


def _series(ev: pl.DataFrame) -> pl.DataFrame:
    """Hourly counts (and late deliveries, score sums) by segment, for factual-vs-twin effects."""
    e = ev.filter(pl.col("event_type").is_in(SERIES_TYPES)).with_columns(
        pl.col("event_time").dt.truncate("1h").alias("hour"),
        pl.when(pl.col("event_type") == "delivered").then(pl.col("value") > 0).alias("late"))
    parts = []
    for dim in ["all", "device", "category", "region", "seller_id"]:
        dv = pl.lit("all") if dim == "all" else pl.col(dim).fill_null("")
        g = e.with_columns(dv.alias("dim_value"))
        parts.append(g.group_by("hour", "event_type", "dim_value").agg(pl.len().alias("n"))
                     .with_columns(pl.lit(dim).alias("dim")).rename({"event_type": "metric"}))
        parts.append(g.filter(pl.col("late") == True).group_by("hour", "dim_value").agg(pl.len().alias("n"))  # noqa: E712
                     .with_columns(pl.lit(dim).alias("dim"), pl.lit("late_delivered").alias("metric")))
        parts.append(g.filter(pl.col("event_type") == "order_created").group_by("hour", "dim_value")
                     .agg(pl.col("amount").sum().alias("n")).with_columns(pl.lit(dim).alias("dim"), pl.lit("gmv").alias("metric")))
    return pl.concat([p.select("hour", "metric", "dim", "dim_value", pl.col("n").cast(pl.Float64)) for p in parts])
