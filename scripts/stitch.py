#!/usr/bin/env python3
"""Data stitching stage: fill every data type Olist lacks from the other datasets.

Design: docs/ARCHITECTURE.md §7.5, diagram docs/stitching.excalidraw.

    python scripts/stitch.py plan                       # what would run, which sources exist; touches nothing
    python scripts/stitch.py run                        # all steps whose sources are local
    python scripts/stitch.py run --allow-download       # also steps that read remote sources (ESCI, Open Bandit, Criteo, BCB)
    python scripts/stitch.py run --steps sessions,funnel,anchor

Outputs go to <out>/<step>/ plus <out>/stitch_manifest.json (method, provenance label, inputs with
sha256 or URL, outputs with row counts, assumptions, key numbers, seconds per step).

Methods (never an identity claim across datasets, §3.2):
  bridge    a synthetic entity is later dressed with a real record from the pool built here
  transfer  a relative shape (curve, ratio, distribution) learned elsewhere; levels come from Olist
  pool      real values sampled by condition (e.g. review text by score)
  join      exact join on a real shared key (seller_id, date + state)
The last step, `anchor`, solves the free base rates so the stitched chain reproduces Olist volume.

Remote reads stream only the needed columns where the format allows (Parquet); CSV sources are read once.
Requires: duckdb, polars (both in the project venv). Optional: openpyxl (UCI .xlsx).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

import duckdb
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATA = pathlib.Path("~/.cache/mkt-probe/data").expanduser()
DEFAULT_OUT = ROOT / "artifacts" / "stitch"

HF_ESCI = "hf://datasets/tasksource/esci/data/*.parquet"
HF_OBD_RANDOM = "hf://datasets/zozonext/open-bandit/random/all/all.csv"
HF_OBD_BTS = "hf://datasets/zozonext/open-bandit/bts/all/all.csv"
HF_CRITEO = "hf://datasets/criteo/criteo-uplift/criteo-research-uplift-v2.1.csv.gz"
HF_ASIN_MAP = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/asin2category.json"
BCB = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial=01/01/2016&dataFinal=31/12/2018"
BCB_SERIES = {"ipca_monthly_pct": 433, "selic_target_pct": 432, "usd_brl": 1}

# Olist category -> Amazon Reviews 2023 main category. Hand-made: assumption A-13.
# Categories not listed get no bridge (their products keep generic attributes).
CATEGORY_MAP = {
    "bed_bath_table": "Home and Kitchen", "furniture_decor": "Home and Kitchen", "housewares": "Home and Kitchen",
    "kitchen_dining_laundry_garden_furniture": "Home and Kitchen", "home_confort": "Home and Kitchen",
    "home_comfort_2": "Home and Kitchen", "furniture_living_room": "Home and Kitchen", "furniture_bedroom": "Home and Kitchen",
    "health_beauty": "Beauty and Personal Care", "perfumery": "Beauty and Personal Care",
    "sports_leisure": "Sports and Outdoors", "computers_accessories": "Electronics", "electronics": "Electronics",
    "computers": "Electronics", "audio": "Electronics", "tablets_printing_image": "Electronics",
    "telephony": "Cell Phones and Accessories", "fixed_telephony": "Cell Phones and Accessories",
    "watches_gifts": "Clothing Shoes and Jewelry", "fashion_bags_accessories": "Clothing Shoes and Jewelry",
    "luggage_accessories": "Clothing Shoes and Jewelry", "fashion_shoes": "Clothing Shoes and Jewelry",
    "fashion_male_clothing": "Clothing Shoes and Jewelry", "fashion_underwear_beach": "Clothing Shoes and Jewelry",
    "garden_tools": "Patio Lawn and Garden", "auto": "Automotive", "toys": "Toys and Games", "cool_stuff": "Toys and Games",
    "baby": "Baby Products", "stationery": "Office Products", "office_furniture": "Office Products",
    "pet_shop": "Pet Supplies", "consoles_games": "Video Games", "musical_instruments": "Musical Instruments",
    "construction_tools_construction": "Tools and Home Improvement", "home_construction": "Tools and Home Improvement",
    "construction_tools_lights": "Tools and Home Improvement", "construction_tools_safety": "Tools and Home Improvement",
    "costruction_tools_garden": "Tools and Home Improvement", "costruction_tools_tools": "Tools and Home Improvement",
    "home_appliances": "Appliances", "home_appliances_2": "Appliances", "small_appliances": "Appliances",
    "small_appliances_home_oven_and_coffee": "Appliances", "air_conditioning": "Appliances",
    "books_general_interest": "Books", "books_technical": "Books", "books_imported": "Books",
    "food": "Grocery and Gourmet Food", "drinks": "Grocery and Gourmet Food", "food_drink": "Grocery and Gourmet Food",
    "industry_commerce_and_business": "Industrial and Scientific", "agro_industry_and_commerce": "Industrial and Scientific",
    "art": "Arts Crafts and Sewing", "arts_and_craftmanship": "Arts Crafts and Sewing", "party_supplies": "Arts Crafts and Sewing",
}
POOL_PER_CATEGORY = 20_000          # bridged items kept per Amazon category (seeded sample)
SEED = 20261121


# ----------------------------------------------------------------------------- plumbing
@dataclass
class Source:
    name: str
    where: str                      # local relative path under --data, or a remote URL / hf:// path
    remote: bool = False
    approx: str = ""                # human size hint
    optional: bool = False


@dataclass
class Step:
    name: str
    method: str
    prov: str
    sources: list[Source]
    fn: Callable
    assumptions: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)    # other steps whose outputs are used
    desc: str = ""


class Ctx:
    def __init__(self, data: pathlib.Path, out: pathlib.Path, allow_download: bool, with_bts: bool):
        self.data, self.out, self.allow_download, self.with_bts = data, out, allow_download, with_bts
        self.con = duckdb.connect()
        self.cache = out / "_cache"
        self.cache.mkdir(parents=True, exist_ok=True)

    def local(self, rel: str) -> pathlib.Path:
        return self.data / rel

    def one(self, sql: str):
        return self.con.execute(sql).fetchone()


class Skip(Exception):
    pass


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def need_local(ctx: Ctx, rel: str) -> pathlib.Path:
    p = ctx.local(rel)
    if not p.exists():
        raise Skip(f"missing local source {p}")
    return p


def need_remote(ctx: Ctx, what: str):
    if not ctx.allow_download:
        raise Skip(f"remote source {what} needs --allow-download")


def write(df: pl.DataFrame, path: pathlib.Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return {"file": str(path.relative_to(path.parents[1])), "rows": df.height}


def quantiles(con: Ctx, sql: str, qs=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)) -> dict:
    vals = con.one(f"select quantile_cont(x, {list(qs)}) from ({sql}) where x is not null")[0]
    return {f"q{int(q * 100):02d}": float(v) for q, v in zip(qs, vals)} if vals else {}


# ----------------------------------------------------------------------------- steps
def step_catalog(ctx: Ctx) -> dict:
    """Bridge pool: real Amazon items (title, brand, category) for products in each mapped category."""
    tsv = ctx.cache / "rev_ids.tsv"
    if not tsv.exists():
        src = ctx.local("rev_ids.tsv")
        if src.exists():
            shutil.copy(src, tsv)
        else:
            js = ctx.local("asin2category.json")
            if not js.exists():
                need_remote(ctx, HF_ASIN_MAP)
                js = ctx.cache / "asin2category.json"
                urllib.request.urlretrieve(HF_ASIN_MAP, js)
            import mmap
            import re
            pat = re.compile(rb'"([^"]+)": "([^"]*)"')
            with open(js, "rb") as f, open(tsv, "wb") as o, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for m in pat.finditer(mm):
                    o.write(m.group(1) + b"\t" + m.group(2) + b"\n")
    esci = ctx.cache / "esci_us.parquet"
    if not esci.exists():
        need_remote(ctx, HF_ESCI)
        ctx.con.execute(f"""copy (select query_id, query, product_id, esci_label, product_title, product_brand
                         from '{HF_ESCI}' where product_locale = 'us') to '{esci}' (format parquet)""")
    ctx.con.execute(f"""create or replace view amz as select * from read_csv('{tsv}', delim='\t', header=false,
                    columns={{'asin':'VARCHAR','amazon_category':'VARCHAR'}}, quote='')""")
    cats = sorted(set(CATEGORY_MAP.values()))
    cat_list = ", ".join("'" + c.replace("'", "''") + "'" for c in cats)
    pool = ctx.con.execute(f"""
        with items as (select distinct product_id asin, first(product_title) title, first(product_brand) brand
                       from '{esci}' group by 1)
        select i.asin, a.amazon_category, i.title, i.brand from items i join amz a using(asin)
        where a.amazon_category in ({cat_list}) and i.title is not null
        qualify row_number() over (partition by a.amazon_category order by hash(i.asin || '{SEED}')) <= {POOL_PER_CATEGORY}
    """).pl()
    out = [write(pool, ctx.out / "catalog" / "bridge_pool.parquet"),
           write(pl.DataFrame({"olist_category": list(CATEGORY_MAP), "amazon_category": list(CATEGORY_MAP.values())}),
                 ctx.out / "catalog" / "category_map.parquet")]
    summary = {"pool_items": pool.height, "categories": pool["amazon_category"].n_unique()}
    items = ctx.local("olist/olist_order_items_dataset.csv")
    prods = ctx.local("olist/olist_products_dataset.csv")
    trans = ctx.local("olist/product_category_name_translation.csv")
    if items.exists() and prods.exists() and trans.exists():
        mapped = list(CATEGORY_MAP)
        cov = ctx.one(f"""select avg((coalesce(t.product_category_name_english, '') in ({", ".join(repr(m) for m in mapped)}))::int)
                          from read_csv_auto('{items}') i join read_csv_auto('{prods}') p using(product_id)
                          left join read_csv_auto('{trans}') t using(product_category_name)""")[0]
        summary["olist_item_volume_mapped"] = round(float(cov), 4)
    return {"outputs": out, "summary": summary}


def step_queries(ctx: Ctx) -> dict:
    """Real queries and human relevance labels for the bridged items."""
    esci = ctx.cache / "esci_us.parquet"
    pool = ctx.out / "catalog" / "bridge_pool.parquet"
    if not esci.exists() or not pool.exists():
        raise Skip("needs the catalog step first")
    q = ctx.con.execute(f"""select e.product_id asin, e.query_id, e.query, e.esci_label
                         from '{esci}' e semi join '{pool}' p on e.product_id = p.asin""").pl()
    mix = q.group_by("esci_label").len().with_columns((pl.col("len") / pl.col("len").sum()).alias("share"))
    return {"outputs": [write(q, ctx.out / "queries" / "queries.parquet")],
            "summary": {"rows": q.height, "queries": q["query_id"].n_unique(),
                        "label_mix": dict(zip(mix["esci_label"], [round(x, 4) for x in mix["share"]]))}}


def _obd(ctx: Ctx, path: str) -> dict:
    cols = [r[0] for r in ctx.con.execute(f"describe select * from read_csv_auto('{path}') limit 1").fetchall()]
    for c in ("position", "click", "propensity_score"):
        if c not in cols:
            raise RuntimeError(f"Open Bandit file lacks column {c!r}; has {cols[:12]}")
    rows = ctx.con.execute(f"""select position, count(*) n, avg(click) ctr, avg(propensity_score) mean_propensity
                            from read_csv_auto('{path}') group by 1 order by 1""").fetchall()
    base = rows[0][2] or 1e-9
    return {"by_position": [{"position": int(p), "n": int(n), "ctr": float(c), "relative_to_top": float(c) / base,
                             "mean_propensity": float(mp)} for p, n, c, mp in rows]}


def step_positions(ctx: Ctx) -> dict:
    """Position effect from ZOZOTOWN's uniformly random policy: unbiased relative click curve."""
    need_remote(ctx, HF_OBD_RANDOM)
    res = {"random": _obd(ctx, HF_OBD_RANDOM)}
    if ctx.with_bts:
        res["bts"] = _obd(ctx, HF_OBD_BTS)
    p = ctx.out / "positions" / "position_curve.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return {"outputs": [{"file": "positions/position_curve.json"}], "summary": res["random"]}


def step_sessions(ctx: Ctx) -> dict:
    """Session length and adjacent-revisit structure from Amazon-M2 (official files == Kaggle mirror)."""
    src = need_local(ctx, "m2/sessions_train.csv")
    ctx.con.execute(f"""create or replace view m2 as select locale,
        regexp_extract_all(prev_items, '[A-Z0-9]{{10}}') items from read_csv('{src}', header=true, parallel=false)""")
    ctx.con.execute("""create or replace view m2l as select locale, len(items) + 1 n,
        (select count(*) from range(1, len(items)) r(i) where items[i] = items[i + 1]) adj_repeats,
        len(list_distinct(items)) distinct_prev from m2""")
    hist = ctx.con.execute("select least(n, 50) len, count(*) c from m2l group by 1 order by 1").pl()
    hist = hist.with_columns((pl.col("c") / pl.col("c").sum()).alias("p"))
    by_loc = ctx.con.execute("""select locale, avg(n) mean_len, median(n) median_len,
        sum(adj_repeats) / sum(n - 2) adj_repeat_rate, avg(distinct_prev / (n - 1)) distinct_share from m2l group by 1""").pl()
    return {"outputs": [write(hist, ctx.out / "sessions" / "length_hist.parquet"),
                        write(by_loc, ctx.out / "sessions" / "by_locale.parquet")],
            "summary": {"sessions": int(hist["c"].sum()), "mean_len": float(by_loc["mean_len"].mean()),
                        "note": "M2 has no event types or timestamps: lengths count viewed items only"}}


def step_funnel(ctx: Ctx) -> dict:
    """Funnel probabilities and inter-step timing from Retailrocket (session ≈ visitor-day, A-02)."""
    src = need_local(ctx, "retailrocket/events.csv")
    ctx.con.execute(f"""create or replace view rr as select visitorid v, itemid i, event e,
        to_timestamp(timestamp / 1000) t from read_csv_auto('{src}')""")
    ctx.con.execute("""create or replace view vd as select v, cast(t as date) d, sum((e='view')::int) views,
        sum((e='addtocart')::int) carts, sum((e='transaction')::int) txns from rr group by 1, 2""")
    probs = dict(zip(["sessions", "views_mean", "p_cart", "p_buy_given_cart", "p_buy_direct", "p_buy"], map(float, ctx.one("""
        select count(*), avg(views), avg((carts>0)::int),
               sum(((carts>0) and (txns>0))::int) / sum((carts>0)::int),
               sum(((carts=0) and (txns>0))::int) / sum((carts=0)::int), avg((txns>0)::int) from vd"""))))
    ctx.con.execute("""create or replace view firsts as select v, i,
        min(t) filter (where e='view') tv, min(t) filter (where e='addtocart') tc, min(t) filter (where e='transaction') tt
        from rr group by 1, 2""")
    gaps = {
        "view_to_cart_min": quantiles(ctx, "select date_diff('second', tv, tc) / 60.0 x from firsts where tc >= tv"),
        "cart_to_buy_min": quantiles(ctx, "select date_diff('second', tc, tt) / 60.0 x from firsts where tt >= tc"),
    }
    no_view = float(ctx.one("select avg((tv is null or tv > tt)::int) from firsts where tt is not null")[0])
    res = {**probs, "gaps": gaps, "purchases_without_prior_view": no_view}
    p = ctx.out / "funnel" / "funnel.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return {"outputs": [{"file": "funnel/funnel.json"}], "summary": res}


def step_stock(ctx: Ctx) -> dict:
    """Stock-out spells from Retailrocket's time-varying 'available' flag (snapshots, not exact change times)."""
    p1 = need_local(ctx, "retailrocket/item_properties_part1.csv")
    p2 = need_local(ctx, "retailrocket/item_properties_part2.csv")
    ctx.con.execute(f"""create or replace view av as select itemid, to_timestamp(timestamp / 1000) t, value = '1' available
        from read_csv_auto(['{p1}', '{p2}']) where property = 'available'""")
    ctx.con.execute("""create or replace view spells as with s as (
            select *, lag(available) over w prev, lead(t) over w next_t from av window w as (partition by itemid order by t))
        select itemid, t start_t, next_t,
               date_diff('hour', t, next_t) / 24.0 days from s where not available and (prev is null or prev) and next_t is not null""")
    share = float(ctx.one("select avg((not available)::int) from av")[0])
    spell = quantiles(ctx, "select days x from spells")
    n = int(ctx.one("select count(*) from spells")[0])
    res = {"snapshot_share_unavailable": share, "outage_spell_days": spell, "spells": n,
           "caveat": "spell ends at the next snapshot, so durations are upper bounds at weekly snapshot resolution"}
    p = ctx.out / "stock" / "stock.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return {"outputs": [{"file": "stock/stock.json"}], "summary": res}


def step_ads(ctx: Ctx) -> dict:
    """Randomised ad treatment with non-compliance (treatment != exposure) from Criteo."""
    need_remote(ctx, HF_CRITEO)
    cols = [r[0] for r in ctx.con.execute(f"describe select * from read_csv_auto('{HF_CRITEO}') limit 1").fetchall()]
    for c in ("treatment", "conversion", "visit", "exposure"):
        if c not in cols:
            raise RuntimeError(f"Criteo file lacks {c!r}; has {cols}")
    rows = {int(t): dict(zip(["n", "visit", "conversion", "exposure"], map(float, r))) for t, *r in ctx.con.execute(
        f"select treatment, count(*), avg(visit), avg(conversion), avg(exposure) from read_csv_auto('{HF_CRITEO}') group by 1").fetchall()}
    t, c = rows[1], rows[0]
    res = {"by_arm": rows, "treated_share": t["n"] / (t["n"] + c["n"]),
           "visit_lift_ratio": t["visit"] / c["visit"], "conversion_lift_ratio": t["conversion"] / c["conversion"],
           "exposure_rate_treated": t["exposure"],
           "cace_conversion": (t["conversion"] - c["conversion"]) / max(t["exposure"], 1e-9)}
    p = ctx.out / "ads" / "ads.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return {"outputs": [{"file": "ads/ads.json"}], "summary": res}


def step_review_text(ctx: Ctx) -> dict:
    """Pool of real (anonymised, public) Olist review texts by score."""
    src = need_local(ctx, "olist/olist_order_reviews_dataset.csv")
    pool = ctx.con.execute(f"""select distinct review_score score, review_comment_title title, review_comment_message text
        from read_csv_auto('{src}') where review_comment_message is not null""").pl()
    by = pool.group_by("score").len().sort("score")
    return {"outputs": [write(pool, ctx.out / "review_text" / "pool.parquet")],
            "summary": {"texts": pool.height, "by_score": dict(zip(by["score"], by["len"])), "language": "pt-BR"}}


def step_returns(ctx: Ctx) -> dict:
    """Cancellation/unavailability rates from Olist (real); UCI return share as an optional prior."""
    src = need_local(ctx, "olist/olist_orders_dataset.csv")
    st = dict(ctx.con.execute(f"""select order_status, count(*) / (select count(*) from read_csv_auto('{src}'))
                                  from read_csv_auto('{src}') group by 1""").fetchall())
    res = {"olist_status_share": {k: float(v) for k, v in st.items()}}
    uci = ctx.local("uci/online_retail_II.xlsx")
    if uci.exists():
        try:
            df = pl.concat([pl.read_excel(uci, sheet_name=s) for s in ("Year 2009-2010", "Year 2010-2011")])
            inv = df["Invoice"].cast(pl.Utf8)
            res["uci_cancelled_line_share"] = float(inv.str.starts_with("C").mean())
        except Exception as e:  # noqa: BLE001
            res["uci_error"] = f"{type(e).__name__}: {e} (needs fastexcel/openpyxl)"
    else:
        res["uci"] = "not present: download https://archive.ics.uci.edu/dataset/502/online+retail+ii into <data>/uci/"
    p = ctx.out / "returns" / "returns.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return {"outputs": [{"file": "returns/returns.json"}], "summary": res}


def step_sellers(ctx: Ctx) -> dict:
    """Seller acquisition -> first sale, exact join on seller_id (Olist Marketing Funnel)."""
    deals = need_local(ctx, "olist/funnel/olist_closed_deals_dataset.csv")
    mql = need_local(ctx, "olist/funnel/olist_marketing_qualified_leads_dataset.csv")
    items = need_local(ctx, "olist/olist_order_items_dataset.csv")
    orders = need_local(ctx, "olist/olist_orders_dataset.csv")
    df = ctx.con.execute(f"""
        with first_sale as (select i.seller_id, min(o.order_purchase_timestamp) first_sale
                            from read_csv_auto('{items}') i join read_csv_auto('{orders}') o using(order_id) group by 1)
        select d.seller_id, m.origin, d.business_segment, d.won_date, f.first_sale,
               date_diff('day', d.won_date, f.first_sale) days_to_first_sale
        from read_csv_auto('{deals}') d join read_csv_auto('{mql}') m using(mql_id) left join first_sale f using(seller_id)""").pl()
    sold = df.filter(pl.col("first_sale").is_not_null())
    return {"outputs": [write(df, ctx.out / "sellers" / "acquisition.parquet")],
            "summary": {"won": df.height, "with_sales": sold.height,
                        "days_to_first_sale_median": float(sold["days_to_first_sale"].median()) if sold.height else None,
                        "origin_mix": dict(df.group_by("origin").len().sort("len", descending=True).head(6).iter_rows())}}


def _easter(y: int) -> dt.date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    return dt.date(y, month, (h + l_ - 7 * m + 114) % 31 + 1)


def step_context(ctx: Ctx) -> dict:
    """Daily context: Brazil national holidays, shopping events, documented shocks, BCB macro series."""
    rows = []
    for y in (2016, 2017, 2018):
        e = _easter(y)
        fixed = {(1, 1): "new_year", (4, 21): "tiradentes", (5, 1): "labour_day", (9, 7): "independence",
                 (10, 12): "our_lady_aparecida", (11, 2): "all_souls", (11, 15): "republic", (12, 25): "christmas"}
        for (mo, d), n in fixed.items():
            rows.append((dt.date(y, mo, d), n, "holiday"))
        for off, n in ((-48, "carnival_mon"), (-47, "carnival_tue"), (-2, "good_friday"), (60, "corpus_christi")):
            rows.append((e + dt.timedelta(days=off), n, "holiday"))
        nov1 = dt.date(y, 11, 1)
        bf = nov1 + dt.timedelta(days=(4 - nov1.weekday()) % 7 + 21)
        rows.append((bf, "black_friday", "shopping_event"))
    strike = [(dt.date(2018, 5, 21) + dt.timedelta(days=i), "truckers_strike", "shock") for i in range(11)]
    ev = pl.DataFrame(rows + strike, schema=["date", "name", "kind"], orient="row")
    days = pl.DataFrame({"date": pl.date_range(dt.date(2016, 1, 1), dt.date(2018, 12, 31), eager=True)})
    daily = days.join(ev.group_by("date").agg(pl.col("name").str.join("|").alias("events"),
                                              pl.col("kind").str.join("|").alias("kinds")), on="date", how="left")
    macro = {}
    if ctx.allow_download:
        for name, code in BCB_SERIES.items():
            with urllib.request.urlopen(BCB.format(code=code), timeout=60) as r:
                js = json.load(r)
            s = pl.DataFrame({"date": [dt.datetime.strptime(x["data"], "%d/%m/%Y").date() for x in js],
                              name: [float(x["valor"]) for x in js]})
            daily = daily.join(s, on="date", how="left").with_columns(pl.col(name).forward_fill())
            macro[name] = s.height
    else:
        macro = "skipped (needs --allow-download for the BCB SGS API)"
    return {"outputs": [write(daily, ctx.out / "context" / "daily.parquet")],
            "summary": {"days": daily.height, "event_days": ev.height, "macro_series": macro,
                        "note": "strike dates from public reporting (21 May - 31 May 2018)"}}


def step_prices(ctx: Ctx) -> dict:
    """Price-change frequency and promo depth from M5 (if downloaded; needs Kaggle competition rules)."""
    src = need_local(ctx, "m5/sell_prices.csv")
    ctx.con.execute(f"""create or replace view sp as select item_id, store_id, wm_yr_wk, sell_price,
        lag(sell_price) over (partition by item_id, store_id order by wm_yr_wk) prev from read_csv_auto('{src}')""")
    res = {"weekly_change_prob": float(ctx.one("select avg((sell_price <> prev)::int) from sp where prev is not null")[0]),
           "drop_depth": quantiles(ctx, "select 1 - sell_price / prev x from sp where prev is not null and sell_price < prev")}
    p = ctx.out / "prices" / "prices.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return {"outputs": [{"file": "prices/prices.json"}], "summary": res}


def step_repeat(ctx: Ctx) -> dict:
    """Inter-purchase intervals (Instacart, if downloaded) next to Olist's real repeat rate."""
    orders = need_local(ctx, "olist/olist_orders_dataset.csv")
    cust = need_local(ctx, "olist/olist_customers_dataset.csv")
    res = {"olist_repeat_buyer_share": float(ctx.one(f"""select avg((n>1)::int) from (select customer_unique_id, count(*) n
        from read_csv_auto('{orders}') o join read_csv_auto('{cust}') c using(customer_id) group by 1)""")[0])}
    ic = ctx.local("instacart/orders.csv")
    if ic.exists():
        res["instacart_days_since_prior"] = quantiles(ctx, f"select days_since_prior_order x from read_csv_auto('{ic}')")
    else:
        res["instacart"] = "not present: accept the Instacart competition rules on Kaggle, then download into <data>/instacart/"
    p = ctx.out / "repeat" / "repeat.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return {"outputs": [{"file": "repeat/repeat.json"}], "summary": res}


def step_fraud(ctx: Ctx) -> dict:
    need_local(ctx, "fraud/YelpChi.mat")
    raise Skip("fraud source present but its parser is not specified yet (decision pending: YelpChi vs Amazon fraud graph)")


def step_anchor(ctx: Ctx) -> dict:
    """Re-anchoring solve: free levels chosen so the stitched chain reproduces Olist volume.

    chain per session: [search -> 3 positions -> click] -> view(s) -> cart -> purchase (+ direct purchase)
    Free parameters: sessions/day S and base top-position click rate c0.
      c0: expected clicked views per session (sum_k c0 * rel_k) equals the M2/Retailrocket views per session
      S : S * P(order | session) equals Olist orders/day on the fit period
    Hold-out check: same S vs Olist orders/day in 2018-01..08 (reports drift; growth is not modelled).
    """
    orders = need_local(ctx, "olist/olist_orders_dataset.csv")
    f = json.loads((ctx.out / "funnel" / "funnel.json").read_text()) if (ctx.out / "funnel" / "funnel.json").exists() else None
    if not f:
        raise Skip("needs the funnel step")
    pos = json.loads((ctx.out / "positions" / "position_curve.json").read_text()) if (ctx.out / "positions" / "position_curve.json").exists() else None
    rel = [p["relative_to_top"] for p in pos["random"]["by_position"]] if pos else [1.0, 0.8, 0.65]   # fallback A-15
    views = f["views_mean"]
    sess_file = ctx.out / "sessions" / "by_locale.parquet"
    if sess_file.exists():
        m2 = pl.read_parquet(sess_file)["mean_len"].mean()
        views_lo, views_hi = sorted([views, float(m2)])            # conflict rule: Retailrocket default, M2 bound
    else:
        views_lo = views_hi = views
    c0 = min(1.0, views / sum(rel))
    p_order = f["p_buy"]
    fit = float(ctx.one(f"""select median(n) from (select cast(order_purchase_timestamp as date) d, count(*) n
        from read_csv_auto('{orders}') where order_purchase_timestamp >= '2017-10-01' and order_purchase_timestamp < '2018-01-01' group by 1)""")[0])
    hold = float(ctx.one(f"""select median(n) from (select cast(order_purchase_timestamp as date) d, count(*) n
        from read_csv_auto('{orders}') where order_purchase_timestamp >= '2018-01-01' and order_purchase_timestamp < '2018-09-01' group by 1)""")[0])
    S = fit / p_order
    ads = ctx.out / "ads" / "ads.json"
    res = {"sessions_per_day": S, "top_position_click_rate": c0, "position_relative": rel,
           "views_per_session_default": views, "views_per_session_bounds": [views_lo, views_hi],
           "p_order_per_session": p_order, "olist_orders_per_day_fit": fit, "olist_orders_per_day_holdout": hold,
           "holdout_level_error": (fit - hold) / hold,
           "ads_conversion_lift": json.loads(ads.read_text())["conversion_lift_ratio"] if ads.exists() else None,
           "position_source": "open_bandit" if pos else "assumed A-15"}
    p = ctx.out / "anchor" / "anchored.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return {"outputs": [{"file": "anchor/anchored.json"}], "summary": res}


STEPS = [
    Step("catalog", "bridge", "bridged:esci+reviews2023",
         [Source("reviews23 id map", "asin2category.json | rev_ids.tsv", approx="1.25 GB, or local"),
          Source("ESCI-US text columns", HF_ESCI, remote=True, approx="~0.5 GB read (Parquet projection)"),
          Source("Olist items/products", "olist/", optional=True)], step_catalog, ["A-13"], desc="real titles/brands per mapped category"),
    Step("queries", "bridge", "bridged:esci", [Source("ESCI cache from catalog", "_cache/esci_us.parquet")], step_queries,
         ["A-13"], needs=["catalog"], desc="real queries + E/S/C/I labels"),
    Step("positions", "transfer", "calibrated:transfer:open_bandit",
         [Source("Open Bandit random/all", HF_OBD_RANDOM, remote=True, approx="0.70 GB CSV")], step_positions,
         ["A-15"], desc="relative click by position, logged propensities"),
    Step("sessions", "transfer", "calibrated:transfer:amazon_m2", [Source("M2 sessions", "m2/sessions_train.csv", approx="259 MB")],
         step_sessions, [], desc="session length, adjacent revisits"),
    Step("funnel", "transfer", "calibrated:transfer:retailrocket", [Source("Retailrocket events", "retailrocket/events.csv")],
         step_funnel, ["A-02"], desc="funnel probabilities + timing"),
    Step("stock", "transfer", "calibrated:transfer:retailrocket",
         [Source("Retailrocket item properties", "retailrocket/item_properties_part1.csv (+part2)", approx="0.9 GB")],
         step_stock, ["A-08"], desc="stock-out share and spell lengths"),
    Step("ads", "transfer", "calibrated:transfer:criteo", [Source("Criteo uplift v2.1", HF_CRITEO, remote=True, approx="0.31 GB gz")],
         step_ads, ["A-14"], desc="ad lift, non-compliance"),
    Step("review_text", "pool", "bridged:pool:olist", [Source("Olist reviews", "olist/olist_order_reviews_dataset.csv")],
         step_review_text, [], desc="real review texts by score"),
    Step("returns", "join+transfer", "observed:olist + calibrated:transfer:uci",
         [Source("Olist orders", "olist/olist_orders_dataset.csv"), Source("UCI Online Retail II", "uci/online_retail_II.xlsx", optional=True)],
         step_returns, ["A-16"], desc="cancellation / return rates"),
    Step("sellers", "join", "matched:olist_funnel",
         [Source("Olist funnel", "olist/funnel/"), Source("Olist items/orders", "olist/")], step_sellers, [], desc="acquisition -> first sale"),
    Step("context", "join", "observed:calendar+bcb",
         [Source("computed holidays", "(built in)"), Source("BCB SGS API", "api.bcb.gov.br", remote=True, approx="<1 MB", optional=True)],
         step_context, [], desc="holidays, Black Friday, strike, macro"),
    Step("prices", "transfer", "calibrated:transfer:m5", [Source("M5 sell_prices", "m5/sell_prices.csv", approx="~200 MB")],
         step_prices, ["A-04"], desc="price change frequency, promo depth"),
    Step("repeat", "join+transfer", "observed:olist + calibrated:transfer:instacart",
         [Source("Olist orders/customers", "olist/"), Source("Instacart orders", "instacart/orders.csv", optional=True)],
         step_repeat, [], desc="repeat share, inter-purchase intervals"),
    Step("fraud", "transfer", "calibrated:transfer:fraud", [Source("YelpChi", "fraud/YelpChi.mat")], step_fraud, [], desc="pending"),
    Step("anchor", "solve", "calibrated:anchor:olist", [Source("Olist orders", "olist/olist_orders_dataset.csv")], step_anchor,
         ["A-15"], needs=["funnel", "sessions", "positions", "ads"], desc="re-anchor levels on Olist"),
]
BY_NAME = {s.name: s for s in STEPS}


# ----------------------------------------------------------------------------- commands
def plan(args) -> int:
    data = pathlib.Path(args.data).expanduser()
    print(f"data: {data}\nout:  {pathlib.Path(args.out)}\n")
    print(f"{'step':12} {'method':14} {'sources':58} status")
    for s in STEPS:
        parts, status = [], "ready"
        for src in s.sources:
            if src.remote:
                tag = "remote" + (f", {src.approx}" if src.approx else "")
                if not args.allow_download and not src.optional:
                    status = "needs --allow-download"
            elif src.where.startswith("(") or src.where.startswith("_cache") or "|" in src.where:
                tag = "derived/local"
            else:
                first = src.where.split(" ")[0]
                ok = (data / first).exists()
                tag = "ok" if ok else ("absent, optional" if src.optional else "MISSING")
                if not ok and not src.optional:
                    status = "skip (source missing)"
            parts.append(f"{src.name} [{tag}]")
        if s.needs:
            parts.append("after: " + ",".join(s.needs))
        print(f"{s.name:12} {s.method:14} {'; '.join(parts)[:58]:58} {status}")
        if len("; ".join(parts)) > 58:
            print(f"{'':28}{'; '.join(parts)[58:]}")
    print("\nNothing was read or written. Run with: python scripts/stitch.py run [--steps ...] [--allow-download]")
    return 0


def run(args) -> int:
    data, out = pathlib.Path(args.data).expanduser(), pathlib.Path(args.out)
    names = args.steps.split(",") if args.steps else [s.name for s in STEPS]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        print(f"unknown steps: {unknown}; known: {list(BY_NAME)}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(data, out, args.allow_download, args.with_bts)
    mpath = out / "stitch_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"steps": {}}
    manifest.update({"created": dt.datetime.now().isoformat(timespec="seconds"), "seed": SEED, "data": str(data)})
    rc = 0
    for s in [BY_NAME[n] for n in names]:
        t0 = time.time()
        rec = {"method": s.method, "provenance": s.prov, "assumptions": s.assumptions,
               "inputs": [], "description": s.desc}
        for src in s.sources:
            p = data / src.where.split(" ")[0]
            if not src.remote and p.is_file():
                rec["inputs"].append({"path": str(p), "sha256": sha256(p)})
            else:
                rec["inputs"].append({"where": src.where, "remote": src.remote})
        try:
            res = s.fn(ctx)
            rec.update(status="ok", **res)
        except Skip as e:
            rec.update(status="skipped", reason=str(e))
        except Exception as e:  # noqa: BLE001 - record and continue with other steps
            rec.update(status="failed", error=f"{type(e).__name__}: {e}")
            rc = 1
        rec["seconds"] = round(time.time() - t0, 1)
        manifest["steps"][s.name] = rec
        print(f"{s.name:12} {rec['status']:8} {rec['seconds']:>7}s  {rec.get('reason') or rec.get('error') or ''}")
        mpath.write_text(json.dumps(manifest, indent=1, default=str))
    print(f"\nmanifest: {mpath}")
    return rc


SUPERSEDED = """scripts/stitch.py is SUPERSEDED (22 Nov 2026) and refuses to run.
It predates the dataset audit and the adopted architecture: it reads ~/.cache/mkt-probe, uses Amazon-M2
(dropped; Taobao now backs sessions), takes position effects from Open Bandit (no position effect measured;
Expedia random-order replaces it) and writes no transport cards or trained_through dates.
Per ARCHITECTURE.md §13.0 phase 7, transferred sources are added one at a time with a transport card and a
held-out gain. Set MKT_ALLOW_OLD_STITCH=1 to run the old plan-only mode anyway."""


def main(argv=None) -> int:
    import os
    if os.environ.get("MKT_ALLOW_OLD_STITCH") != "1":
        print(SUPERSEDED, file=sys.stderr)
        return 3
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "run"])
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="raw data root (default: %(default)s)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output dir (default: %(default)s)")
    ap.add_argument("--steps", help="comma-separated subset, e.g. sessions,funnel,anchor")
    ap.add_argument("--allow-download", action="store_true", help="allow steps that read remote sources")
    ap.add_argument("--with-bts", action="store_true", help="positions: also read the 6.3 GB Bernoulli-TS log")
    a = ap.parse_args(argv)
    return plan(a) if a.command == "plan" else run(a)


if __name__ == "__main__":
    sys.exit(main())
