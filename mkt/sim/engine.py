"""LEGACY prototype (22 Nov 2026; build path is mkt/sim/chain.py). Discrete-time structural simulator (docs §7, decision L2).

One tick = one hour. Within a tick mechanisms run in a fixed order (a DAG); feedback flows
only across ticks (inventory, ratings, restocks), so each tick is a well-defined SCM.
Every random draw uses rng.stream(seed, <mechanism>, tick): a counterfactual twin run with
different interventions shares all exogenous noise.

Mechanisms (ids from docs §5):
  arrivals       M-19/M-20 calendar & demand levers        calibrated (Olist)
  offer choice   M-05 competition (price, rating, stock)   structure Olist, coefficients assumed (A-05)
  funnel         M-03 view -> cart -> checkout             transferred (Retailrocket, A-02)
  price response M-04                                      assumed elasticity (A-04)
  infra          M-17 checkout errors, latency             assumed (A-07)
  stock          M-06/M-07 allocation, stockouts, restock  assumed policy (A-08)
  fulfilment     M-08 approval, handling, transit, promise calibrated (Olist)
  reviews        M-09/M-10 score | lateness, presence      calibrated (Olist) + latent defects (A-09)
  accounting     M-24 amount = price + freight = payment   deterministic
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from .. import rng
from ..contracts import EVENT_STREAMS, Intervention, conform
from ..interventions import Schedule
from .world import World

H = 3_600_000_000          # one hour in microseconds
DAY = 24 * H
BUCKETS = ["on_time", "late_1_3", "late_4_10", "late_gt10"]
DEVICES = np.array(["mobile", "desktop"])


@dataclass
class SimResult:
    events: pl.DataFrame          # world events (complete, no measurement effects)
    orders: pl.DataFrame          # per-order latent truth (defect, true lateness, ...)
    stats: dict


def _us(dt: datetime) -> int:
    return int(np.datetime64(dt, "us").astype(np.int64))


class _Events:
    def __init__(self):
        self.batches: list[pl.DataFrame] = []

    def add(self, etype: str, t: np.ndarray, **cols):
        n = len(t)
        if n == 0:
            return
        d = {"event_type": np.full(n, etype), "event_time": np.asarray(t, np.int64)}
        for k, v in cols.items():
            d[k] = np.full(n, v) if np.isscalar(v) else np.asarray(v)
        self.batches.append(pl.DataFrame(d))

    def frame(self) -> pl.DataFrame:
        return pl.concat(self.batches, how="diagonal_relaxed") if self.batches else pl.DataFrame()


def simulate(world: World, start: datetime, days: int, interventions: list[Intervention] | None = None,
             world_tag: str = "w") -> SimResult:
    P, A, F = world.params, world.params["assumed"], world.params["funnel"]
    seed = world.seed
    ivs = [i for i in (interventions or []) if i.layer == "world"]
    sched = Schedule(ivs, world.aliases)
    start_us, n_ticks = _us(start), days * 24
    end_us = start_us + n_ticks * H

    hour = np.array(P["calendar"]["hour"])
    dow = np.array(P["calendar"]["dow_mon_first"])
    sessions_per_day = P["daily_orders"] * world.scale / F["p_buy"]
    e0 = A["checkout_error_base"]
    elast = A["price_elasticity"]
    cats = np.array(world.categories)
    n_off, n_sel = world.n_offers, len(world.seller_weight)

    off_cat = world.product_category[world.offer_product]
    off_attrs = {"category": cats[off_cat], "seller": world.seller_id(world.offer_seller),
                 "product": np.char.add("p", world.offer_product.astype(str)),
                 "seller_region": world.seller_region[world.offer_seller]}
    prod_base = np.zeros(world.product_category.shape[0])
    first_offer = np.unique(world.offer_product, return_index=True)[1]
    prod_base[world.offer_product[first_offer]] = world.offer_price[first_offer]
    rel_price = world.offer_price / prod_base[world.offer_product]

    # category-sorted offer layout for vectorised weighted choice
    order_ = np.argsort(off_cat, kind="stable")
    seg_start = np.searchsorted(off_cat[order_], np.arange(len(cats)))
    seg_end = np.searchsorted(off_cat[order_], np.arange(len(cats)), side="right")

    # review score CPT adjusted for explicit latent defects (marginal preserved at base defect rate)
    d0 = float(world.product_defect.mean())
    cpt = np.array([P["score_given_late"][b] for b in BUCKETS])
    adj = cpt.copy()
    adj[:, 0] = np.maximum(0, (cpt[:, 0] - d0) / (1 - d0))
    adj /= adj.sum(1, keepdims=True)
    cpt_cum = np.cumsum(adj, 1)
    rq = np.array(P["review"]["offset_days_q"])
    rq_p = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    # state
    stock = np.ceil(world.offer_daily_demand * A["stock_cover_days"]).astype(np.int64) + 1
    cover = stock.copy()
    reorder_pt = np.ceil(world.offer_daily_demand * A["reorder_point_days"])
    pending = np.zeros(n_off, bool)
    restock_q: dict[int, list] = defaultdict(list)
    rating_sum = np.full(n_sel, 4.1 * 20.0)
    rating_n = np.full(n_sel, 20.0)
    review_q: dict[int, list] = defaultdict(list)
    prev_pm = np.ones(n_off)
    ev = _Events()
    order_truth = []
    n_sessions = n_orders = 0
    tel = defaultdict(float)

    for k in range(n_ticks):
        t0 = start_us + k * H
        dt64 = np.datetime64(t0, "us")
        h = int((dt64 - dt64.astype("datetime64[D]")) // np.timedelta64(1, "h"))
        dw = int(((dt64.astype("datetime64[D]").astype(np.int64)) + 3) % 7)   # 1970-01-01 was a Thursday

        # --- feedback arriving this tick -------------------------------------------------
        for offs, qty in restock_q.pop(k, []):
            stock[offs] += qty
            pending[offs] = False
            ev.add("restock", np.full(len(offs), t0), offer=offs, amount=qty.astype(float))
        for sel, sc in review_q.pop(k, []):
            np.add.at(rating_sum, sel, sc)
            np.add.at(rating_n, sel, 1.0)

        # --- levers (interventions) -------------------------------------------------------
        err = sched.apply("infra.checkout.error_rate", np.full(2, e0), {"device": DEVICES}, t0).clip(0, 1)
        pm = sched.apply("price.multiplier", np.ones(n_off), off_attrs, t0).clip(0.05, 20)
        dm = sched.apply("demand.multiplier", np.ones(len(cats)), {"category": cats}, t0).clip(0, 100)
        changed = np.flatnonzero(pm != prev_pm)
        if len(changed):
            ev.add("price_change", np.full(len(changed), t0), offer=changed,
                   amount=world.offer_price[changed] * pm[changed])
            prev_pm = pm
        for iv in ivs:
            if iv.observable and iv.window and _us(iv.window[0]) // H == t0 // H:
                ev.add("announcement", np.array([t0]), metric=f"{iv.target}:{iv.selector}", amount=float(iv.value))

        # --- arrivals (M-19/M-20) --------------------------------------------------------
        g = rng.stream(seed, "arrivals", k)
        cat_p = world.category_share * dm
        lam = sessions_per_day * 7 * dow[dw] * hour[h] * cat_p.sum()
        n = int(g.poisson(lam))
        if n:
            s_cat = g.choice(len(cats), size=n, p=cat_p / cat_p.sum())
            buyer = g.integers(0, len(world.buyer_device), n)
            s_dev = world.buyer_device[buyer]
            s_reg = world.buyer_region[buyer]
            s_time = t0 + (g.random(n) * H).astype(np.int64)
            sid = n_sessions + np.arange(n)
            n_sessions += n
            ev.add("session_start", s_time, session=sid, buyer=buyer, device=s_dev, region=s_reg, category=cats[s_cat])

            # --- offer choice (M-05) ---------------------------------------------------------
            rating = rating_sum / rating_n
            w = (world.offer_popularity * (rel_price * pm) ** (-A["choice_price_beta"])
                 * np.exp(A["choice_rating_beta"] * (rating[world.offer_seller] - 4.0)) * (stock > 0))
            cs = np.cumsum(w[order_])
            base = np.where(seg_start > 0, cs[np.maximum(seg_start - 1, 0)], 0.0)
            tot = np.where(seg_end > seg_start, cs[np.maximum(seg_end - 1, 0)], 0.0) - base

            def pick(c: np.ndarray, u: np.ndarray) -> np.ndarray:
                idx = np.searchsorted(cs, base[c] + u * tot[c], side="right")
                idx = np.clip(idx, seg_start[c], np.maximum(seg_end[c] - 1, seg_start[c]))
                out = order_[np.minimum(idx, len(order_) - 1)]
                return np.where(tot[c] > 0, out, -1)

            gc = rng.stream(seed, "choice", k)
            offer = pick(s_cat, gc.random(n))
            has = offer >= 0
            vt = s_time + (gc.uniform(5, 60, n) * 1e6).astype(np.int64)
            sel = np.flatnonzero(has)
            ev.add("product_view", vt[sel], session=sid[sel], buyer=buyer[sel], device=s_dev[sel], region=s_reg[sel], offer=offer[sel])
            extra = gc.poisson(max(F["views_mean"] - 1, 0), n) * has
            if extra.sum():
                rep = np.repeat(np.arange(n), extra)
                eo = pick(s_cat[rep], gc.random(len(rep)))
                ok = eo >= 0
                et = vt[rep] + (gc.uniform(20, 600, len(rep)) * 1e6).astype(np.int64)
                ev.add("product_view", et[ok], session=sid[rep][ok], buyer=buyer[rep][ok], device=s_dev[rep][ok],
                       region=s_reg[rep][ok], offer=eo[ok])

            # --- funnel, price response, infra (M-03, M-04, M-17) -----------------------------
            gf = rng.stream(seed, "funnel", k)
            u_c, u_k, u_d, u_e, u_t = gf.random((5, n))
            safe_off = np.maximum(offer, 0)
            pf = np.where(has, (pm[safe_off]) ** elast, 0.0).clip(0, 5)
            cart = has & (u_c < F["p_cart"] * pf)
            pc = np.minimum(1, F["p_buy_given_cart"] * pf / (1 - e0))
            pd = np.minimum(1, F["p_buy_direct"] * pf / (1 - e0) / max(1 - F["p_cart"], 1e-9))
            go = (cart & (u_k < pc)) | (has & ~cart & (u_d < pd))
            dev_i = (s_dev == "desktop").astype(int)
            error = go & (u_e < err[dev_i])
            ok_ck = go & ~error
            t_cart = vt + (gf.uniform(30, 600, n) * 1e6).astype(np.int64)
            t_ck = t_cart + (gf.uniform(30, 900, n) * 1e6).astype(np.int64)
            for name, m, tt in (("add_to_cart", cart, t_cart), ("checkout_start", go, t_ck), ("checkout_error", error, t_ck + 2_000_000)):
                i = np.flatnonzero(m)
                ev.add(name, tt[i], session=sid[i], buyer=buyer[i], device=s_dev[i], region=s_reg[i], offer=offer[i])
            for d in (0, 1):
                tel[(k, d, "ck")] += float((go & (dev_i == d)).sum())
                tel[(k, d, "err")] += float((error & (dev_i == d)).sum())

            # --- stock allocation (M-06) ------------------------------------------------------
            cand = np.flatnonzero(ok_ck)
            if len(cand):
                co = offer[cand]
                srt = np.lexsort((u_t[cand], co))
                co_s = co[srt]
                first = np.searchsorted(co_s, co_s)
                rank = np.empty(len(cand), np.int64)
                rank[srt] = np.arange(len(cand)) - first
                avail = rank < stock[co]
                lost = cand[~avail]
                ev.add("order_unavailable", t_ck[lost] + 5_000_000, session=sid[lost], buyer=buyer[lost],
                       device=s_dev[lost], region=s_reg[lost], offer=offer[lost])
                won = cand[avail]
                before = stock.copy()
                np.subtract.at(stock, offer[won], 1)
                assert (stock >= 0).all(), "inventory invariant violated: negative stock"
                out_now = np.flatnonzero((before > 0) & (stock <= 0))
                ev.add("stock_out", np.full(len(out_now), t0 + H - 1), offer=out_now)
                if len(won):
                    n_orders = _orders(ev, order_truth, world, sched, won, n_orders, offer, buyer, sid, s_dev, s_reg,
                                       t_ck, pm, seed, k, end_us, cpt_cum, rq, rq_p, review_q, start_us)

        # --- replenishment policy (M-07, assumed A-08) ---------------------------------------
        need = np.flatnonzero((stock <= reorder_pt) & ~pending)
        if len(need):
            gr = rng.stream(seed, "restock", k)
            lead = np.maximum(1, np.round(gr.lognormal(A["restock_lead_days"]["mu"], A["restock_lead_days"]["sigma"], len(need)) * 24)).astype(int)
            qty = np.maximum(cover[need] - stock[need], 1)
            pending[need] = True
            for kk in np.unique(lead):
                m = lead == kk
                if k + kk < n_ticks:
                    restock_q[k + int(kk)].append((need[m], qty[m]))

        # --- telemetry (measured infra state) ------------------------------------------------
        gt = rng.stream(seed, "telemetry", k)
        for d in (0, 1):
            ck, er = tel.pop((k, d, "ck"), 0.0), tel.pop((k, d, "err"), 0.0)
            rate = er / ck if ck else np.nan
            lat = A["latency_p95_ms_base"] * (1 + 8 * err[d]) * gt.lognormal(0, 0.05)
            ev.add("metric_sample", np.array([t0 + H - 1_000_000] * 2), device=DEVICES[d],
                   metric=np.array(["checkout_error_rate", "latency_p95_ms"]), value=np.array([rate, lat]))

    events = _finalize(ev.frame(), world, world_tag)
    orders = pl.DataFrame(order_truth and {k: np.concatenate([o[k] for o in order_truth]) for k in order_truth[0]}) \
        if order_truth else pl.DataFrame()
    if orders.height:
        orders = orders.with_columns(pl.format(f"{world_tag}:o{{}}", pl.col("order")).alias("order_id")).drop("order")
    return SimResult(events=events, orders=orders,
                     stats={"ticks": n_ticks, "sessions": n_sessions, "orders": n_orders, "events": events.height})


def _orders(ev, order_truth, world: World, sched: Schedule, won, n_orders, offer, buyer, sid, s_dev, s_reg,
            t_ck, pm, seed, k, end_us, cpt_cum, rq, rq_p, review_q, start_us) -> int:
    """Fulfilment and feedback for orders created this tick (M-08, M-09, M-10, M-24)."""
    P, A = world.params, world.params["assumed"]
    go_ = rng.stream(seed, "fulfilment", k)
    n = len(won)
    off = offer[won]
    sel = world.offer_seller[off]
    prod = world.offer_product[off]
    oid = n_orders + np.arange(n)
    t_order = t_ck[won] + 10_000_000
    price = world.offer_price[off] * pm[off]
    amount = np.round(price + world.offer_freight[off], 2)
    breg = s_reg[won]
    same = breg == world.seller_region[sel]

    ap = P["approval_hours"]
    approved = t_order + (go_.lognormal(ap["mu"], ap["sigma"], n).clip(0, 240) * H).astype(np.int64)
    hh = P["handling_hours"]
    sw = np.sqrt(max(hh["sigma"] ** 2 - hh["between_seller_sd"] ** 2, 0.05))
    shipped = approved + (go_.lognormal(world.seller_handling_mu[sel], sw).clip(0.1, 24 * 60) * H).astype(np.int64)
    tr = P["transit_days"]
    tdays = np.where(same, go_.lognormal(tr["same"]["mu"], tr["same"]["sigma"], n),
                     go_.lognormal(tr["diff"]["mu"], tr["diff"]["sigma"], n))
    seller_ids = world.seller_id(sel)
    tmult = sched.apply("carrier.transit_multiplier", np.ones(n), {"region": breg, "seller": seller_ids}, shipped)
    delivered = shipped + (tdays * tmult * DAY).astype(np.int64)
    pr = P["promise_days"]
    pdays = np.where(same, go_.normal(pr["same"]["mean"], pr["same"]["sd"], n), go_.normal(pr["diff"]["mean"], pr["diff"]["sd"], n))
    pdays = np.maximum(3, np.round(pdays + sched.apply("promise.days_shift", np.zeros(n), {"region": breg, "seller": seller_ids}, t_order)))
    promised = t_order + (pdays * DAY).astype(np.int64)
    late_days = np.ceil((delivered - promised) / DAY)
    bucket = np.select([late_days <= 0, late_days <= 3, late_days <= 10], [0, 1, 2], 3)
    cat = np.array(world.categories)[world.product_category[prod]]
    dprob = sched.apply("product.defect_prob", world.product_defect[prod],
                        {"category": cat, "seller": seller_ids, "product": np.char.add("p", prod.astype(str))}, t_order)
    u_def, u_s, u_r, u_o = go_.random((4, n))
    defect = u_def < dprob
    score = np.where(defect, 1, (u_s[:, None] > cpt_cum[bucket]).sum(1) + 1).astype(float)
    reviewed = u_r < P["review"]["p_review"]
    rtime = delivered + (np.interp(u_o, np.concatenate([[0], np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]), [1]]),
                                   np.concatenate([[rq[0]], rq, [rq[-1]]])) * DAY).astype(np.int64)
    rtime = np.maximum(rtime, shipped + H)        # survey cannot precede shipment (Olist offsets are vs. delivery)

    common = dict(order=oid, buyer=buyer[won], session=sid[won], device=s_dev[won], region=breg, offer=off)

    def emit(etype, t, mask=None, **extra):
        m = (t < end_us) if mask is None else (mask & (t < end_us))
        i = np.flatnonzero(m)
        ev.add(etype, t[i], **{c: v[i] for c, v in common.items()}, **{c: (v[i] if not np.isscalar(v) else v) for c, v in extra.items()})

    emit("order_created", t_order, amount=amount, value=pdays)
    emit("order_item", t_order, amount=price, value=world.offer_freight[off])
    emit("payment_approved", approved, amount=amount)
    emit("handed_to_carrier", shipped)
    emit("delivered", delivered, value=late_days)
    emit("review_submitted", rtime, mask=reviewed, value=score)
    due = reviewed & (rtime < end_us)
    tick_of = (rtime[due] - start_us) // H
    for kk in np.unique(tick_of):
        m = tick_of == kk
        review_q[int(kk)].append((sel[due][m], score[due][m]))
    order_truth.append({"order": oid, "defect": defect, "late_days_true": late_days, "promise_days": pdays,
                        "transit_multiplier": tmult, "same_state": same})
    return n_orders + n


def _finalize(df: pl.DataFrame, world: World, tag: str) -> pl.DataFrame:
    if df.height == 0:
        return conform(pl.DataFrame(), with_available=False)
    for c in ("session", "buyer", "offer", "order", "device", "region", "category", "amount", "value", "metric"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    off = df["offer"].cast(pl.Int64)
    valid = off.is_not_null() & (off >= 0)
    offc = off.fill_null(0).clip(0).to_numpy()
    seller = np.char.add("s", world.offer_seller[offc].astype(str))
    product = np.char.add("p", world.offer_product[offc].astype(str))
    cat = np.array(world.categories)[world.product_category[world.offer_product[offc]]]
    df = df.with_columns(
        pl.when(valid).then(pl.Series(seller)).alias("seller_id"),
        pl.when(valid).then(pl.Series(product)).alias("product_id"),
        pl.when(valid).then(pl.Series(cat)).otherwise(pl.col("category")).alias("category"),
        pl.format(f"{tag}:b{{}}", pl.col("buyer")).alias("buyer_id"),
        pl.format(f"{tag}:x{{}}", pl.col("session")).alias("session_id"),
        pl.format(f"{tag}:o{{}}", pl.col("order")).alias("order_id"),
        pl.from_epoch(pl.col("event_time"), time_unit="us").alias("event_time"),
        pl.col("event_type").replace_strict(EVENT_STREAMS, default="web").alias("stream"),
        pl.lit("BRL").alias("currency"),
        pl.lit("mechanism").alias("prov_class"),
        pl.concat_str([pl.lit("prov:sim:"), pl.col("event_type")]).alias("prov_id"),
    )
    df = df.sort(["event_time", "event_type"], maintain_order=True)
    df = df.with_columns(pl.format(f"{tag}:e{{}}", pl.int_range(pl.len())).alias("event_id"))
    return conform(df, with_available=False)
