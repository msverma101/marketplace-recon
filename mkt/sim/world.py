"""Synthetic population built from calibrated parameters (docs §7 'Seed population').

Entities are drawn from fitted distributions, not copied from Olist rows: seller sizes are
resampled from the empirical size distribution, prices from per-category log-normals,
regions from observed shares. Latent attributes (seller handling speed, product defect
probability) exist only here and in the truth store.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import rng


@dataclass
class World:
    seed: int
    scale: float
    params: dict
    # sellers
    seller_region: np.ndarray
    seller_weight: np.ndarray
    seller_handling_mu: np.ndarray          # latent: log-hours
    # products
    product_category: np.ndarray            # category index
    product_defect: np.ndarray              # latent: P(defect)
    # offers (product x seller)
    offer_product: np.ndarray
    offer_seller: np.ndarray
    offer_price: np.ndarray                 # base price
    offer_freight: np.ndarray
    offer_popularity: np.ndarray
    offer_daily_demand: np.ndarray          # expected units/day at base conditions
    # buyers
    buyer_region: np.ndarray
    buyer_device: np.ndarray
    categories: list[str]
    category_share: np.ndarray
    aliases: dict[str, str] = field(default_factory=dict)
    shared_products: np.ndarray = field(default_factory=lambda: np.zeros(0, int))

    @property
    def n_offers(self) -> int:
        return len(self.offer_product)

    def seller_id(self, i) -> np.ndarray:
        return np.char.add("s", np.asarray(i).astype(str))


def _choice(g: np.random.Generator, shares: dict, n: int) -> np.ndarray:
    keys = np.array(list(shares.keys()))
    p = np.array(list(shares.values()), float)
    return keys[g.choice(len(keys), size=n, p=p / p.sum())]


def build_world(params: dict, seed: int, scale: float = 1.0) -> World:
    P, A = params, params["assumed"]
    g = rng.stream(seed, "population")
    sizes = np.array(P["seller_sizes"], float)
    n_sellers = max(30, int(round(len(sizes) * scale)))
    n_products = max(200, int(round(n_sellers * 10.6)))            # Olist: 32,951 products / 3,095 sellers
    daily_orders = P["daily_orders"] * scale

    seller_weight = g.choice(sizes, size=n_sellers) * g.lognormal(0, 0.25, n_sellers)
    seller_region = _choice(g, P["seller_regions"], n_sellers)
    hh = P["handling_hours"]
    seller_handling_mu = hh["mu"] + g.normal(0, hh["between_seller_sd"], n_sellers)

    cats = P["categories"]
    names = [c["name"] for c in cats]
    share = np.array([c["share"] for c in cats])
    share = share / share.sum()
    product_category = g.choice(len(cats), size=n_products, p=share)
    owner = g.choice(n_sellers, size=n_products, p=seller_weight / seller_weight.sum())
    mu = np.array([c["price_mu"] for c in cats])[product_category]
    sd = np.array([c["price_sigma"] for c in cats])[product_category]
    price = np.round(np.exp(g.normal(mu, sd)), 2).clip(1.0)
    freight = np.round(price * np.array([c["freight_ratio"] for c in cats])[product_category], 2)
    defect = (A["defect_prob"] * g.lognormal(0, 0.5, n_products)).clip(0, 0.5)
    product_pop = g.lognormal(0, 1.0, n_products) * seller_weight[owner] / seller_weight.mean()

    offer_product, offer_seller, offer_price, offer_freight = [np.arange(n_products)], [owner], [price], [freight]
    # competition (M-05): extra offers on a share of products, by other sellers
    multi = g.random(n_products) < P["multi_seller_product_share"]
    extra = np.flatnonzero(multi)
    k = 1 + (g.random(len(extra)) < 0.3)                      # 1 or 2 extra sellers
    rep = np.repeat(extra, k)
    offer_product.append(rep)
    offer_seller.append(g.choice(n_sellers, size=len(rep), p=seller_weight / seller_weight.sum()))
    offer_price.append(np.round(price[rep] * g.lognormal(0, 0.08, len(rep)), 2))
    offer_freight.append(freight[rep])

    # constructed focal competition pair for scenario S5 (documented as constructed, not observed)
    order_ = np.argsort(-seller_weight)
    seller_a, seller_b = int(order_[2]), int(order_[3])
    a_products = np.flatnonzero(owner == seller_a)
    a_top = a_products[np.argsort(-product_pop[a_products])][:10]
    offer_product.append(a_top)
    offer_seller.append(np.full(len(a_top), seller_b))
    offer_price.append(np.round(price[a_top] * g.lognormal(0, 0.03, len(a_top)), 2))
    offer_freight.append(freight[a_top])

    op = np.concatenate(offer_product)
    os_ = np.concatenate(offer_seller)
    keep = np.ones(len(op), bool)                               # drop accidental self-competition duplicates
    _, first = np.unique(np.stack([op, os_]), axis=1, return_index=True)
    keep[:] = False
    keep[first] = True
    op, os_ = op[keep], os_[keep]
    oprice = np.concatenate(offer_price)[keep]
    ofr = np.concatenate(offer_freight)[keep]
    opop = product_pop[op] * seller_weight[os_] / seller_weight[os_].mean()

    # expected units/day per offer: category volume split by within-category popularity
    cat_of = product_category[op]
    cat_tot = np.bincount(cat_of, weights=opop, minlength=len(cats))
    demand = daily_orders * share[cat_of] * opop / cat_tot[cat_of]

    n_buyers = max(1000, int(daily_orders * 120))
    buyer_region = _choice(g, P["buyer_regions"], n_buyers)
    buyer_device = np.where(g.random(n_buyers) < A["mobile_share"], "mobile", "desktop")

    return World(seed=seed, scale=scale, params=P,
                 seller_region=seller_region, seller_weight=seller_weight, seller_handling_mu=seller_handling_mu,
                 product_category=product_category, product_defect=defect,
                 offer_product=op, offer_seller=os_, offer_price=oprice, offer_freight=ofr,
                 offer_popularity=opop, offer_daily_demand=demand,
                 buyer_region=buyer_region, buyer_device=buyer_device,
                 categories=names, category_share=share,
                 aliases={"@seller_A": f"s{seller_a}", "@seller_B": f"s{seller_b}"},
                 shared_products=a_top)
