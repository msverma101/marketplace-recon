"""Internal-only TabPFN evaluation against the NumPy methods (v3.2 §6.4; user decision 2026-11-22; CP-037).

Runs OUTSIDE the core: TabPFN is never imported by causal_core/. Use the isolated environment:

    ~/.venvs/causal-core-tabpfn/bin/python -m scripts.tabpfn_eval            # from causal_core/
    (one-time: accept the licence at https://ux.priorlabs.ai and export TABPFN_TOKEN=<your API key>)

Two questions, both on emulator worlds with known truth:
  A. covariate baseline prediction (an allowed use, §6.4): how much day-to-day variance of a monitored metric does each
     model explain on held-out days? Less unexplained variance = more power for the learned baseline (CP-030).
  B. equation shape (EXPLORATORY, outside §6.4's allowed uses): for spend → visitors in a curved world, do held-out
     90% intervals cover, and how close is the predicted effect of spend 100 → 130 to the truth?
Nothing here feeds a decision path. Results print to the terminal and docs/tabpfn/eval.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from emulator import dynamics_world as D
from evaluation.fixture_io import PROJECT_ROOT


def tabpfn_regressor():
    """(regressor or None, reason)."""
    try:
        from tabpfn import TabPFNRegressor  # noqa: F401
    except ImportError:
        return None, "tabpfn not installed in this environment (use ~/.venvs/causal-core-tabpfn/bin/python)"
    if not os.environ.get("TABPFN_TOKEN") and not os.path.exists(os.path.expanduser("~/.cache/tabpfn")):
        return None, "licence not accepted yet: accept at https://ux.priorlabs.ai and export TABPFN_TOKEN"
    from tabpfn import TabPFNRegressor
    return (lambda: TabPFNRegressor(device="cpu", random_state=0)), "ok"


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    s2 = float(((y - X1 @ b) ** 2).sum() / (len(y) - X1.shape[1]))
    return b, s2, np.linalg.inv(X1.T @ X1)


def question_a(seeds=range(10), make=None) -> dict:
    """Daily metric = weekday effect + saturating effect of an exogenous driver + noise. Train 120 days, test 60."""
    out = {"linear_calendar": [], "linear_calendar_plus_driver": [], "tabpfn_calendar_plus_driver": []}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        n = 180
        wd = np.arange(n) % 7
        driver = rng.uniform(0, 10, n)
        y = 100 + np.array([3, 2, 0, -1, -1, -2, -1])[wd] + 12 * (1 - np.exp(-driver / 2)) + rng.normal(0, 1.5, n)
        cal = np.eye(7)[wd][:, 1:]
        tr, te = slice(0, 120), slice(120, 180)
        for name, X in (("linear_calendar", cal), ("linear_calendar_plus_driver", np.column_stack([cal, driver]))):
            b, _, _ = ols(X[tr], y[tr])
            pred = np.column_stack([np.ones(60), X[te]]) @ b
            out[name].append(float(np.var(y[te] - pred)))
        if make is not None:
            m = make()
            Xf = np.column_stack([wd, driver])
            m.fit(Xf[tr], y[tr])
            out["tabpfn_calendar_plus_driver"].append(float(np.var(y[te] - m.predict(Xf[te]))))
    base = np.mean(out["linear_calendar"])
    return {k: {"residual_variance": float(np.mean(v)), "explained_vs_calendar": float(1 - np.mean(v) / base)}
            if v else {"residual_variance": None, "explained_vs_calendar": None} for k, v in out.items()} | {"noise_variance": 1.5 ** 2}


def question_b(seeds=range(200, 215), make=None) -> dict:
    """spend → visitors in the curved world. Held-out 90% coverage and the 100 → 130 effect vs truth."""
    rows = {"linear": [], "tabpfn": []}
    for sd in seeds:
        w = D.build(seed=sd, family="saturating")
        ts = sorted(w.series["spend"])
        s = np.array([w.series["spend"][t] for t in ts]); v = np.array([w.series["visitors"][t] for t in ts])
        tr, te = slice(0, 120), slice(120, 200)
        truth = float(D.visitors_mean([130.0], "saturating")[0] - D.visitors_mean([100.0], "saturating")[0])
        b, s2, G = ols(s[tr, None], v[tr])
        X1 = np.column_stack([np.ones(80), s[te]])
        sd_pred = np.sqrt(s2 + np.einsum("ij,jk,ik->i", X1, G, X1))
        cov = float(np.mean(np.abs(v[te] - X1 @ b) <= 1.645 * sd_pred))
        se_eff = 30 * float(np.sqrt(s2 * G[1, 1]))
        rows["linear"].append({"coverage": cov, "effect": 30 * b[1], "covers_effect": abs(30 * b[1] - truth) <= 1.645 * se_eff,
                               "truth": truth})
        if make is not None:
            m = make()
            m.fit(s[tr, None], v[tr])
            q = m.predict(s[te, None], output_type="quantiles", quantiles=[0.05, 0.5, 0.95])
            cov_t = float(np.mean((v[te] >= q[0]) & (v[te] <= q[2])))
            med = m.predict(np.array([[100.0], [130.0]]))
            rows["tabpfn"].append({"coverage": cov_t, "effect": float(med[1] - med[0]), "covers_effect": None, "truth": truth})
    def summ(r):
        if not r:
            return None
        return {"holdout_coverage_at_90": float(np.mean([x["coverage"] for x in r])),
                "effect_mean_abs_error": float(np.mean([abs(x["effect"] - x["truth"]) for x in r])),
                "effect_interval_covers_truth": (float(np.mean([x["covers_effect"] for x in r]))
                                                 if r[0]["covers_effect"] is not None else "TabPFN gives no effect interval")}
    return {k: summ(v) for k, v in rows.items()} | {"truth_effect_100_to_130": rows["linear"][0]["truth"]}


def main() -> int:
    make, why = tabpfn_regressor()
    print(f"TabPFN: {why}")
    t = time.time()
    a = question_a(make=make)
    b = question_b(make=make)
    print("\nA. covariate baseline (allowed use): unexplained variance on 60 held-out days, 10 worlds (noise floor "
          f"{a['noise_variance']:.2f})")
    for k in ("linear_calendar", "linear_calendar_plus_driver", "tabpfn_calendar_plus_driver"):
        x = a[k]
        print(f"  {k:<30} " + ("not run" if x["residual_variance"] is None else
                                f"{x['residual_variance']:6.2f}  ({x['explained_vs_calendar']:+.0%} vs calendar only)"))
    print("\nB. equation shape (exploratory): spend → visitors in curved worlds, 15 seeds")
    for k in ("linear", "tabpfn"):
        x = b[k]
        print(f"  {k:<8} " + ("not run" if x is None else
                             f"held-out 90% coverage {x['holdout_coverage_at_90']:.2f}; effect 100→130 mean abs error "
                             f"{x['effect_mean_abs_error']:.1f} (truth {b['truth_effect_100_to_130']:.1f}); effect interval "
                             f"covers truth: {x['effect_interval_covers_truth']}"))
    out = PROJECT_ROOT / "docs" / "tabpfn"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps({"tabpfn": why, "A_covariate_baseline": a, "B_equation_shape": b,
                                               "seconds": round(time.time() - t, 1)}, indent=2) + "\n")
    print(f"\nwritten docs/tabpfn/eval.json ({time.time() - t:.1f} s)")
    return 0 if make is not None else 3


if __name__ == "__main__":
    sys.exit(main())
