# Marketplace Emulator × Causal Core

**An autonomous monitoring agent that investigates a realistic, data-grounded marketplace, and knows when it can't tell.**

A marketplace emulator calibrated on real e-commerce data (Olist, Brazil) and stitched with 18 public datasets. It is wired to a domain-agnostic **causal reasoning core** that detects changes, separates look-alike causes, asks for the evidence it needs, and abstains when the data can't decide. A real microservice shop, the [OpenTelemetry Astronomy Shop](https://github.com/open-telemetry/opentelemetry-demo), can serve the emulator's shoppers, so faults and telemetry are real.

---

## 🎬 Demo video

<!-- Replace the placeholder below with your video. Options:
     - YouTube/Vimeo:  [![Demo](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
     - File in repo:   put it under docs/media/demo.mp4 and link it: [Watch the demo](docs/media/demo.mp4)
     - GitHub upload:  drag the .mp4 into this README in the GitHub editor; GitHub inserts the player link. -->

> **📺 Demo video: coming soon**
>
> _Add the video link here._

**What the demo shows (≈ 5 min):**
1. **A normal day:** the emulator produces orders, sessions, deliveries and reviews; the real shop serves the same shoppers.
2. **A hidden incident:** a sealed scenario is injected (a logging failure *or* a real demand drop; or a payment fault in the shop).
3. **Detection and caveats:** the core raises an alarm and says what it *can't* yet tell.
4. **The core asks for evidence:** it requests a discriminating measurement; a policy gate (not the core) grants it; the core revises its answer.
5. **The reveal:** the sealed answer is opened and the core's verdict is scored.

---

## Why it's different
| Usual monitoring | This project |
|---|---|
| "Orders dropped 40%" | "Orders dropped **and** payments didn't, so the identity broke: **a measurement problem**, not lost demand" |
| Thresholds on one chart | Walks the declared funnel (visits → carts → checkouts → orders) to **locate** where a change starts |
| Always answers | **Abstains** and asks for the cheapest evidence that would decide |
| Trained on its own data | **Ground truth is sealed** outside the core; evidence carries honest provenance labels |

---

## How it works
```
 19 public datasets ──► stitch cards ──► EMULATOR ──────────────────► Astronomy Shop (real services, optional)
 (Olist + 18 others)   (learned shapes,   order chain · sessions ·        catalog sync · shopper sessions ·
                        held-out checks)   observation layer · twin         fault flags · redacted telemetry
                                               │ read-only contract port            │
                                               ▼                                    ▼
                                     CAUSAL CORE (causal_core/) ◄──────────── telemetry port
                                     detect → diagnose → ask for evidence → revise
                                               │
                                     sealed answers + scoring (privileged; the core never sees them)
```
- **Real levels, borrowed shapes.** Order volumes, delivery times and review behaviour come from Olist; each borrowed pattern (e.g. Expedia's click-by-position curve) has a **transport card** with a held-out check and a status.
- **One contract.** Everything the core reads is a `causal_core.contracts` object (schema 2.0): typed payloads, UTC clocks, provenance and roles.

---

## Quick start
**Requirements:** Python ≥ 3.10, ~1 GB disk. Docker only for the optional shop.

```bash
git clone https://github.com/msverma101/marketplace-recon && cd marketplace-recon
python3 -m venv .venv
.venv/bin/python -m pip install duckdb polars "pydantic>=2.7" numpy scipy pyarrow fastexcel
echo "$PWD/causal_core" > "$(.venv/bin/python -c 'import site;print(site.getsitepackages()[0])')/causal_core_local.pth"
mkdir -p ~/data/mkt/stitch && unzip -q share/stitch_samples.zip -d ~/data/mkt/stitch/

.venv/bin/python -m pytest -q          # ~190 tests, ~75 s
```
`artifacts/` (processed Olist tables) ships with the repo, so the emulator runs without downloading anything.

### Run the showcase pieces
| What | Command | Time |
|---|---|---|
| Real causal core on the emulator (logging drop vs demand drop vs no fault) | `.venv/bin/python -I scripts/core_on_emulator.py` | ~2 min |
| Closed loop: core asks for evidence → policy grants it → core revises, scored vs sealed answer | `.venv/bin/python -I scripts/mvs_headless.py 7 8 9` | ~1 min |
| Contract check against the core's contracts (+ the core's own parser over 340 samples) | `.venv/bin/python -I scripts/conform.py` | ~70 s |
| Contract samples for all 106 variables | `.venv/bin/python -I scripts/stitch_samples.py` | ~35 s |
| Re-stitch every dataset into cards (needs the raw datasets) | `.venv/bin/python -I scripts/stitch.py run` | ~80 s |

### Optional: the live shop
```bash
git clone --branch 3.1.0 --depth 1 https://github.com/open-telemetry/opentelemetry-demo.git ~/.cache/otel-demo-3.1.0
mkdir -p ~/data/mkt/astronomy/flags ~/data/mkt/astronomy/live && chmod 777 ~/data/mkt/astronomy/live
cp ~/.cache/otel-demo-3.1.0/src/flagd/demo.flagd.json ~/data/mkt/astronomy/flags/
cd ~/.cache/otel-demo-3.1.0 && DEMO_VERSION=3.1.0 OTEL_COLLECTOR_CONFIG_EXTRAS=$HOME/marketplace-recon/deploy/m0/otelcol-redact.yml \
  docker compose -p m0 -f compose.yaml -f $HOME/marketplace-recon/deploy/emulator/compose.emulator.yaml up -d --no-deps \
  flagd otel-collector valkey-cart astronomy-db product-catalog cart currency payment shipping quote email ad recommendation checkout image-provider frontend frontend-proxy
cd ~/marketplace-recon
.venv/bin/python -I scripts/storefront_sessions.py --browse 300 --buy 100   # shoppers through the real shop
.venv/bin/python -I scripts/storefront_s5.py                                # payment-fault scenario + leak scan
docker compose -p m0 stop                                                   # when done
```
The shop UI is at **http://localhost:8080** while it runs.

---

## Results so far
| Check | Result |
|---|---|
| Realism vs held-out real data (Olist 2018) | 9/10 checks pass (the one miss, a weekday mix shift, is documented and accepted) |
| Session layer reproduces its source datasets | 8/8 fidelity checks |
| Session layer helps locate causes (value gate S10) | checkout fault 5/5 · demand drop 4/5 · no false alarms 5/5 (without it: 0/5) |
| Closed loop (ask → grant → revise) | 6/6 seeds correct, bit-identical reruns |
| Real core separates logging drop from demand drop (payments-per-order ratio) | correct on all completed seeds |
| Live shop | 400/400 sessions OK; payment fault caught in telemetry, 0 leaked fault details |
| Contracts | 340/340 sample observations accepted by the core's own parser |

---

## Repository map
| Path | What's there |
|---|---|
| `mkt/sim/` | emulator: order chain, session layer, observation layer, counterfactual twin |
| `mkt/emulator/`, `mkt/harness/` | emulator API (step / intervene / snapshot), closed loop, sealed answers |
| `mkt/core_port.py`, `mkt/taxonomy.py`, `mkt/variable_catalog.py` | the contract port and the 106-variable catalogue |
| `mkt/stitch/`, `scripts/stitch*.py` | dataset stitching (33 data types) and contract samples |
| `mkt/adapters/astronomy/`, `deploy/` | Astronomy Shop integration (overrides only; upstream untouched) |
| `causal_core/` | the causal reasoning core (Subsystem 2): graph, reasoning engine, its own contracts |
| `docs/` | architecture, gates, reports, diagrams (`*.excalidraw`), `HANDOVER.md` (what's done / not done) |

---

## Technical documentation
**Start here**
- [`docs/HANDOVER.md`](docs/HANDOVER.md): what's done, what's not, setup, commands (two pages)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the full design (taxonomy L0–L10, stitching §7.5, transport cards §7.6, gates S1–S10 §12.5, phased plan §13)
- Diagrams (open in [excalidraw.com](https://excalidraw.com)): [`docs/architecture-and-stitching.excalidraw`](docs/architecture-and-stitching.excalidraw) · [`docs/current-architecture.excalidraw`](docs/current-architecture.excalidraw) · [`docs/stitch-pipeline.excalidraw`](docs/stitch-pipeline.excalidraw)

**Causal core**
- [`docs/master_implementation_blueprint.md`](docs/master_implementation_blueprint.md): Subsystem 2 specification (ports, tiers T0–T4, demo portfolio §10)
- [`causal_core/README.md`](causal_core/README.md) and [`causal_core/docs/`](causal_core/docs/): the core's phases, demos and audits
- [`docs/CONTRACT_SUGGESTIONS.md`](docs/CONTRACT_SUGGESTIONS.md): proposed contract additions (also GitHub issue #1)

**Data and stitching**
- [`docs/datasets/AUDIT.md`](docs/datasets/AUDIT.md): what every dataset contains, verified claims, what was dropped and why
- [`docs/stitch/README.md`](docs/stitch/README.md): where the stitched data lives and how it's built
- [`docs/stitch/STITCH_REPORT.md`](docs/stitch/STITCH_REPORT.md): all 33 transport cards, held-out checks, status
- [`docs/p0/VARIABLE_CATALOG.md`](docs/p0/VARIABLE_CATALOG.md) · [`docs/p0/VARIABLE_TAXONOMY.md`](docs/p0/VARIABLE_TAXONOMY.md): the 106 variables and the served taxonomy
- [`docs/stitch/SAMPLES_REPORT.md`](docs/stitch/SAMPLES_REPORT.md): contract samples per variable

**Emulator build and gates** (each phase: design and tolerances first, then results)
- [`PHASE1_SOURCE_SEMANTICS`](docs/p0/PHASE1_SOURCE_SEMANTICS.md) · [`PHASE2_REAL_INTEGRATION`](docs/p0/PHASE2_REAL_INTEGRATION.md) · [`PHASE3_TOLERANCES`](docs/p0/PHASE3_TOLERANCES.md) · [`PHASE3_MINIMAL_CHAIN`](docs/p0/PHASE3_MINIMAL_CHAIN.md) · [`PHASE4_OBSERVATION`](docs/p0/PHASE4_OBSERVATION.md) · [`PHASE5_INTERVENTION`](docs/p0/PHASE5_INTERVENTION.md)
- Session layer: [`P1_2_SESSION_LAYER_PREREG`](docs/p0/P1_2_SESSION_LAYER_PREREG.md) → [`P1_2_SESSION_LAYER`](docs/p0/P1_2_SESSION_LAYER.md) · [`P1_2_S10_V2_PREREG`](docs/p0/P1_2_S10_V2_PREREG.md)
- Decisions taken along the way, with reasons and how to undo them: [`docs/p0/STRATEGY_CHANGES.md`](docs/p0/STRATEGY_CHANGES.md)

**Contracts and integration**
- [`docs/p0/CONFORMANCE_AND_M1_M5_DESIGN.md`](docs/p0/CONFORMANCE_AND_M1_M5_DESIGN.md): contract conformance and integration design
- [`docs/CONFORMANCE_REPORT.md`](docs/CONFORMANCE_REPORT.md): latest field-by-field contract diff and checks (`scripts/conform.py`)
- [`docs/ASTRONOMY_SHOP_INTEGRATION.md`](docs/ASTRONOMY_SHOP_INTEGRATION.md): how the shop connects (projection, transactions, telemetry, redaction)
- [`docs/experiments/M0_REPORT.md`](docs/experiments/M0_REPORT.md): shop feasibility experiments (throughput, leaks, catalog scale, idempotency)
- [`docs/p0/INTEGRATION_M1_M5.md`](docs/p0/INTEGRATION_M1_M5.md): emulator API, closed loop, storefront and telemetry results

---

## Data and licences
- **Olist Brazilian E-Commerce** (Kaggle): CC BY-NC-SA 4.0. The processed tables in `artifacts/` derive from it; non-commercial use with attribution.
- The other datasets (Retailrocket, Taobao, Expedia, Google Analytics sample, Statcounter, M5, ESCI, Open Bandit, Criteo, Hillstrom, Instacart, UCI Online Retail II, Coat, YelpChi, Amazon fraud graph, NAB/LogHub, Banco Central do Brasil) are **not redistributed**. `scripts/datasets.py` downloads them under each source's own terms; Kaggle and AIcrowd need your own accounts.
- The Astronomy Shop is Apache-2.0 and used unmodified (release 3.1.0).

## Status and next steps
Working: everything in *Results so far*. Next: fix the core's default monitoring window, make its baseline robust to marketplace growth, and connect the remaining 18 stitched data types (prices, stock, search, recommendations, returns, reviews, sellers, ads, fraud) as simulator layers. Details: [`docs/HANDOVER.md`](docs/HANDOVER.md).

## Team
<!-- Add names, roles and links here. -->
_Add team members here._
