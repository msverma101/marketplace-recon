# Decisions and strategy changes made overnight (for your review)

Each entry: what, why (evidence), alternatives, how to undo.

| # | When | Decision | Why | Alternatives considered | Undo |
|---|---|---|---|---|---|
| 1 | phase 3 | Fix attempt 1: locality, destination-specific fulfilment, 7 lateness bins | F8, F10 failed; all fitted on 2017 | none needed (mechanism refinement within plan) | revert commit 79cf74f's chain changes |
| 2 | phase 3 | Fix attempt 2: 2017 transit drift | F10 gap failed; diagnosed a Feb–Mar 2018 disruption not in 2017; drift is in 2017 data | fit on 2018 (rejected: breaks the held-out test) | same |
| 3 | phase 3 | **Did not use attempt 3**; accepted S5 as preregistered (seed 1) and reported fragility (seeds 2–5: 8–9/10) | remaining biases are only visible in held-out data | tune F2/F6 (rejected: would be fitting on 2018) | — |
| 4 | phase 4 | Real-data and emulator ports share one query engine (`EventFramePort`) | phases 2 and 4 must serve the same contract | separate ports (rejected: contract drift) | — |
