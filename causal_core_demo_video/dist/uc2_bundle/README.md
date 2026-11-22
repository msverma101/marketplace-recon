# Video 2: marketplace watcher on the Astronomy Shop (runnable bundle)

An autonomous agent watches a marketplace that runs through the real OpenTelemetry Astronomy Shop. Shoppers come from
an emulator fitted on real Olist orders. A payment fault is injected; the agent detects it from telemetry, rules out a
demand drop, proposes a fix with an expected range, and applies it only after approval, only to the local shop's flags.

Needs: macOS with Docker Desktop (4 GB+ memory for the shop), Python 3.11+, Node 22+, ffmpeg, Google Chrome, ~6 GB disk.

```bash
./setup.sh                                   # once: Python venv, OpenTelemetry demo @7a88536, data folders
cd demo
scripts/uc2_shop.sh up --traces              # start the shop + Jaeger (first time pulls ~3-5 GB of images)
scripts/uc2_engine.sh start                  # the causal engine as its own local service (watch/record also start it)
scripts/uc2_watch.sh                         # WATCH IT LIVE (no recording): one page, side by side: the agent
                                             #   dashboard | Jaeger traces, failed requests, Monitor rates, the shop
scripts/uc2_record.sh --open                 # optional: run at watchable speed and record the dashboard → videos/uc2_screen.mp4
scripts/uc2_assemble.sh                      # silent cut with subtitles → videos/uc2_silent.mp4 (≤ 3 min)
scripts/uc2_voice.sh                         # plays the silent cut and records your voice → videos/uc2_voice.m4a
scripts/uc2_assemble.sh videos/uc2_voice.m4a # final video with your voice → videos/uc2.mp4
scripts/uc2_shop.sh down                     # stop the shop
```

Watching what happens: the agent dashboard (left: shoppers going through the shop, what the agent sees, its console;
right: each engine step with its measured time, the policy decision, the action card). In Jaeger, pick service
"checkout" or "payment" to see real requests flow through the services; during the fault, failed payments show as
errors. Container logs: `docker logs -f --since 1s payment`. Live resource use: `docker stats`.

What is real and what is simulated: the shop, its services and their telemetry are real (local containers); shoppers
are simulated by the emulator (fitted on 2017 Olist orders); the payment fault is injected by the scenario; approvals
are scripted ("on-call marketplace lead"). Nothing outside this machine is changed.

Licences: Olist-derived files under marketplace-recon/artifacts are CC BY-NC-SA 4.0. marketplace-recon is a private
repository; this copy is for internal use with its owner's agreement.
