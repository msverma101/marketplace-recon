#!/usr/bin/env bash
# Run the video-2 scenario and watch it live, without recording anything.
#
#   scripts/uc2_watch.sh            # watchable speed on the last normal day and the fault day (~8 min)
#   scripts/uc2_watch.sh --fast     # full speed (~6 min)
# Opens: the agent dashboard (http://127.0.0.1:8765/scenes/uc2_live.html) and, if the shop was started with
# `scripts/uc2_shop.sh up --traces`, Jaeger (http://localhost:8080/jaeger/ui/) to watch requests inside the services.
# To watch the containers themselves, in another terminal:
#   docker logs -f --since 1s payment       # payment service log: failed charges show up here during the fault
#   docker logs -f --since 1s checkout      # checkout service log
#   docker stats                            # live CPU / memory per container
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
PY="${PY:-$HOME/.venvs/mkt-emulator/bin/python}"
PORT="${PORT:-8765}"
ARGS=(--paced-days "${PACED_DAYS:-2018-02-15,2018-02-16}" --pace "${PACE:-1}" --event-pause "${EVENT_PAUSE:-7}")
[ "${1:-}" = "--fast" ] && ARGS=()
curl -sf -o /dev/null http://localhost:8080/ || { echo "The shop is not running: scripts/uc2_shop.sh up [--traces]"; exit 1; }
python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 & SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 1
rm -f outputs/uc2/live.json
if [ -z "${NO_OPEN:-}" ]; then
  open "http://127.0.0.1:$PORT/scenes/uc2_live.html"
  if curl -sf -o /dev/null http://localhost:8080/jaeger/ui/; then open "http://localhost:8080/jaeger/ui/search?service=payment"; fi
fi
echo "Dashboard: http://127.0.0.1:$PORT/scenes/uc2_live.html   shop: http://localhost:8080   traces: http://localhost:8080/jaeger/ui/ (if started with --traces)"
"$PY" -u -m agent.uc2 "${ARGS[@]}"
echo "Done. Outputs in outputs/uc2/. The dashboard keeps showing the final state until you close this script (Ctrl-C)."
[ -t 0 ] && read -r -p "Press Enter to stop the dashboard server. " _ || true
