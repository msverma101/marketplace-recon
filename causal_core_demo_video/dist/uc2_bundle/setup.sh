#!/usr/bin/env bash
# One-time setup on a new Mac (Docker Desktop, Python 3.11+, Node 22+, ffmpeg and Google Chrome must be installed).
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
python3 -m venv "$HOME/.venvs/mkt-emulator"
"$HOME/.venvs/mkt-emulator/bin/python" -m pip install -q --index-url https://pypi.org/simple duckdb polars "pydantic>=2.7" numpy scipy pyarrow fastexcel
DEMO="$HOME/.cache/otel-demo-3.1.0"
if [ ! -d "$DEMO/.git" ]; then
  git clone -q --depth 1 https://github.com/open-telemetry/opentelemetry-demo.git "$DEMO"
fi
git -C "$DEMO" fetch -q --depth 1 origin 7a88536cf02f7a6ba5f4ee2b3a2b242a8c56c74c
git -C "$DEMO" -c advice.detachedHead=false checkout -q 7a88536cf02f7a6ba5f4ee2b3a2b242a8c56c74c
mkdir -p "$HOME/data/mkt/astronomy/flags" "$HOME/data/mkt/astronomy/live" "$HOME/data/mkt/raw/bcb"
chmod 777 "$HOME/data/mkt/astronomy/live"
cp "$DEMO/src/flagd/demo.flagd.json" "$HOME/data/mkt/astronomy/flags/demo.flagd.json"
cp "$B/demo/data/bcb_usd_brl_2017.json" "$HOME/data/mkt/raw/bcb/usd_brl.json"
chmod +x "$B"/demo/scripts/*.sh
echo "Setup done. Next: start Docker Desktop, then:"
echo "  cd $B/demo && scripts/uc2_shop.sh up --traces && scripts/uc2_watch.sh"
