#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/totalsegmentator-2.16.0" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS_ENV_PATH="$1"
uv venv --python 3.11 "$TS_ENV_PATH"
uv pip sync \
  --python "$TS_ENV_PATH/bin/python" \
  "$PROJECT_DIR/requirements/totalsegmentator-2.16.0-cu124.txt"
"$TS_ENV_PATH/bin/python" - <<'PY'
from importlib.metadata import version

expected = "2.16.0"
installed = version("TotalSegmentator")
if installed != expected:
    raise SystemExit(f"expected TotalSegmentator {expected}, found {installed}")
print(f"TotalSegmentator {installed} environment ready")
PY
