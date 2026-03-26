#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/archive_audit.sh
  ./scripts/archive_audit.sh --summary-only

Description:
  Standard entrypoint for archive governance checks.
  Runs the semi-automated archive audit defined in tools/archive_audit.py.
EOF
  exit 0
fi

python3 tools/archive_audit.py "$@"
