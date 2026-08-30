#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "$0")/_run_one_baseline.sh" STGCN "$@"
