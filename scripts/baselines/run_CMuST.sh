#!/usr/bin/env bash
set -euo pipefail

DATASETS=SIP bash "$(dirname "$0")/_run_one_baseline.sh" CMuST "$@"
