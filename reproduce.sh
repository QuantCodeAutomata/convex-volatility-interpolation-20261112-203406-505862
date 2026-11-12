#!/usr/bin/env bash
# Runs the test suite and every experiment of the Convex Volatility
# Interpolation replication, using the same commands as README.md.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
