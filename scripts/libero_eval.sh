#!/usr/bin/env bash
# Evaluate pi05_libero on LIBERO suites: our policy server (scripts/serve.py, a
# drop-in for openpi's serve_policy.py) and the stock openpi LIBERO client.
# One policy server; one client per suite, run in parallel. The simulator is
# lock-step (it waits for each action), so server latency does not affect success.
# Usage: scripts/libero_eval.sh <out_dir> <trials_per_task> <suite>...
#   VARIANT=<pi05_accel variant> COMPILE=<mode|none> PORT=<port> override the defaults;
#   RECORD=<dir> saves every observation the server receives; NUM_STEPS=<n> sets denoising steps;
#   SEED=<s> seeds the server's observation-derived noise (default 0).
# Environment: OPENPI_DIR (openpi checkout), OPENPI_PYTHON (its Python, default python),
#   LIBERO_PYTHON (Python of the LIBERO client environment).
set -euo pipefail
OPENPI=${OPENPI_DIR:?set OPENPI_DIR to the openpi checkout}
LIBERO_PY=${LIBERO_PYTHON:?set LIBERO_PYTHON to the python of the LIBERO client environment}
OUT=$(realpath -m "$1"); TRIALS=$2; shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
VARIANT=${VARIANT:-stock}; COMPILE=${COMPILE:-max-autotune}; PORT=${PORT:-18000}
RECORD=${RECORD:+$(realpath -m "$RECORD")}  # resolve before the cd below
mkdir -p "$OUT"

cd "$OPENPI"
setsid "${OPENPI_PYTHON:-python}" "$REPO/scripts/serve.py" --variant "$VARIANT" --compile "$COMPILE" --port "$PORT" --num-steps "${NUM_STEPS:-10}" --seed "${SEED:-0}" ${RECORD:+--record "$RECORD"} \
    > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill -- -$SERVER 2>/dev/null || true' EXIT
until (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; do
    kill -0 $SERVER 2>/dev/null || { echo "server died, see $OUT/server.log"; exit 1; }
    sleep 2
done

pids=()
for SUITE in "$@"; do
    mkdir -p "$OUT/$SUITE"
    # First LIBERO import asks where datasets live; "n" keeps the package defaults.
    echo n | MUJOCO_GL=egl PYTHONPATH=$OPENPI/third_party/libero "$LIBERO_PY" examples/libero/main.py \
        --args.port "$PORT" --args.task-suite-name "$SUITE" --args.num-trials-per-task "$TRIALS" \
        --args.video-out-path "$OUT/$SUITE/videos" > "$OUT/$SUITE/client.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done  # fail if any client fails
for SUITE in "$@"; do
    echo "$SUITE: $(grep "# successes" "$OUT/$SUITE/client.log" | tail -1 | sed 's/.*# successes: //')"
done
