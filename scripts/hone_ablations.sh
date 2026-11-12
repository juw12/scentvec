#!/bin/sh
# Refinement-pass ablations.  Every run uses the same seed, data and number of
# epochs; only one switch changes.  Decisions are made on the validation split.
#
#   sh scripts/hone_ablations.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PYTHON=./.venv/bin/python
EPOCHS=8
THREADS=3
mkdir -p artifacts/logs

run() {
    name="$1"
    shift
    done_epochs=$(wc -l < "artifacts/runs/$name/log.jsonl" 2>/dev/null || echo 0)
    if [ "$done_epochs" -ge "$EPOCHS" ]; then
        echo "$name already finished, skipping"
        return 0
    fi
    "$PYTHON" -m scentvec.train --run-name "$name" --epochs "$EPOCHS" --threads "$THREADS" "$@" \
        > "artifacts/logs/$name.log" 2>&1
}

run h_full --n-layers 1 &
run h_nochem --n-layers 1 --use-chem false &
wait
echo "pair 1 done"

run h_notok --n-layers 1 --use-tokens false &
run h_nopairs --n-layers 1 --use-cross-source-pairs false &
wait
echo "pair 2 done"

run h_l0 --n-layers 0 &
run h_l2 --n-layers 2 &
wait
echo "pair 3 done"
