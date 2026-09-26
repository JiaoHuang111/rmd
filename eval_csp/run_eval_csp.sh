#!/bin/bash
# CSP benchmark driver: GT-mask init -> generation -> official postprocess -> metrics.
# =====================================================================================
# The evaluation protocol has three mandatory steps. Skipping any one of them makes the
# reported match rate meaningless (e.g. candidates that never went through the official
# postprocess score MR = 0):
#
#   1. generate_gtmask.py         fixed-length init built from the GT mask
#   2. postprocess_candidates.py  the official postprocess, mandatory
#   3. csp_metrics.py             StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
#
# Metrics are reported for both match rates against two GT sets (`gt_prep` = the
# preprocessed corpus the model was trained on, `orig` = the raw CIF corpus) and for
# --num-gens 1 and 20 (top-1 and top-20 selection). A raw-vs-gt_prep run, i.e. metrics
# computed on candidates *before* postprocess, is also recorded for reference.
#
# Usage:
#   ./run_eval_csp.sh --ckpt CKPT [--ckpt CKPT ...] [options]
#
# Examples:
#   # one checkpoint, 20 prompts x 20 shots, on the default dataset (mp_20)
#   ./run_eval_csp.sh --ckpt /path/to/last.ckpt
#
#   # two checkpoints, tagged, on a chosen GPU
#   ./run_eval_csp.sh --ckpt /path/a.ckpt --name cspv3last \
#                     --ckpt /path/b.ckpt --name cspv3best --device cuda:3
#
#   # quick smoke run: 5 prompts, 2 shots
#   ./run_eval_csp.sh --ckpt /path/last.ckpt --limit 5 --shots 2
#
# Interpreters (two environments are used because pymatgen/smact are evaluation-only
# dependencies and may live in a separate env):
#   DMLM_PY   interpreter that can import byprot + torch   (default: python3)
#   EVAL_PY   interpreter that can import pymatgen + smact  (default: $DMLM_PY)
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

DMLM_PY=${DMLM_PY:-python3}
EVAL_PY=${EVAL_PY:-$DMLM_PY}

DS=mp_20
DEVICE=cuda
SHOTS=20
MAX_ITER=500
BATCH_SIZE=8
SEED=1337
LIMIT=0
IDS_JSON=
SUMMARY=eval_csp/eval_summary.txt
GT_SETS="gt_prep orig"
NUM_GENS="1 20"
DO_RAW=1
CKPTS=()
NAMES=()

usage () { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt)       CKPTS+=("$2"); shift 2 ;;
        --name)       NAMES+=("$2"); shift 2 ;;
        --ds)         DS=$2; shift 2 ;;
        --device)     DEVICE=$2; shift 2 ;;
        --shots)      SHOTS=$2; shift 2 ;;
        --max-iter)   MAX_ITER=$2; shift 2 ;;
        --batch-size) BATCH_SIZE=$2; shift 2 ;;
        --seed)       SEED=$2; shift 2 ;;
        --limit)      LIMIT=$2; shift 2 ;;
        --ids-json)   IDS_JSON=$2; shift 2 ;;
        --summary)    SUMMARY=$2; shift 2 ;;
        --gt-sets)    GT_SETS=$2; shift 2 ;;
        --num-gens)   NUM_GENS=$2; shift 2 ;;
        --no-raw)     DO_RAW=0; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[ ${#CKPTS[@]} -gt 0 ] || { echo "error: at least one --ckpt is required" >&2; usage 1; }
[ ${#NAMES[@]} -eq 0 ] && for c in "${CKPTS[@]}"; do NAMES+=("$(basename "$(dirname "$c")")"); done
[ ${#NAMES[@]} -eq ${#CKPTS[@]} ] || { echo "error: --name must be given once per --ckpt (or not at all)" >&2; exit 1; }

D=eval_csp/data/$DS
[ -n "$IDS_JSON" ] || IDS_JSON=$D/eval_ids20.json
mkdir -p "$(dirname "$SUMMARY")"

report () {  # report <json> <label>
    "$EVAL_PY" -c "
import json
d = json.load(open('$1'))
keys = ('n_prompts', 'num_gens', 'match_rate', 'mean_rmsd_matched', 'n_matched')
print('$2', {k: (round(d[k], 4) if isinstance(d[k], float) else d[k]) for k in keys if k in d})
" >> "$SUMMARY"
}

run_ckpt () {  # run_ckpt <name> <ckpt>
    local name=$1 ckpt=$2
    local raw=$D/gen20_$name post=$D/gen20_${name}_post log=eval_csp/gen20_$name.log
    echo "=== $name  start $(date +%F' '%T)  ckpt=$ckpt" >> "$SUMMARY"

    # 1. Generate with a fixed-length GT-mask init.
    local extra=()
    [ "$LIMIT" -gt 0 ] && extra+=( --limit "$LIMIT" )
    [ -n "$IDS_JSON" ] && [ -f "$IDS_JSON" ] && extra+=( --ids-json "$IDS_JSON" )
    "$DMLM_PY" eval_csp/generate_gtmask.py --ckpt "$ckpt" --ds "$DS" \
        --outdir "$raw" --shots "$SHOTS" --max-iter "$MAX_ITER" \
        --batch-size "$BATCH_SIZE" --seed "$SEED" --device "$DEVICE" \
        "${extra[@]}" > "$log" 2>&1
    local rc=$?
    local n
    n=$(ls "$raw"/*.cif 2>/dev/null | wc -l)
    echo "$name  gen rc=$rc  candidates=$n  last: $(tail -1 "$log")" >> "$SUMMARY"
    if [ "$n" -eq 0 ]; then
        echo "$name  SKIP metrics (no candidates)" >> "$SUMMARY"
        return
    fi

    # 2. Official postprocess -- mandatory before matching.
    "$EVAL_PY" eval_csp/postprocess_candidates.py "$raw" "$post" >> "$log" 2>&1
    echo "$name  post: $(tail -1 "$log")" >> "$SUMMARY"

    # 3. Metrics on postprocessed candidates, against each GT set and num-gens.
    local gt ng
    for gt in $GT_SETS; do
        for ng in $NUM_GENS; do
            "$EVAL_PY" eval_csp/csp_metrics.py "$post" "$D/$gt" --num-gens "$ng" \
                --out-json "$D/m20_${name}_post_${gt}_${ng}.json" \
                --detail-json "$D/md20_${name}_post_${gt}_${ng}.json" > /dev/null 2>&1
            report "$D/m20_${name}_post_${gt}_${ng}.json" "$name  post vs $gt ng=$ng"
        done
    done

    # Reference: candidates before postprocess (expected to score much lower).
    if [ "$DO_RAW" -eq 1 ]; then
        for ng in $NUM_GENS; do
            "$EVAL_PY" eval_csp/csp_metrics.py "$raw" "$D/gt_prep" --num-gens "$ng" \
                --out-json "$D/m20_${name}_raw_gtprep_${ng}.json" > /dev/null 2>&1
            report "$D/m20_${name}_raw_gtprep_${ng}.json" "$name  raw (not postprocessed) vs gt_prep ng=$ng"
        done
    fi

    echo "=== $name  done $(date +%F' '%T)" >> "$SUMMARY"
}

for i in "${!CKPTS[@]}"; do
    run_ckpt "${NAMES[$i]}" "${CKPTS[$i]}"
done
echo "ALL DONE $(date +%F' '%T)" >> "$SUMMARY"
echo "summary written to $SUMMARY"
