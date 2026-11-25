#!/bin/bash
# plot_start_scenes.sh - one bar plot per start scene, checkpoints side by side.
# Usage: bash scripts/plot_start_scenes.sh "<label1>=<tag1>" "<label2>=<tag2>" [...]
#   e.g. bash scripts/plot_start_scenes.sh "T80 ep576=T80_ep576" "T320 ep144=T320_ep144"
set -e
OUT_DIR=${OUT_DIR:-figures/start_scene_eval}
PY=${PY:-python}
declare -A TITLES=(
  [default]="Default start scene (reference)"
  [overlap_obj_nontarget]="Overlap: task object starts on the non-target receptacle"
  [overlap_nonobj_target]="Overlap: non-task object starts on the target receptacle"
  [overlap_nonobj_nontarget]="Overlap: non-task object starts on the non-target receptacle"
  [chaos]="Chaos: objects and receptacles dropped at random (overlap allowed)"
)
for SCENE in default overlap_obj_nontarget overlap_nonobj_target overlap_nonobj_nontarget chaos; do
  RUNS=()
  for spec in "$@"; do
    LABEL=${spec%%=*}; TAG=${spec#*=}
    # only completed runs (eval_report.txt is written at the very end)
    REPORT=$(ls eval_runs/${TAG}/${SCENE}/wandb/offline-run-*/glob/eval_report.txt 2>/dev/null | tail -1)
    [ -n "$REPORT" ] && RUNS+=(--run "${LABEL}=$(dirname "$REPORT")/eval_per_trial.csv")
  done
  if [ ${#RUNS[@]} -eq 0 ]; then echo "[skip] ${SCENE}: no results"; continue; fi
  "${PY}" tools/plot_start_scene_eval.py "${RUNS[@]}" --title "${TITLES[$SCENE]}" --out "${OUT_DIR}/${SCENE}.png"
done
