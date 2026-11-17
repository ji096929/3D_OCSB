#!/bin/bash
# Step 1+2: paired manifest + balanced prototype quota.
# Four runs: {cross,same} x {fps,balanced}, all on the same first N manifest
# episodes so every comparison is paired.
set -u
cd /data01/jh/code
PY=/data01/jh/conda_envs/coseg/bin/python
MAN=/data01/jh/data/manifest_fold0.json
N=${N:-200}

run() {
  local gpu=$1 tag=$2 mode=$3 proto=$4
  local out=/data01/jh/data/pm_${tag}
  local log=/data01/jh/data/pm_${tag}.log
  CUDA_VISIBLE_DEVICES=${gpu} setsid nohup ${PY} -u diag_bc.py \
    --episodes ${N} --manifest ${MAN} --u-mode ${mode} \
    --proto-mode ${proto} --M 3 --seed 7 \
    --out ${out} > ${log} 2>&1 < /dev/null &
  echo "launched gpu=${gpu} ${tag} mode=${mode} proto=${proto} pid=$!"
}

run 4 x_fps  cross_scene fps
run 5 x_bal  cross_scene balanced
run 6 s_fps  same_scene  fps
run 0 s_bal  same_scene  balanced
