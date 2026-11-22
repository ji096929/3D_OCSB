#!/bin/bash
# Task 6: asymmetric fusion + RSV gate + full K sweep, on both U protocols.
# Seeds match the earlier shards so episodes stay comparable.
set -u
cd /data01/jh/code
PY=/data01/jh/conda_envs/coseg/bin/python

run() {
  local gpu=$1 seed=$2 mode=$3 tag=$4
  local out=/data01/jh/data/diag_${tag}_s${seed}
  local log=/data01/jh/data/diag_${tag}_s${seed}.log
  CUDA_VISIBLE_DEVICES=${gpu} setsid nohup ${PY} -u diag_bc.py \
    --episodes 168 --M 3 --seed ${seed} --u-mode ${mode} \
    --out ${out} > ${log} 2>&1 < /dev/null &
  echo "launched gpu=${gpu} seed=${seed} mode=${mode} -> ${out} pid=$!"
}

run 4 121 cross_scene t6x
run 5 123 cross_scene t6x
run 6 131 same_scene  t6s
run 0 133 same_scene  t6s
