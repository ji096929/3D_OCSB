#!/bin/bash
# Same-scene U with a wider end-to-end K sweep (3/5/10/20 x lam 0.3/0.5).
set -u
cd /data01/jh/code
PY=/data01/jh/conda_envs/coseg/bin/python

run() {
  local gpu=$1 seed=$2
  local out=/data01/jh/data/diag_k_s${seed}
  local log=/data01/jh/data/diag_k_s${seed}.log
  CUDA_VISIBLE_DEVICES=${gpu} setsid nohup ${PY} -u diag_bc.py \
    --episodes 168 --M 3 --seed ${seed} --u-mode same_scene \
    --out ${out} > ${log} 2>&1 < /dev/null &
  echo "launched gpu=${gpu} seed=${seed} pid=$!"
}

run 4 131
run 6 133
