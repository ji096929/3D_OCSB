#!/bin/bash
# same_scene U control run. Seeds match diag_gate_s12{1,2,3} so that
# s_info/q_info are identical episode by episode and only U changes.
set -u
cd /data01/jh/code
PY=/data01/jh/conda_envs/coseg/bin/python
mkdir -p /data01/jh/data

run() {
  local gpu=$1 seed=$2
  local out=/data01/jh/data/diag_same_s${seed}
  local log=/data01/jh/data/diag_same_s${seed}.log
  CUDA_VISIBLE_DEVICES=${gpu} setsid nohup ${PY} -u diag_bc.py \
    --episodes 168 --M 3 --seed ${seed} --u-mode same_scene \
    --out ${out} > ${log} 2>&1 < /dev/null &
  echo "launched gpu=${gpu} seed=${seed} pid=$!"
}

run 4 121
run 5 122
run 6 123
