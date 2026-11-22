#!/bin/bash
PY=/data01/jh/conda_envs/coseg/bin/python
export PYTHONPATH=/data01/jh/tmp/py_extra
cd /data01/jh/code/COSeg
cat /tmp/todo.txt | xargs -P 4 -I{} timeout 120 $PY preprocess/collect_scannet_data_sp.py \
  --data_path /data01/jh/data/scannet_raw/scans \
  --save_path /data01/jh/data/ScanNet --scene {} --skip-existing >> /data01/jh/data/prep_rest.log 2>&1
echo ALL_DONE >> /data01/jh/data/prep_rest.log
