#!/bin/bash
# 16 线程分段下载 torch 2.5.0 whl（清华镜像单连接限速 ~204KB/s）
set -u
U='https://pypi.tuna.tsinghua.edu.cn/packages/f1/82/adc3a77b9fbbcb79d398d565d39dc0e09f43fff088599d15da81e6cfaaec/torch-2.5.0-cp310-cp310-manylinux1_x86_64.whl'
N=16
SIZE=906443143
OUT=/tmp/torch-2.5.0-cp310-cp310-manylinux1_x86_64.whl
CHUNK=$(( (SIZE + N - 1) / N ))

rm -rf /tmp/torch_parts
mkdir -p /tmp/torch_parts

for i in $(seq 0 $((N - 1))); do
  S=$((i * CHUNK))
  E=$((S + CHUNK - 1))
  if [ "$E" -ge "$SIZE" ]; then E=$((SIZE - 1)); fi
  curl -s -H "Range: bytes=${S}-${E}" --retry 5 --max-time 3600 \
       -o "/tmp/torch_parts/p$(printf %03d "$i")" "$U" &
done
wait

cat /tmp/torch_parts/p* > "$OUT"
ls -lh "$OUT"
sha256sum "$OUT" | cut -d' ' -f1
