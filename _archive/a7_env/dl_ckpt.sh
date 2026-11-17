#!/bin/bash
# 下载 Sonata 主权重（hf-mirror 不支持 Range，只能整份重下）
set -u
cd /data01/jh/code/sonata/ckpt
for n in sonata; do
  echo "== $n start $(date +%H:%M:%S)"
  curl -sL --retry 5 --max-time 2400 \
       -o "${n}.pth.part" \
       "https://hf-mirror.com/facebook/sonata/resolve/main/${n}.pth"
  SZ=$(stat -c %s "${n}.pth.part" 2>/dev/null || echo 0)
  echo "== $n got ${SZ} bytes"
  if [ "$SZ" -gt 400000000 ]; then
    mv "${n}.pth.part" "${n}.pth"
    echo "== $n OK"
  else
    echo "== $n INCOMPLETE"
  fi
done
ls -lh /data01/jh/code/sonata/ckpt/*.pth
