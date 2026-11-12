#!/bin/bash
# Watchdog for the Stage II run: appends the latest validation point every
# two minutes and records whether the process is still alive.
LOG=/data01/jh/data/s2_watch.log
TRAINLOG=/data01/jh/data/s2_main2.log
echo "=== watchdog start $(date +%H:%M:%S) ===" >> $LOG
while true; do
  TS=$(date +%H:%M:%S)
  PID=$(ps -eo pid,cmd | grep "[t]rain_stage2.py" | awk '{print $1}' | head -1)
  if [ -z "$PID" ]; then
    echo "$TS TRAIN PROCESS GONE" >> $LOG
    break
  fi
  ETA=$(ps -o etime= -p "$PID" | tr -d ' ')
  LAST=$(grep -E "VAL" "$TRAINLOG" | tail -1)
  EPD=$(grep -E "epoch .* done" "$TRAINLOG" | tail -1)
  echo "$TS pid=$PID el=$ETA | $EPD | $LAST" >> $LOG
  sleep 120
done
echo "=== watchdog exit $(date +%H:%M:%S) ===" >> $LOG
