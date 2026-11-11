#!/bin/bash
# Supervised BKS worker: restarts on crash (finished problems are skipped), then writes a done marker.
# usage: bks_worker.sh <gpu> <which: main|hard> <shard> <nshards>
cd /home/pkicki/curobo
R=research/timeopt_guided/results; E=$R/eval_20261114
export PYTHONPATH=/home/pkicki/toc-k107-94d25ec/src:$PWD OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$1
P=/home/pkicki/anaconda3/envs/curobo-timeopt/bin/python
if [ "$2" = main ]; then
  ARGS="--bank $R/benchmark_20261111/bank --output $R/bks_main_20261115 --eval-run $E/main --oracle-run $R/benchmark_20261111/oracle"
else
  ARGS="--bank $E/hard_bank --output $R/bks_hard_20261115 --eval-run $E/hard --hard-manifest $R/hard_benchmark_20261114/manifest.json"
fi
LOG=$R/bks_$2_20261115.shard$3.log
for attempt in 1 2 3 4 5; do
  $P -m research.timeopt_guided.best_known $ARGS --eval-tags native_R1 hybrid_R1 hybrid2_R1 B_R1 B2_R1 C_R1 C2_R1 \
     --split all --shard $3 --nshards $4 >> $LOG 2>&1 && break
  echo "WORKER_RESTART attempt=$attempt" >> $LOG
  sleep 10
done
echo "rc=$? attempts=$attempt" > $R/bks_$2_20261115.shard$3.done
