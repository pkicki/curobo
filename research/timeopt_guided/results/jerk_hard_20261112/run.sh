#!/bin/bash
cd /home/pkicki/curobo
R=research/timeopt_guided/results; E=$R/eval_20261114; J=$R/jerk_hard_20261112
export PYTHONPATH=/home/pkicki/toc-k107-94d25ec/src:$PWD CUDA_VISIBLE_DEVICES=1
P=/home/pkicki/anaconda3/envs/curobo-timeopt/bin/python
for v in native native_jerk945 native_jerk940; do
  $P -m research.timeopt_guided.mine_native --bank $E/hard_bank --output $J --split development --variant $v --dense-samples 8193 > $J/$v.log 2>&1
done
$P -m research.timeopt_guided.evaluate export --run $J --bank $E/hard_bank --tags native_R1 native_jerk945_R1 native_jerk940_R1 --gap-check > $J/export.log 2>&1
for i in 0 1 2 3 4 5; do OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD /tmp/oracle-ipopt-env/bin/python -m research.timeopt_guided.independent_validate --root $J/validation --shard $i --nshards 6 > $J/validate.$i.log 2>&1 & done; wait
echo JDONE
