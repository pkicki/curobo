#!/bin/bash
# Hybrid v2 on the affected cells, then B/C on them in a scratch run dir, merged back as B2_R1/C2_R1.
set -e
cd /home/pkicki/curobo
E=research/timeopt_guided/results/eval_20261114
export PYTHONPATH=/home/pkicki/toc-k107-94d25ec/src:$PWD
P=/home/pkicki/anaconda3/envs/curobo-timeopt/bin/python
declare -A BANK=([main]=research/timeopt_guided/results/benchmark_20261111/bank [hard]=$E/hard_bank)
declare -A GPU=([main]=0 [hard]=1)
for run in main hard; do (
  CUDA_VISIBLE_DEVICES=${GPU[$run]} $P -m research.timeopt_guided.mine_native --bank ${BANK[$run]} --output $E/$run --cells $E/${run}_hybrid2_cells.json --variant hybrid2 > $E/${run}_hybrid2.log 2>&1
  T=$E/${run}_v2_scratch; rm -rf $T; mkdir -p $T/cells
  $P - <<PY
import json, os, shutil
for c in json.load(open("$E/${run}_hybrid2_cells.json")):
    src=f"$E/$run/cells/{c['task_id']}/{c['payload']}"; dst=f"$T/cells/{c['task_id']}/{c['payload']}"
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.startswith("hybrid2_R1"): shutil.copy(os.path.join(src,f), dst)
PY
  CUDA_VISIBLE_DEVICES=${GPU[$run]} $P -m research.timeopt_guided.lane_retime --run $T --bank ${BANK[$run]} --tag hybrid2_R1 --method-tag B2_R1 > $E/${run}_B2.log 2>&1
  [ -f $T/lane_retime.json ] || echo "[]" > $T/lane_retime.json
  CUDA_VISIBLE_DEVICES=${GPU[$run]} $P -m research.timeopt_guided.run_refine --run $T --bank ${BANK[$run]} --tag hybrid2_R1 --all --input-tag B2_R1 --method-tag C2_R1 --out refine_C2 > $E/${run}_C2.log 2>&1
  $P - <<PY
import glob, os, shutil
# merge scratch B2/C2 back; unaffected cells: B2=B, C2=C (identical inputs)
for f in glob.glob("$T/cells/*/*/[BC]2_R1*"):
    rel=os.path.relpath(f, "$T"); shutil.copy(f, os.path.join("$E/$run", rel))
for f in glob.glob("$E/$run/cells/*/*/[BC]_R1*"):
    d, n = os.path.split(f); t = os.path.join(d, n.replace("B_R1","B2_R1").replace("C_R1","C2_R1"))
    if not os.path.exists(t): shutil.copy(f, t)
PY
  $P -m research.timeopt_guided.evaluate export --run $E/$run --bank ${BANK[$run]} --tags hybrid2_R1 B2_R1 C2_R1 >> $E/${run}_C2.log 2>&1
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD /tmp/oracle-ipopt-env/bin/python -m research.timeopt_guided.independent_validate --root $E/$run/validation > $E/${run}_validate_v2.log 2>&1
) & done
wait
echo V2DONE
