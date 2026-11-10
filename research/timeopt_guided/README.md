# TimeOpt-guided geometric planning (research harness)

Plan, findings and acceptance criteria: [PLAN.md](PLAN.md). Not a public API.

## Environments

```bash
cd /home/pkicki/curobo
# Pinned K107 (TOPP-RA + Newton) source, created once:
git -C /home/pkicki/time_optimal_curobo worktree add /home/pkicki/toc-k107-94d25ec 94d25ecde95bc7f31851b0b9a78f61bd5f7cb65e
export TOC=/home/pkicki/toc-k107-94d25ec
export CUDA_PY=/home/pkicki/anaconda3/envs/curobo-timeopt/bin/python
export PIN_PY=/tmp/oracle-ipopt-env/bin/python
export PYTHONPATH=$TOC/src:$PWD          # CUDA env only
export OUT=$PWD/research/timeopt_guided/results/campaign_20261110
```

Do not put the Pinocchio interpreter's site-packages into the CUDA environment,
or the reverse. `independent_validate.py` and `report.py` run with `$PIN_PY` and
`PYTHONPATH=$PWD`.

## Tests

```bash
$CUDA_PY -m pytest -q -p no:xdist -o addopts="" tests/test_timeopt_guided.py
PYTHONPATH=$PWD $PIN_PY -m pytest -q -p no:xdist -o addopts="" -p no:cacheprovider tests/test_timeopt_guided.py
```

## Pipeline

```bash
# 1. task bank + endpoint admissibility (all rejections stored)
$CUDA_PY -m research.timeopt_guided.task_bank --output $OUT/bank
# 2. native ladder (failed cells frozen before each escalation)
$CUDA_PY -m research.timeopt_guided.mine_native --bank $OUT/bank --output $OUT/native --restarts 1
$CUDA_PY -m research.timeopt_guided.mine_native --bank $OUT/bank --output $OUT/native --restarts 4 --escalate-from 1
$CUDA_PY -m research.timeopt_guided.mine_native --bank $OUT/bank --output $OUT/native --restarts 16 --escalate-from 4
# 3. native TimeOpt diagnosis on exact native lanes (per budget)
$CUDA_PY -m research.timeopt_guided.seed_diagnosis --run $OUT/native --restarts 1
# 4. guided search (development split), matched arms B/C/D
$CUDA_PY -m research.timeopt_guided.run_campaign --bank $OUT/bank --run $OUT/native \
    --output $OUT/search_dev --split development --search-seeds 3 --max-proposals 128 --native-seeds
# 5. independent validation (Pinocchio env) + report
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 $PIN_PY -m research.timeopt_guided.independent_validate --root $OUT/search_dev
PYTHONPATH=$PWD $PIN_PY -m research.timeopt_guided.report --run $OUT/native --campaign $OUT/search_dev --label dev
# 6. freeze, then the held-out confirmation split
$CUDA_PY -m research.timeopt_guided.freeze --config cfg.json --output $OUT/freeze.json --toc-commit 94d25ec
$CUDA_PY -m research.timeopt_guided.run_campaign ... --split confirmation --config cfg.json --freeze $OUT/freeze.json
```

Every K107 result is preliminary until `independent_validate` passes it.
Raw `cells/` and NPZ files stay local (see `.gitignore`). Commit JSON summaries
and HTML only.
