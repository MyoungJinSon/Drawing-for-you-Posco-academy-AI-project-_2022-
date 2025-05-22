#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/eval_abstract.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/abstract_eval.py --cuda --pretrained=abstract_final