#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/sample_composites.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/abstract_demo.py --pretrained=ckpt-049-7.1163-0.4503
