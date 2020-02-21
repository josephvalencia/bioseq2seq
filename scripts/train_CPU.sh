#!/usr/bin/env bash
source venv/bin/activate
export PYTHONPATH=/Users/jdv/Research/bioseq2seq
python bioseq2seq/bin/train.py --input translations.map --accum_steps 4 --save-directory ./checkpoints4/ --num_gpus 0
