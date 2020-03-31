#!/usr/bin/env bash
source venv/bin/activate
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python bioseq2seq/bin/train.py --input ../Fa/h_sapiens/v33/coding.map --save-directory codingold/ --num_gpus 4 --accum_steps 4 --checkpoint best_checkpoints/Jan22_16-39-08_cascade.cgrb.oregonstate.local.step_10450.pt 
