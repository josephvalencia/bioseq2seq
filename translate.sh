#!/usr/bin/env bash
source venv/bin/activate
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python bioseq2seq/bin/translate.py --input ../Fa/h_sapiens/v33/coding.map --checkpoint best_checkpoints/Jan22_16-39-08_cascade.cgrb.oregonstate.local.step_10450.pt


