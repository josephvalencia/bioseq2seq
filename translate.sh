#!/usr/bin/env bash
source venv/bin/activate
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python bioseq2seq/bin/translate.py --input ../Fa/h_sapiens/v32/two_class.map --checkpoint best_checkpoints/Feb17_21-23-04_cascade.cgrb.oregonstate.local.step_37976.pt


