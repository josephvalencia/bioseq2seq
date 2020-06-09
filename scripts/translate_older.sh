#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/h_sapiens/v33/coding_noncoding.map --checkpoint $BIOHOME/bioseq2seq/checkpoints/best/Mar22_16-22-47_step_80000.pt --output_name old --mode combined

