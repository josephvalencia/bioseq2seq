#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/h_sapiens/v33/coding_noncoding.map --checkpoint $BIOHOME/bioseq2seq/checkpoints/best/Apr07_20-56-38_step_54000.pt
