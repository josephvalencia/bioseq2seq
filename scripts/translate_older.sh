#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/h_sapiens/v33/coding.map --checkpoint $BIOHOME/bioseq2seq/checkpoints/original/Mar22_16-22-47/_step_82000.pt

