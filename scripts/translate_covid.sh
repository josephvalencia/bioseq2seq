#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/SARSCoV2_test.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Jun05_13-32-24/_step_95000.pt --output_name covid_1k_layer0head7  --mode combined
