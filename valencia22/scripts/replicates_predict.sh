#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
#parallel -j 4 --tmpdir .  < scripts/bioseq2seq_pred_replicates.txt 
parallel -j 4 --tmpdir .  < scripts/EDC_pred_replicates.txt 
