#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
source commands.sh
parallel -j 4 --tmpdir .  < bioseq2seq_grad_replicates.txt 
parallel -j 4 --tmpdir .  < EDC_grad_replicates.txt 
#parallel -j 4 --tmpdir .  < reduced_attr.txt 
#parallel -j 4 --tmpdir .  < sEDC_attr_replicates.txt 
