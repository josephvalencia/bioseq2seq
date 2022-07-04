#!/usr/bin/env bash
source venv/bin/activate
export BIOHOME=/home/bb/valejose/home/bioseq2seq
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
parallel -j 4 --tmpdir .  < scripts/train_best_bioseq2seq.txt 
parallel -j 4 --tmpdir .  < scripts/train_best_EDC.txt 
parallel -j 4 --tmpdir .  < scripts/train_equivalents_EDC.txt 
