#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
source commands.sh
$EG_VAL_BIO --checkpoint ${CHKPT_DIR}bioseq2seq_1_Jun25_07-51-41/_step_8500.pt --name bioseq2seq_1_val --rank 0 --max_tokens 400 --mutation_prob 1.0

