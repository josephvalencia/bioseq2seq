#!/usr/bin/env bash
source commands.sh
source $BIOHOME/bioseq2seq/venv/bin/activate
parallel --lb -j 4 --tmpdir .  < bioseq2seq_pred_short_replicates.txt 
#parallel --lb -j 4 --tmpdir .  < bioseq2seq_pred_replicates.txt 
#parallel  --lb -j 4 --tmpdir .  < EDC_pred_replicates.txt 
