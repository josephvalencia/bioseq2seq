#!/usr/bin/env bash
source commands.sh
source $BIOHOME/bioseq2seq/venv/bin/activate
parallel -j 4  --lb --tmpdir .  < train_best_bioseq2seq.txt 
parallel -j 4  --lb --tmpdir .  < train_best_EDC.txt 
parallel -j 4  --lb --tmpdir .  < train_equivalents_EDC.txt 
