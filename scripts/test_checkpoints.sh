#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate

for s in {12000..35000..1000}
    do
    python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/mammalian_1k_val_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep09_19-39-32/_step_${s}.pt --output_name bioseq2seq_val_step_${s} --mode combined --rank 0 --num_gpus 4
    cat bioseq2seq_val_step_${s}.rank*.preds > bioseq2seq_val_step_${s}.preds
    rm bioseq2seq_val_step_${s}.rank*
    python $BIOHOME/bioseq2seq/bioseq2seq/bin/eval_translation.py bioseq2seq_val_step_${s}.preds combined >> checkpoint_trials.txt
    done
