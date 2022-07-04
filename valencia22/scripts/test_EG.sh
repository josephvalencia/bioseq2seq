#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
python $BIOHOME/bioseq2seq/bioseq2seq/bin/run_attribution.py --input $BIOHOME/bioseq2seq/new_data/mammalian_200-1200_val_RNA_nonredundant_80.fa --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/bioseq2seq_1_Apr23_11-26-20/_step_12500.pt --name bioseq2seq_1 --rank 0 --num_gpus 0 --attribution_mode EG --inference_mode bioseq2seq --max_tokens 1200
