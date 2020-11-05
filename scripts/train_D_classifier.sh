#!/usr/bin/env bash
source venv/bin/activate
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python $BIOHOME/bioseq2seq/bioseq2seq/bin/train_D_classifier.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --num_gpus 4  --save-directory $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/ --accum_steps 1 --max_tokens 5000 --n_enc_layers 6 --n_dec_layers 4 --model_dim 128 --max_rel_pos 10 --report-every 2500 --max-epochs 150000 
