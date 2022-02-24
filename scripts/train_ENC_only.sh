#!/usr/bin/env bash
source venv/bin/activate
export BIOHOME=/home/bb/valejose/home/bioseq2seq
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python $BIOHOME/bioseq2seq/bin/train_E_classifier.py --train mammalian_200-1200/mammalian_200-1200_train_balanced.csv --val mammalian_200-1200/mammalian_200-1200_val_nonredundant_80.csv --num_gpus 4 --mode ENC_only --save-directory $BIOHOME/checkpoints/coding_noncoding/ --accum_steps 4 --max_tokens 4500 --n_enc_layers 8 --model_dim 32 --max_rel_pos 8 --report-every 500 --max-epochs 100000 --patience 5 --lr 1e-3 
