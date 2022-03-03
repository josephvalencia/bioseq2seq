#!/usr/bin/env bash
source venv/bin/activate
export BIOHOME=/home/bb/valejose/home/bioseq2seq
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
#bash scripts/bioseq2seq_pred_val.sh
python $BIOHOME/bioseq2seq/bin/train.py --train data/mammalian_200-1200/mammalian_200-1200_train_balanced.csv --val data/mammalian_200-1200/mammalian_200-1200_val_nonredundant_80.csv --num_gpus 2 --mode bioseq2seq --save-directory $BIOHOME/checkpoints/coding_noncoding/ --accum_steps 8 --max_tokens 9000 --n_enc_layers 16 --n_dec_layers 16 --model_dim 128 --max_rel_pos 8 --report-every 500 --max-epochs 100000 --patience 10 --lr 1.0
#python $BIOHOME/bioseq2seq/bin/train.py --train data/mammalian_200-1200/mammalian_200-1200_train_balanced.csv --val data/mammalian_200-1200/mammalian_200-1200_val_nonredundant_80.csv --num_gpus 0 --mode bioseq2seq --save-directory $BIOHOME/checkpoints/coding_noncoding/ --accum_steps 1 --max_tokens 9000 --n_enc_layers 8 --n_dec_layers 4 --model_dim 32 --max_rel_pos 8 --report-every 500 --max-epochs 100000 --patience 10 --lr 1e-4 --checkpoint $BIOHOME/checkpoints/coding_noncoding/Jan26_08-04-23/_step_1000.pt

