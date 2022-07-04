#!/usr/bin/env bash
source venv/bin/activate
export BIOHOME=/home/bb/valejose/home/bioseq2seq
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python $BIOHOME/bioseq2seq/bin/train_single_model.py --train_src $BIOHOME/new_data/mammalian_200-1200_train_RNA_balanced.fa --train_tgt $BIOHOME/new_data/mammalian_200-1200_train_PROTEIN_balanced.fa --val_src $BIOHOME/new_data/mammalian_200-1200_val_RNA_nonredundant_80.fa --val_tgt $BIOHOME/new_data/mammalian_200-1200_val_PROTEIN_nonredundant_80.fa --num_gpus 4 --rank 0  --mode bioseq2seq --save-directory $BIOHOME/checkpoints/coding_noncoding/ --accum_steps 8 --max_tokens 9000 --n_enc_layers 12 --n_dec_layers 2 --model_dim 64 --report-every 500 --max-epochs 20000 --patience 5 --dropout 0.2 --lr 1.0 --model_type LFNet --window_size 250 --lambd_L1 0.004 --lr_warmup_steps 2000 --name bioseq2seq_1
