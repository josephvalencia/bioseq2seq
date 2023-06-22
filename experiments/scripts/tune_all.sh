#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/valejose/bioseq2seq
export WANDB_KEY="e60c892e60331f36939b382c8086fbeb3bdb8c26"
source venv/bin/activate
#python bioseq2seq/bin/hyperparam_search.py --train_src data/mammalian_200-1200_train_RNA_balanced.fa --train_tgt data/mammalian_200-1200_train_PROTEIN_balanced.fa --val_src data/mammalian_200-1200_val_RNA_nonredundant_80.fa --val_tgt data/mammalian_200-1200_val_PROTEIN_nonredundant_80.fa --num_gpus 1 --mode EDC --save-directory experiments/checkpoints/ --accum_steps 8 --max_tokens 9000 --report-every 500 --model_type LFNet-CNN
python bioseq2seq/bin/hyperparam_search.py --train_src $BIOHOME/data/mammalian_200-1200_train_RNA_balanced.fa --train_tgt $BIOHOME/data/mammalian_200-1200_train_PROTEIN_balanced.fa --val_src $BIOHOME/data/mammalian_200-1200_val_RNA_nonredundant_80.fa --val_tgt $BIOHOME/data/mammalian_200-1200_val_PROTEIN_nonredundant_80.fa --num_gpus 1 --mode bioseq2seq --save-directory $BIOHOME/experiments/checkpoints/ --accum_steps 8 --max_tokens 9000 --report-every 500 --model_type CNN-Transformer

