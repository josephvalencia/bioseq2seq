#!/usr/bin/env bash
source venv/bin/activate
export BIOHOME=/home/bb/valejose/home/bioseq2seq
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
#python $BIOHOME/bioseq2seq/bin/hyperparam_search.py --train_src $BIOHOME/new_data/mammalian_200-1200_train_RNA_balanced.fa --train_tgt $BIOHOME/new_data/mammalian_200-1200_train_PROTEIN_balanced.fa --val_src $BIOHOME/new_data/mammalian_200-1200_val_RNA_nonredundant_80.fa --val_tgt $BIOHOME/new_data/mammalian_200-1200_val_PROTEIN_nonredundant_80.fa --num_gpus 1 --mode EDC --save-directory $BIOHOME/checkpoints/coding_noncoding/ --accum_steps 8 --max_tokens 9000 --report-every 500 --model_type LFNet
python $BIOHOME/bioseq2seq/bin/hyperparam_search.py --train_src $BIOHOME/new_data/mammalian_200-1200_train_RNA_balanced.fa --train_tgt $BIOHOME/new_data/mammalian_200-1200_train_PROTEIN_balanced.fa --val_src $BIOHOME/new_data/mammalian_200-1200_val_RNA_nonredundant_80.fa --val_tgt $BIOHOME/new_data/mammalian_200-1200_val_PROTEIN_nonredundant_80.fa --num_gpus 1 --mode bioseq2seq --save-directory $BIOHOME/checkpoints/coding_noncoding/ --accum_steps 8 --max_tokens 9000 --report-every 500 --model_type LFNet

