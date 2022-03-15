#!/usr/bin/env bash
source venv/bin/activate
export BIOHOME=/home/bb/valejose/home/bioseq2seq
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python $BIOHOME/bioseq2seq/bin/hyperparam_search.py --train $BIOHOME/data/mammalian_200-1200/mammalian_200-1200_train_balanced.csv --val $BIOHOME/data/mammalian_200-1200/mammalian_200-1200_val_nonredundant_80.csv --num_gpus 0 --mode bioseq2seq --save-directory $BIOHOME/checkpoints/coding_noncoding/ --accum_steps 8 --max_tokens 9000 --report-every 500 --model_type GFNet

