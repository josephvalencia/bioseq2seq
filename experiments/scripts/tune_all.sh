#!/usr/bin/env bash
source venv/bin/activate
for mode in 'bioseq2seq' 'EDC' 'start' do
for arch in 'LFNet' 'CNN-Transformer' do
CUDA_LAUNCH_BLOCKING=1 python bioseq2seq/bin/hyperparam_search.py --train_src $BIOHOME/data/mammalian_200-1200_train_RNA_balanced.fa --train_tgt $BIOHOME/data/mammalian_200-1200_train_PROTEIN_balanced.fa --val_src $BIOHOME/data/mammalian_200-1200_val_RNA_nonredundant_80.fa --val_tgt $BIOHOME/data/mammalian_200-1200_val_PROTEIN_nonredundant_80.fa --num_gpus 1 --mode $mode --save-directory $BIOHOME/experiments/checkpoints/ --accum_steps 8 --max_tokens 9000 --report-every 500 --model_type $arch
done
done
