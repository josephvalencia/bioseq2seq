#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --output_name best_seq2seq_test_layer0_B  --mode combined --attn_save_layer 0 --dataset test --rank 0 
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --output_name best_seq2seq_test_layer1_B  --mode combined --attn_save_layer 1 --dataset test --rank 0
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --output_name best_seq2seq_test_layer2_B  --mode combined --attn_save_layer 2 --dataset test --rank 0
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --output_name best_seq2seq_test_layer3_B  --mode combined --attn_save_layer 3 --dataset test --rank 0
