#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test_reduced_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct19_18-34-47/_step_150000.pt --output_name seq2seq_1_test_reduced_80  --mode combined --dataset test --rank 0
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test_reduced_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep23_20-07-01/_step_150000.pt --output_name seq2seq_2_test_reduced_80  --mode combined --dataset test --rank 0
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test_reduced_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep19_19-12-12/_step_150000.pt --output_name seq2seq_3_test_reduced_80  --mode combined --dataset test --rank 0
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/mammalian_1k_to_2k.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --output_name seq2seq_4_longer_B  --mode combined --rank 0 --num_gpus 4
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test_reduced_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep30_12-23-54/_step_120000.pt --output_name seq2seq_5_longer_B --mode combined --rank 0 --num_gpus 1
