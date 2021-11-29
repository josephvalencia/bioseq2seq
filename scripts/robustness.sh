#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
# run EDC on alt test sets
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/zebrafish_1k_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --output_name EDC_3_test_zebrafish --mode ED_classify --num_gpus 4 --rank 0 
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/mammalian_1k-2k_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --output_name EDC_3_test_mammalian_1k-2k --mode ED_classify --num_gpus 4 --rank 0 
# rn bioseq2seq on alt test sets
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/zebrafish_1k_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep19_19-12-12/_step_150000.pt --output_name seq2seq_3_test_zebrafish --mode combined --rank 0 --num_gpus 4 
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/mammalian_1k-2k_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep19_19-12-12/_step_150000.pt --output_name seq2seq_3_test_mammalian_1k-2k --mode combined --rank 0 --num_gpus 4 
