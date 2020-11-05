#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/refseq_combined.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --output_name best_ED_classify_layer0  --mode ED_classify --attn_save_layer 0
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/refseq_combined.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --output_name best_ED_classify_layer1  --mode ED_classify --attn_save_layer 1
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/refseq_combined.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --output_name best_ED_classify_layer2  --mode ED_classify --attn_save_layer 2
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/refseq_combined.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --output_name best_ED_classify_layer3  --mode ED_classify --attn_save_layer 3
