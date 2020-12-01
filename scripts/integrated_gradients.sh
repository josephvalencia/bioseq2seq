#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep19_19-12-12/_step_150000.pt --name best_seq2seq --rank 0 --num_gpus 4
python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct14_20-21-43/_step_150000.pt --name best_enc_dec --rank 0 --num_gpus 4
