#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/val.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep19_19-12-12/_step_150000.pt --name seq2seq_3_zero_pos_val_redo --rank 0 --num_gpus 1 --attribution_mode ig --dataset val --baseline zero
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --name seq2seq_4_zero_pos_val --rank 0 --num_gpus 4 --attribution_mode ig --dataset val --baseline zero
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --name seq2seq_4_A_pos_val --rank 0 --num_gpus 4 --attribution_mode ig --dataset val --baseline A
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --name seq2seq_4_C_pos_val --rank 0 --num_gpus 4 --attribution_mode ig --dataset val --baseline C
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --name seq2seq_4_G_pos_val --rank 0 --num_gpus 4 --attribution_mode ig --dataset val --baseline G
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --name seq2seq_4_T_pos_val --rank 0 --num_gpus 4 --attribution_mode ig --dataset val --baseline T
