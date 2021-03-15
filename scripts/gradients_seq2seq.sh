#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --name best_ED_classify_A__pos --rank 0 --num_gpus 4 --attribution_mode ig --dataset validation --baseline A
#pkill python
python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep19_19-12-12/_step_150000.pt --name seqseq_3_avg_pos --rank 0 --num_gpus 4 --attribution_mode ig --dataset train --baseline avg
#pkill python
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep19_19-12-12/_step_150000.pt --name seq2seq_3_G_pos --rank 0 --num_gpus 4 --attribution_mode ig --dataset validation --baseline G
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep19_19-12-12/_step_150000.pt --name seq2seq_3_T_pos --rank 0 --num_gpus 4 --attribution_mode ig --dataset validation --baseline T
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --name best_ED_classify_C_pos --rank 0 --num_gpus 4 --attribution_mode ig --dataset validation --baseline C
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --name best_ED_classify_G_pos --rank 0 --num_gpus 4 --attribution_mode ig --dataset validation --baseline G
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/integrated_gradients.py --input $BIOHOME/Fa/refseq_combined_cds.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --name best_ED_classify_T_pos --rank 0 --num_gpus 4 --attribution_mode ig --dataset validation --baseline T
