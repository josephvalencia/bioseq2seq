#!/usr/bin/env bash
#pip install torch==1.5.1
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/mammalian_1k_val_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct19_18-34-47/_step_150000.pt --output_name seq2seq_1_val  --mode combined --rank 0 --num_gpus 4
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/mammalian_1k_val_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep23_20-07-01/_step_150000.pt --output_name seq2seq_2_val  --mode combined  --rank 0 --num_gpus 4
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/mammalian_1k_val_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep19_19-12-12/_step_150000.pt --output_name seq2seq_3_val  --mode combined --rank 0 --num_gpus 1
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input data/mammalian_200-1200/mammalian_200-1200_val_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Dec03_05-19-44/_step_5500.pt  --output_name seq2seq_val_super_augmented --mode bioseq2seq --rank 0 --num_gpus 4
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/mammalian_1k_val_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep17_20-05-15/_step_52500.pt --output_name seq2seq_4_val --mode combined --rank 0 --num_gpus 1
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/mammalian_1k_val_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Sep30_12-23-54/_step_120000.pt --output_name seq2seq_5_val  --mode combined --rank 0 --num_gpus 4
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input data/mammalian_200-1200/mammalian_200-1200_val_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Jan30_16-32-20/_step_2500.pt --output_name seq2seq_val --mode bioseq2seq --rank 0 --num_gpus 4 --beam_size 4 
