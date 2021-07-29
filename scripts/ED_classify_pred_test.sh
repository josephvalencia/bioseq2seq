#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
# run predictions on test data
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct06_18-51-35/_step_150000.pt --output_name ED_classify_1_test  --mode ED_classify --dataset test --rank 0 
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct10_09-23-37/_step_150000.pt --output_name ED_classify_2_test  --mode ED_classify --dataset test --rank 0
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/bioseq2seq/data/mammalian_1k_test_nonredundant_80.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct12_15-42-19/_step_150000.pt --output_name ED_classify_3_test --mode ED_classify --num_gpus 4 --rank 0 --save_SA
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct14_20-21-43/_step_150000.pt --output_name ED_classify_4_test  --mode ED_classify --dataset test --rank 0
#python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/test.csv --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Oct17_13-56-45/_step_150000.pt --output_name ED_classify_5_test  --mode ED_classify --dataset test --rank 0
