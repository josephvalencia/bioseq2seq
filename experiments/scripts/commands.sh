export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=$BIOHOME/bioseq2seq
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/coding_noncoding/"
export test_set="$BIOHOME/bioseq2seq/data/mammalian_200-1200_test_RNA_nonredundant_80.fa"
export val_set="$BIOHOME/bioseq2seq/data/mammalian_200-1200_val_RNA_nonredundant_80.fa"
export out_dir="$BIOHOME/bioseq2seq/experiments/output"

export train="python $PYTHONPATH/bioseq2seq/bin/train_single_model.py --train_src $BIOHOME/data/mammalian_200-1200_train_RNA_balanced.fa --train_tgt $BIOHOME/data/mammalian_200-1200_train_PROTEIN_balanced.fa --val_src $BIOHOME/data/mammalian_200-1200_val_RNA_nonredundant_80.fa --val_tgt $BIOHOME/data/mammalian_200-1200_val_PROTEIN_nonredundant_80.fa --num_gpus 1 --save-directory $BIOHOME/checkpoints/coding_noncoding/ --accum_steps 8 --max_tokens 9000 --report-every 500 --max-epochs 20000 --patience 5 --lr 1.0"

export TRAIN_BIO="$train --mode bioseq2seq --n_enc_layers 12 --n_dec_layers 2 --model_dim 64 --dropout 0.2 --model_type LFNet --window_size 250 --lambd_L1 0.004 --lr_warmup_steps 2000 "

export TRAIN_EDC="$train --mode EDC --n_enc_layers 16 --n_dec_layers 16 --model_dim 128 --dropout 0.1 --model_type LFNet --window_size 200 --lambd_L1 0.011 --lr_warmup_steps 4000 "

export TRAIN_EDC_EQ="$train --mode EDC --n_enc_layers 12 --n_dec_layers 2 --model_dim 64 --dropout 0.2 --model_type LFNet --window_size 250 --lambd_L1 0.004 --lr_warmup_steps 2000"

export PRED_VAL_BIO="python $PYTHONPATH/bioseq2seq/bioseq2seq/bin/translate_new.py --input $val_set --mode bioseq2seq  --num_gpus 1 --beam_size 4 --n_best 4 --max_decode_len 400 --max_tokens 1200" 

export PRED_VAL_EDC="python $PYTHONPATH/bioseq2seq/bin/translate_new.py --input $val_set --mode EDC --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200" 

export PRED_TEST_BIO="python $PYTHONPATH/bioseq2seq/bin/translate_new.py --input $test_set --mode bioseq2seq  --num_gpus 1 --beam_size 4 --n_best 4 --max_decode_len 400 --max_tokens 1200" 

export PRED_TEST_BIO_CLASS="python $PYTHONPATH/bioseq2seq/bin/translate_new.py --input $test_set --mode bioseq2seq  --num_gpus 1 --beam_size 4 --n_best 4 --max_decode_len 1 --max_tokens 1200 --save_EDA" 

export PRED_TEST_EDC="python $PYTHONPATH/bioseq2seq/bin/translate_new.py --input $test_set --mode EDC --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200 --save_EDA" 

export EG_VAL_EDC="python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $val_set --inference_mode EDC --num_gpus 1 --attribution_mode EG --max_tokens 300" 
export EG_TEST_EDC="python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $test_set --inference_mode EDC --num_gpus 1 --attribution_mode EG --max_tokens 300" 
export EG_VAL_BIO="python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $val_set --inference_mode bioseq2seq --num_gpus 1 --attribution_mode EG --max_tokens 1200" 
export EG_TEST_BIO="python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $test_set --inference_mode bioseq2seq --num_gpus 1 --attribution_mode EG --max_tokens 300" 
export GRAD_VAL_EDC="python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $val_set --inference_mode EDC --num_gpus 1 --attribution_mode grad  --max_tokens 300" 
export GRAD_TEST_EDC="python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $test_set --inference_mode EDC --num_gpus 1 --attribution_mode grad --max_tokens 300" 
export GRAD_VAL_BIO="python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $val_set --inference_mode bioseq2seq --num_gpus 1 --attribution_mode grad --max_tokens 1200" 
export GRAD_TEST_BIO="python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $test_set --inference_mode bioseq2seq --num_gpus 1 --attribution_mode grad --max_tokens 300" 
