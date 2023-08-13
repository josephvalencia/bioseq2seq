export CHKPT_DIR="experiments/checkpoints/coding_noncoding"
export OUT_DIR="experiments/output"
export test_rna="data/mammalian_200-1200_test_RNA_nonredundant_80.fa"
export test_prot="data/mammalian_200-1200_test_PROTEIN_nonredundant_80.fa"
export val_rna="data/mammalian_200-1200_val_RNA_nonredundant_80.fa"
export val_prot="data/mammalian_200-1200_val_PROTEIN_nonredundant_80.fa"
export verified_test_rna="data/verified_test_RNA.fa"
export verified_test_prot="data/verified_test_PROTEIN.fa"
export verified_val_rna="data/verified_val_RNA.fa"
export verified_val_prot="data/verified_val_PROTEIN.fa"

export train="python bioseq2seq/bin/train_single_model.py \
--train_src data/mammalian_200-1200_train_RNA_balanced.fa \
--train_tgt data/mammalian_200-1200_train_PROTEIN_balanced.fa \
--val_src data/mammalian_200-1200_val_RNA_nonredundant_80.fa \
--val_tgt data/mammalian_200-1200_val_PROTEIN_nonredundant_80.fa 
--num_gpus 1 --save-directory experiments/checkpoints/coding_noncoding/ 
--accum_steps 8 --max_tokens 9000 --report-every 500 
--max-epochs 20000 --patience 5 --lr 1.0"

export train_alt="python bioseq2seq/bin/train_single_model.py \
--train_src data/mammalian_200-1200_train_RNA_no_lncPEP.fa \
--train_tgt data/mammalian_200-1200_train_PROTEIN_no_lncPEP.fa \
--val_src data/mammalian_200-1200_val_nonredundant_80_RNA_no_lncPEP.fa \
--val_tgt data/mammalian_200-1200_val_nonredundant_80_PROTEIN_no_lncPEP.fa 
--num_gpus 1 --save-directory experiments/checkpoints/coding_noncoding/ 
--accum_steps 8 --max_tokens 9000 --report-every 500 
--max-epochs 20000 --patience 5 --lr 1.0"

export TRAIN_BIO_CNN="$train --mode bioseq2seq \
--n_enc_layers 8 --n_dec_layers 2 \
--model_dim 128 --dropout 0.5 --model_type CNN-Transformer \
--encoder_kernel_size 6 --encoder_dilation_factor 2 --lr_warmup_steps 2000"

export TRAIN_CDS="$train --mode start \
--n_enc_layers 16 --model_dim 64 --dropout 0.1 --model_type LFNet \
--window_size 200 --lambd_L1 0.001 --lr_warmup_steps 4000"

export TRAIN_CDS_CNN="$train --mode start \
--n_enc_layers 12 --model_dim 64 --dropout 0.2 --model_type CNN-Transformer \
--lr_warmup_steps 4000 --encoder_kernel_size 3 --encoder_dilation_factor 2 "

export TRAIN_BIO="$train --mode bioseq2seq \
--n_enc_layers 12 --n_dec_layers 2 \
--model_dim 64 --dropout 0.2 --model_type LFNet \
--window_size 250 --lambd_L1 0.004 --lr_warmup_steps 2000"

export TRAIN_EDC="$train --mode EDC \
--n_enc_layers 16 --n_dec_layers 16 \
--model_dim 128 --dropout 0.1 --model_type LFNet \
--window_size 200 --lambd_L1 0.011 --lr_warmup_steps 4000"

export TRAIN_EDC_CNN="$train --mode EDC \
--n_enc_layers 4 --n_dec_layers 12 \
--model_dim 64 --dropout 0.4 --model_type CNN-Transformer \
--encoder_kernel_size 9 --encoder_dilation_factor 2 --lr_warmup_steps 8000"

export TRAIN_EDC_EQ="$train --mode EDC \
--n_enc_layers 12 --n_dec_layers 2 \
--model_dim 64 --dropout 0.2 --model_type LFNet \
--window_size 250 --lambd_L1 0.004 --lr_warmup_steps 2000"

export PRED_TEST_BIO="python bioseq2seq/bin/translate.py \
--input $test_rna --mode bioseq2seq  --num_gpus 1 --beam_size 4
--n_best 4 --max_decode_len 400 --max_tokens 1200" 

export PRED_TEST_BIO_CLASS="python bioseq2seq/bin/translate.py \
--input $test_rna --mode bioseq2seq  --num_gpus 1 --beam_size 4 \
--n_best 4 --max_decode_len 1 --max_tokens 1200 --save_EDA" 

export PRED_TEST_START="python bioseq2seq/bin/seq2start.py \
--input $test_rna --mode start  --num_gpus 1 --max_tokens 1200" 

export PRED_TEST_EDC="python bioseq2seq/bin/translate.py \
--input $test_rna --mode EDC --num_gpus 1 --beam_size 4 \
--n_best 4 --max_tokens 1200 --save_EDA" 

export ATTR_VAL_VERIFIED_BIO="python bioseq2seq/interpret/run_attribution.py \
--input $verified_val_rna --tgt_input $verified_val_prot --mode bioseq2seq \
--num_gpus 1 --max_tokens 300"

export ATTR_VAL_VERIFIED_EDC="python bioseq2seq/interpret/run_attribution.py \
--input $verified_val_rna --tgt_input $verified_val_prot --mode EDC --num_gpus 1 --max_tokens 300"

export ATTR_TEST_VERIFIED_BIO="python bioseq2seq/interpret/run_attribution.py \
--input $verified_test_rna --tgt_input $verified_test_prot --mode bioseq2seq --num_gpus 1 --max_tokens 300"

export ATTR_TEST_VERIFIED_EDC="python bioseq2seq/interpret/run_attribution.py \
--input $verified_test_rna --tgt_input $verified_test_prot --mode EDC --num_gpus 1 --max_tokens 300"

export ATTR_TEST_FULL_BIO="python bioseq2seq/interpret/run_attribution.py \
--input $test_rna --tgt_input $test_prot --mode bioseq2seq --num_gpus 1 --max_tokens 300"
