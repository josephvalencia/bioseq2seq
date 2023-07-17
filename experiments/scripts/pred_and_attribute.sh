#!/usr/bin/env bash
export dir="experiments/scripts"
source $dir/templates.sh

# regular test set predictions
#bash $dir/run.sh $dir/txt/pred_bioseq2seq.txt
bash $dir/run.sh $dir/txt/pred_bioseq2seq_weighted.txt
#bash $dir/run.sh $dir/txt/pred_bioseq2seq_CNN.txt

#bash $dir/run.sh $dir/txt/pred_class_bioseq2seq.txt
#bash $dir/run.sh $dir/txt/pred_EDC.txt
#bash $dir/run.sh $dir/txt/pred_EDC_small.txt
bash $dir/run.sh $dir/txt/pred_class_bioseq2seq_weighted.txt
#bash $dir/run.sh $dir/txt/pred_class_bioseq2seq_CNN.txt

# test set preds with encoder-decoder attention
#bash $dir/run.sh $dir/txt/pred_with_attn_bioseq2seq.txt
#bash $dir/run.sh $dir/txt/pred_with_attn_EDC.txt
#bash $dir/run.sh $dir/txt/pred_with_attn_bioseq2seq_CNN.txt
bash $dir/run.sh $dir/txt/pred_with_attn_bioseq2seq_weighted.txt

# verified validation set attributions
#bash $dir/run.sh $dir/txt/mdig_val_verified_bioseq2seq.txt
#bash $dir/run.sh $dir/txt/mdig_val_verified_EDC.txt
#bash $dir/run.sh $dir/txt/mdig_val_verified_bioseq2seq_CNN.txt
bash $dir/run.sh $dir/txt/mdig_val_verified_bioseq2seq_weighted.txt

#bash $dir/run.sh $dir/txt/ism_val_verified_bioseq2seq.txt
#bash $dir/run.sh $dir/txt/ism_val_verified_EDC.txt
#bash $dir/run.sh $dir/txt/ism_val_verified_bioseq2seq_CNN.txt
bash $dir/run.sh $dir/txt/ism_val_verified_bioseq2seq_weighted.txt

#bash $dir/run.sh $dir/txt/uniform_ig_val_verified_bioseq2seq.txt
#bash $dir/run.sh $dir/txt/uniform_ig_val_verified_EDC.txt
#bash $dir/run.sh $dir/txt/uniform_ig_val_verified_bioseq2seq_CNN.txt
bash $dir/run.sh $dir/txt/uniform_ig_val_verified_bioseq2seq_weighted.txt

#bash $dir/run.sh $dir/txt/grad_val_verified_bioseq2seq.txt
#bash $dir/run.sh $dir/txt/grad_val_verified_EDC.txt
#bash $dir/run.sh $dir/txt/grad_val_verified_bioseq2seq_CNN.txt
bash $dir/run.sh $dir/txt/grad_val_verified_bioseq2seq_weighted.txt

# verified test set attributions
#bash $dir/run.sh $dir/txt/mdig_test_verified_bioseq2seq.txt
#bash $dir/run.sh $dir/txt/mdig_test_verified_EDC.txt
#bash $dir/run.sh $dir/txt/mdig_test_verified_bioseq2seq_CNN.txt
bash $dir/run.sh $dir/txt/mdig_test_verified_bioseq2seq_weighted.txt

#bash $dir/run.sh $dir/txt/ism_test_verified_bioseq2seq.txt
#bash $dir/run.sh $dir/txt/ism_test_verified_EDC.txt
#bash $dir/run.sh $dir/txt/ism_test_verified_bioseq2seq_CNN.txt
bash $dir/run.sh $dir/txt/ism_test_verified_bioseq2seq_weighted.txt

#bash $dir/run.sh $dir/txt/grad_test_verified_bioseq2seq.txt
#bash $dir/run.sh $dir/txt/grad_test_verified_EDC.txt
bash $dir/run.sh $dir/txt/grad_test_verified_bioseq2seq_CNN.txt
#bash $dir/run.sh $dir/txt/grad_test_verified_bioseq2seq_weighted.txt

# bioseq2seq attributions only on larger datsets
#bash $dir/run.sh $dir/txt/mdig_test_full_bioseq2seq.txt
#bash $dir/run.sh $dir/txt/mdig_test_full_bioseq2seq_CNN.txt
bash $dir/run.sh $dir/txt/mdig_test_full_bioseq2seq_weighted.txt
#bash $dir/ism_full_test.sh
#bash $dir/mdig_grad_train.sh
