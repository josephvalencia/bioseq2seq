#!/usr/bin/env bash
export dir="/nfs/stak/users/valejose/hpc-share/bioseq2seq"
export scripts="${dir}/experiments/scripts"
source $scripts/templates.sh

# regular test set predictions
#bash $scripts/run.sh $dir/txt/pred_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/pred_bioseq2seq_weighted.txt
#bash $scripts/run.sh $dir/txt/pred_bioseq2seq_CNN.txt
#bash $scripts/run.sh $dir/txt/pred_bioseq2seq_weighted_CNN.txt

#bash $scripts/run.sh $dir/txt/pred_class_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/pred_EDC.txt
#bash $scripts/run.sh $dir/txt/pred_EDC_small.txt
#bash $scripts/run.sh $dir/txt/pred_class_bioseq2seq_weighted.txt
#bash $scripts/run.sh $dir/txt/pred_class_bioseq2seq_CNN.txt

# test set preds with encoder-decoder attention
#bash $scripts/run.sh $dir/txt/pred_with_attn_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/pred_with_attn_EDC.txt
#bash $scripts/run.sh $dir/txt/pred_with_attn_bioseq2seq_CNN.txt
bash $scripts/run.sh $dir/txt/pred_with_attn_bioseq2seq_weighted.txt

# verified validation set attributions
#bash $scripts/run.sh $dir/txt/mdig_val_verified_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/mdig_val_verified_EDC.txt
#bash $scripts/run.sh $dir/txt/mdig_val_verified_bioseq2seq_CNN.txt
bash $scripts/run.sh $dir/txt/mdig_val_verified_bioseq2seq_weighted.txt

#bash $scripts/run.sh $dir/txt/ism_val_verified_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/ism_val_verified_EDC.txt
#bash $scripts/run.sh $dir/txt/ism_val_verified_bioseq2seq_CNN.txt
bash $scripts/run.sh $dir/txt/ism_val_verified_bioseq2seq_weighted.txt

#bash $scripts/run.sh $dir/txt/uniform_ig_val_verified_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/uniform_ig_val_verified_EDC.txt
#bash $scripts/run.sh $dir/txt/uniform_ig_val_verified_bioseq2seq_CNN.txt
bash $scripts/run.sh $dir/txt/uniform_ig_val_verified_bioseq2seq_weighted.txt

#bash $scripts/run.sh $dir/txt/grad_val_verified_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/grad_val_verified_EDC.txt
#bash $scripts/run.sh $dir/txt/grad_val_verified_bioseq2seq_CNN.txt
bash $scripts/run.sh $dir/txt/grad_val_verified_bioseq2seq_weighted.txt

# verified test set attributions
#bash $scripts/run.sh $dir/txt/mdig_test_verified_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/mdig_test_verified_EDC.txt
#bash $scripts/run.sh $dir/txt/mdig_test_verified_bioseq2seq_cnn.txt
bash $scripts/run.sh $dir/txt/mdig_test_verified_bioseq2seq_weighted.txt

#bash $scripts/run.sh $dir/txt/ism_test_verified_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/ism_test_verified_EDC.txt
#bash $scripts/run.sh $dir/txt/ism_test_verified_bioseq2seq_cnn.txt
bash $scripts/run.sh $dir/txt/ism_test_verified_bioseq2seq_weighted.txt

#bash $scripts/run.sh $dir/txt/grad_test_verified_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/grad_test_verified_EDC.txt
#bash $scripts/run.sh $dir/txt/grad_test_verified_bioseq2seq_cnn.txt
bash $scripts/run.sh $dir/txt/grad_test_verified_bioseq2seq_weighted.txt

# bioseq2seq attributions only on larger datsets
#bash $scripts/run.sh $dir/txt/mdig_test_full_bioseq2seq.txt
#bash $scripts/run.sh $dir/txt/mdig_test_full_bioseq2seq_CNN.txt
#bash $scripts/run.sh $dir/txt/mdig_test_full_bioseq2seq_weighted.txt
#bash $dir/ism_full_test.sh
#bash $dir/mdig_grad_train.sh
