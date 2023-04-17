export dir="experiments/scripts"
source $dir/templates.sh

# regular test set predictions
bash $dir/run.sh $dir/txt/pred_bioseq2seq.txt
bash $dir/run.sh $dir/txt/pred_class_bioseq2seq.txt
bash $dir/run.sh $dir/txt/pred_EDC.txt
bash $dir/run.sh $dir/txt/pred_EDC_small.txt

# test set preds with encoder-decoder attention
bash $dir/run.sh $dir/txt/pred_with_attn_bioseq2seq.txt
bash $dir/run.sh $dir/txt/pred_with_attn_EDC.txt

# verified validation set attributions
bash $dir/run.sh $dir/txt/mdig_val_bioseq2seq.txt
bash $dir/run.sh $dir/txt/mdig_val_EDC.txt
bash $dir/run.sh $dir/txt/ism_val_bioseq2seq.txt
bash $dir/run.sh $dir/txt/ism_val_EDC.txt
bash $dir/run.sh $dir/txt/uniform_ig_val_bioseq2seq.txt
bash $dir/run.sh $dir/txt/uniform_ig_val_EDC.txt
bash $dir/run.sh $dir/txt/grads_val_verified_bioseq2seq.txt
bash $dir/run.sh $dir/txt/grads_val_verified_EDC.txt

# verified test set attributions
bash $dir/run.sh $dir/txt/mdig_test_bioseq2seq.txt
bash $dir/run.sh $dir/txt/mdig_EDC.txt

# bioseq2seq attributions only on larger datsets
bash $dir/run.sh $dir/txt/mdig_full_test_bioseq2seq.txt
bash $dir/ism_full_test.sh
bash $dir/mdig_grad_train.sh
