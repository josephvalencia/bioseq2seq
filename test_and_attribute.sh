export SCRIPT_DIR="experiments/scripts"
source $SCRIPT_DIR/templates.sh

# regular test set predictions
bash $SCRIPT_DIR/run.sh pred_bioseq2seq.txt
bash $SCRIPT_DIR/run.sh pred_class_bioseq2seq.txt
bash $SCRIPT_DIR/run.sh pred_EDC.txt
bash $SCRIPT_DIR/run.sh pred_EDC_small.txt

# test set preds with encoder-decoder attention
bash $SCRIPT_DIR/run.sh pred_with_attn_bioseq2seq.txt
bash $SCRIPT_DIR/run.sh pred_with_attn_EDC.txt

# verified validation set attributions
bash $SCRIPT_DIR/run.sh mdig_val_bioseq2seq.txt
bash $SCRIPT_DIR/run.sh mdig_val_EDC.txt
bash $SCRIPT_DIR/run.sh ism_val_bioseq2seq.txt
bash $SCRIPT_DIR/run.sh ism_val_EDC.txt
bash $SCRIPT_DIR/run.sh uniform_ig_val_bioseq2seq.txt
bash $SCRIPT_DIR/run.sh uniform_ig_val_EDC.txt
bash $SCRIPT_DIR/run.sh grads_val_verified_bioseq2seq.txt
bash $SCRIPT_DIR/run.sh grads_val_verified_EDC.txt

# verified test set attributions
bash $SCRIPT_DIR/run.sh mdig_test_bioseq2seq.txt
bash $SCRIPT_DIR/run.sh mdig_EDC.txt

# bioseq2seq attributions only from this point
bash $SCRIPT_DIR/run.sh mdig_full_test_bioseq2seq.txt
bash $SCRIPT_DIR/ism_full_test.sh
bash $SCRIPT_DIR/mdig_grad_train.sh
