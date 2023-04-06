export SCRIPT_DIR="experiments/scripts"
source $SCRIPT_DIR/templates.sh

# hyperparam tuning
bash $SCRIPT_DIR/tune_all.sh
# training multiple replicates
bash $SCRIPT_DIR/run.sh $SCRIPT_DIR/train_all_replicates.txt
# identify approximate best models
python $SCRIPT_DIR/parse_tfrecords.py experiments/runs/Jun25_07-51-28_cascade.cgrb.oregonstate.local bioseq2seq Jun25_07-51-41 > top5_bioseq2seq_models.txt
python $SCRIPT_DIR/parse_tfrecords.py experiments/runs/Jun27_08-34-56_cascade.cgrb.oregonstate.local EDC Jun27_08-35-05  > top5_EDC-large_models.txt
python $SCRIPT_DIR/parse_tfrecords.py experiments/runs/Jun29_13-19-50_cascade.cgrb.oregonstate.local EDC_eq Jun29_13-20-08 > top5_EDC-small_models.txt

