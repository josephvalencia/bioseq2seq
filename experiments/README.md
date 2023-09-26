# Reproducing Experiments
Here we provide instructions and code for reproducing our paper results. Follow the installation [instructions](../README.md) if you have not already. 

## Prediction and plotting from pretrained models

Download pretrained models and input data from our [OSF](https://osf.io/xaeqg/) project and copy `data/` to [data/](data/) and `checkpoints/` to [experiments/checkpoints/](experiments/checkpoints/). One way to do that is

```
pip install osfclient
osf -p xaeqg clone
mv xaeqg/osfstorage/data data
mkdir experiments/checkpoints
mv xaeqg/osfstorage/checkpoints experiments/checkpoints
```

Use the provided checkpoint files to automatically generate prediction and attribution scripts for all checkpoints. This will create a directory `txt/' full of commands.
```
export $dir="experiments/scripts/txt"
python experiments/scripts/write_commands.py --bio $dir/seq_LFN.txt --edc $dir/class_LFN.txt --edc_small $dir/class_LFN-small.txt --cnn_bio $dir/seq_CNN.txt --cnn_edc $dir/class_CNN.txt --start $dir/start_LFN.txt --cnn_start $dir/start_CNN.txt --weighted_lfnet $dir/seq-wt_LFN.txt --weighted_cnn $dir/seq-wt_CNN.txt --out_dir txt
```

Then source [experiments/scripts/templates.sh](scripts/templates.sh) to set up necessary commands. 
The driver script [experiments/scripts/run.sh](scripts/run.sh) uses [GNU Parallel](https://www.gnu.org/software/parallel/) to execute multiple (4) independent GPU processes in parallel based on plaintext command files. Change `-j <n>` if you have a different number of GPU devices. Then obtain all predictions and attributions on various datasets. 
```
source experiments/scripts/templates.sh
bash experiments/scripts/run.sh txt/
```
Install [CPAT](https://cpat.readthedocs.io/en/latest/#installation), [CPC2](http://cpc2.gao-lab.org/download.php), and [RNAsamba](https://apcamargo.github.io/RNAsamba/installation/) then train/evaluate on our data splits.
```
bash $dir/install_tools.sh
bash $dir/run_tools.sh
```
Produce all figure panels using Matplotlib/Seaborn. A `.yaml` configuration file controls the pipeline. Additional dependencies are the [EMBOSS](https://emboss.sourceforge.net/download/) package and [MEME](https://meme-suite.org/meme/doc/download.html) suite.
```
export CONFIG="new_config.yaml"
bash experiments/scripts/figs_and_analysis.sh
```
Most plots will save in `.svg` format to `new_config_results/` but some will save under `experiments/output/`, with file names logged to the terminal.

## Tuning and training new models from scratch
### Data preprocessing
Copy `refseq/` from OSF to [refseq/](refseq/). Build a combined mammalian dataset from the raw RefSeq files. Obtain train/test/val split with evaluation sequences limited to a maximum of 80% seq. identity with the train set and maximum length of 1200 nt.
```
mv xaeqg/osfstorage/refseq refseq/ 
bash $dir/gen_datasets.sh
```
### Hyperparameter tuning
Use [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to tune hyperparameters for bioseq2seq and EDC using the [BOHB](https://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) algorithm. First you'll need to provide a [Weights & Biases](https://wandb.ai/site) API key for logging purposes.
```
export WANDB_KEY=<your_key>
export dir='experiments/scripts'
bash $dir/tune_all.sh
```
### Model training 
Then, train multiple replicates (5) from best best bioseq2seq and EDC hyperparams, as well as EDC equivalent.

```
source $dir/templates.sh
bash $dir/run.sh $dir/txt/train_all_replicates.txt
```
### Finding best models and building inference scripts
Training wil save `.tfevents` TensorBoard files in one directory and `.pt` PyTorch model checkpoints in another directory, with both prefixed by their creation time. To find the best-performing models
```
python $dir/parse_tfrecords.py <TF_RECORD_DIR> <MODE> <CHECKPOINT_DIR_PREFIX>
```
For example, 
```
python $dir/parse_tfrecords.py experiments/runs/Jun25_07-51-28_cascade.cgrb.oregonstate.local bioseq2seq Jun25_07-51-41 > top1_bioseq2seq_models.txt
python $dir/parse_tfrecords.py experiments/runs/Jun27_08-34-56_cascade.cgrb.oregonstate.local EDC Jun27_08-35-05  > top1_EDC-large_models.txt
python $dir/parse_tfrecords.py experiments/runs/Jun29_13-19-50_cascade.cgrb.oregonstate.local EDC_eq Jun29_13-20-08 > top1_EDC-small_models.txt
```
**NOTE:** The above script selects the best checkpoints based only on (first-token) classification accuracy on the validation set. In the paper, we used the best checkpoints identified during early stopping, which considers both class accuracy and positionwise accuracy on the validation set. To replicate this, use the training logs to identify the best checkpoints and build the `.txt` files manually.
### Inference and feature attributions on various datasets 
Use the `.txt` lists of top checkpoints to write all inference commands
```
python $dir/write_commands.py --bio_rep_file $dir/txt/top1_bioseq2seq_models.txt --edc_rep_file $dir/txt/top1_EDC-large_models.txt --edc_small_rep_file $dir/txt/top1_EDC-small_models.txt 
```
At this point you can move your `.txt` files to [experiments/scripts/txt](experiments/scripts/txt) and redo the [inference and plotting](#prediction-and-plotting-from-pretrained-models) instructions above.
