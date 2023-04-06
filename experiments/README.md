## Reproducing Experiments
Here we provide instructions and code for reproducing our paper results. Begin from the root directory for the project.

### Data preprocessing
Build a combined mammalian dataset from the raw RefSeq files. Obtain train/test/val split with evaluation sets limited to 80% seq. identity with train set and maximum length of 1200 nt.
```
bash gen_datasets.sh
```
### Hyperparameter tuning
Use Ray Tune to tune hyperparameters for bioseq2seq and EDC using the BOHB algorithm. First you'll need to set your [Weights & Biases](https://wandb.ai/site) API key for logging purposes.

```
export WANDB_KEY=<your_key>
bash experiments/scripts/tune_all.sh
```
### Model training 
Train multiple replicates (4) for best bioseq2seq and EDC hyperparams, as well as EDC equivalent. The driver script `experiments/scripts/run.sh` uses [GNU Parallel](https://www.gnu.org/software/parallel/) to execute multiple commands from a text file in parallel. First, source `experiments/scripts/templates.sh`, which profides many commands for the remaining steps. 

```
source experiments/scripts/templates.sh
bash experiments/scripts/run.sh experiments/scripts/train_all_replicates.txt
```
### Inference and feature attributions on various datasets 
```
bash test_and_attribution.sh
```
### Comparisons with alternative software 
```
bash experiments/scripts/run_tools.sh
```
### Analysis and figure reproduction pipeline
This script produces all figure panels using Matplotlib. A .yaml configuration file controls the pipeline, with both an example and a generic template provided.
```
export CONFIG="example_config.yaml"
bash figs_and_analysis.sh
```
