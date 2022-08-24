## Reproducing Experiments
Here we provide instructions and code for reproducing our paper results.

### Data preprocessing
Build a combined mammalian dataset from the raw RefSeq files. Obtain train/test/val split with evaluation sets limited to 80% seq. identity with train set and maximum length of 1200 nt.
```
./scripts/gen_datasets.sh
```
### Hyperparameter tuning
Use Ray Tune to tune hyperparameters for bioseq2seq and EDC using the BOHB algorithm.
```
./scripts/tune_all.sh
```
### Model training 
Train multiple replicates (4) for best bioseq2seq and EDC hyperparams, as well as EDC equivalent.
```
./scripts/replicates_train.sh
```
### Inference and evaluation 
```
./scripts/replicates_predict.sh
```
### Comparisons with alternative software 
```
./scripts/run_tools.sh
```
### Attribution calculation 
```
./scripts/replicates_attribute.sh
```
### Analysis pipeline
```
./scripts/analysis_pipeline.sh
```
