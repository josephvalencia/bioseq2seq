# bioseq2seq

## Overview
This is the official repositorty for the preprint [Improving deep models of protein-coding potential with a Fourier-transform architecture and machine translation task](https://www.biorxiv.org/content/10.1101/2023.04.03.535488v1).

This repository began as a fork of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), a library for neural machine translation built by Harvard NLP. We have modified and extended it for modeling biological translation.

## Installation

bioseq2seq requires Python 3.6 or higher and PyTorch 1.8 or higher.

Install `bioseq2seq` from source:

```bash
git clone https://github.com/josephvalencia/bioseq2seq.git
cd bioseq2seq
./install.sh
```
All dependencies for single-model training and inference are included in the pip distribution as well as [requirements.txt](requirements.txt). Additional dependencies are needed for fully replicating the paper. 

## Pretrained model

We provide pretrained PyTorch weights for our best model `best_bioseq2seq_mammalian_200-1200.pt` and our testing set
`mammalian_200-1200_test_RNA_nonredundant_80.fasta`

## Basic usage
To obtain a prediction
```
python bioseq2seq/bin/translate_new.py --checkpoint <checkpoint> --input <input_fasta> --num_gpus <n> 
```
e.g.
```
python bioseq2seq/bin/translate_new.py --checkpoint best_bioseq2seq_mammalian_200-1200.pt --input mammalian_200-1200_test_RNA_nonredundant_80.fasta --num_gpus 1 
```
To output peptide predictions for mRNAs, run
```
python bioseq2seq/bin/translate_new.py --checkpoint best_bioseq2seq_mammalian_200-1200.pt --input mammalian_200-1200_test_RNA_nonredundant_80.fasta --mode bioseq2seq --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200 --max_decode_len 400
```
This will produce four prediction hypotheses, ranked in descending order of score.

For further usage options, see
```
python bioseq2seq/bin/translate_new.py --help
```

## Paper experiments

Detailed instructions and code for reproducing the paper results are given in [experiments/](experiments/). This includes:
* data preprocessing
* hyperparameter tuning
* model training 
* inference and evaluation 
* comparisons with alternative software for protein coding potential
* filter visualization
* attribution calculation and motif finding 
* plots for paper figures 
