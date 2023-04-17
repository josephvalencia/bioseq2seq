# bioseq2seq

## Overview
This is the official repository for the preprint [Improving deep models of protein-coding potential with a Fourier-transform architecture and machine translation task](https://www.biorxiv.org/content/10.1101/2023.04.03.535488v1).

This repository began as a fork of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), a library for neural machine translation built by Harvard NLP. We have modified and extended it for modeling biological translation.

## Installation

Install `bioseq2seq` from source:

```bash
git clone https://github.com/josephvalencia/bioseq2seq.git
cd bioseq2seq
./install.sh
```
This codebase requires Python 3.6 or higher. To set up and activate virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
All dependencies for single-model training and inference are included in [requirements.txt](requirements.txt). Use pip to install them.
```
python -m pip install -r requirements.txt
```
## Basic usage
We provide pretrained PyTorch weights for our best model `best_bioseq2seq_mammalian_200-1200.pt` and our testing set
`mammalian_200-1200_test_RNA_nonredundant_80.fasta`, as described in the manuscript. To obtain a prediction
```
python bioseq2seq/bin/translate.py --checkpoint best_bioseq2seq_mammalian_200-1200.pt --input mammalian_200-1200_test_RNA_nonredundant_80.fasta --num_gpus 1 
```
To output peptide predictions for mRNAs
```
python bioseq2seq/bin/translate.py --checkpoint best_bioseq2seq_mammalian_200-1200.pt --input mammalian_200-1200_test_RNA_nonredundant_80.fasta --mode bioseq2seq --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200 --max_decode_len 400
```
This will produce four prediction hypotheses, ranked in descending order of score.

For further usage options, see
```
python bioseq2seq/bin/translate.py --help
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
