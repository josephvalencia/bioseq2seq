# bioseq2seq

## Overview
This is the official code for the preprint [Learning to translate with LocalFilterNets improves prediction of protein coding potential](https://arxiv.org/pdf/1805.11462).
In this work, we demonstrate the utility of incorporating a training task based on biological translation into a deep learning classifier for distinguishing messenger RNAs (mRNAs) from long noncoding RNAs (lncRNAs). We introduce an architecture called LocalFilterNet which helps to capture the nonstationary 3-nucleotide periodicity that is typically present in coding RNA regions. Finally, we perform model interpretation and uncover both well-known and potentially novel sequence features that may be related to translational regulation. 

This repository began as a fork of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py),a library for neural machine translation built by Harvard NLP,  which we have modified for modeling biological translation.

## Citation

If you use our code or data for academic work please cite our preprint:

```
@article{valencia2022translating,
title={Learning to translate with LocalFilterNets improves prediction of protein coding potential},
author={Valencia, Joseph and Hendrix, David},
journaltitle={},
pages={},
year={2022},
pdf={http://arxiv.org/abs/}
}
```

## Installation

bioseq2seq requires Python 3.6 or higher and PyTorch 1.8 or higher.

Install `bioseq2seq` from `pip`:
```bash
pip install bioseq2seq
```
or from  source:
```bash
git clone https://github.com/josephvalencia/bioseq2seq.git
cd bioseq2seq
python setup.py install
```
All dependencies for single-model training and inference are included in the pip distribution as well as [requirements.txt](requirements.txt). Additional dependencies are needed for fully replicating the paper. 

## Pretrained model

We provide pretrained PyTorch weights for our best model `bioseq2seq_mammalian_refseq.pth`

## Usage

### Running inference with a pretrained model

To obtain predictions from a pretrained model, use the `bioseq2seq translate` command. The only required arguments are `--input` for the name of a FASTA file of RNAs, and `--checkpoint` 
for the model weights (.pth).

### Training a new model

To obtain predictions from a pretrained model, use the `bioseq2seq translate` The only required arguments are `--i` for the name of a FASTA file of RNAs, and `--checkpoint` 
for the model weights (.pth).

## Paper experiments

Detailed instructions and code for reproducing the paper results are given in the [valencia22/](valencia22/) directory. This includes:
* data preprocessing
* hyperparameter tuning
* model training 
* inference and evaluation 
* comparisons with alternative software for protein coding potential
* filter visualization
* attribution calculation and motif finding 
* plotting 

