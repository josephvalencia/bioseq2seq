#!/usr/bin/env bash
source venv/bin/activate
python experiments/preprocessing/parse_refseq.py
python experiments/preprocessing/build_datasets.py 1200
# additional BLAST+NEEDLE filtering
python experiments/analysis/parse_blast.py blast data/mammalian_200-1200_train_PROTEIN_balanced.fa data/mammalian_200-1200_test_RNA_nonredundant_80.fa data/mammalian_200-1200_val_RNA_nonredundant_80.fa
python experiments/analysis/parse_blast.py needle data/mammalian_200-1200_test_RNA_nonredundant_80.fa 
