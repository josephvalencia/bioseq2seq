#!/usr/bin/env bash
source ../venv/bin/activate
python preprocessing/parse_refseq.py
python ../preprocessing/build_datasets.py 1200
