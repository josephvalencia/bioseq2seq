#!/usr/bin/env bash
source venv/bin/activate
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python bioseq2seq/bin/translate.py --input ../Fa/h_sapiens/v33/coding_noncoding.map --checkpoint coding_noncoding/Mar24_18-18-28/_step_44000.pt


