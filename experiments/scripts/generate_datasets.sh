#!/usr/bin/env bash
source venv/bin/activate
python experiments/preprocessing/parse_refseq.py
python experiments/preprocessing/build_datasets.py 1200
python experiments/preprocessing/convert_cisbprna.py data/CisBP-RNA_2023_01_27_3_05_pm/PWM.txt
