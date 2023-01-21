#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/
export PYTHONPATH=/home/bb/valejose/valejose/bioseq2seq
source $PYTHONPATH/venv/bin/activate
source templates.sh
cat ${1} | parallel --gnu --lb -j 4 --tmpdir .  eval {} --rank {= '$_ = $job->slot() - 1' =}
