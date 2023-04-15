#!/usr/bin/env bash
source venv/bin/activate
source experiments/scripts/templates.sh
cat ${1} | parallel --gnu --lb -j 4 --tmpdir .  eval {} --rank {= '$_ = $job->slot() - 1' =}
