#!/usr/bin/env bash
#export dir="/nfs/stak/users/valejose/hpc-share/bioseq2seq"
export scripts="${dir}/experiments/scripts"
source $scripts/templates.sh

suffix = ${1}

# verified test set attributions
bash $scripts/run.sh $dir/txt/ism_test_verified_${suffix}.txt
bash $scripts/run.sh $dir/txt/grad_test_verified_${suffix}.txt
bash $scripts/run.sh $dir/txt/mdig_test_verified_${suffix}.txt


