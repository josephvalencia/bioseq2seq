#!/usr/bin/env bash
#export dir="/nfs/stak/users/valejose/hpc-share/bioseq2seq"
export dir="/home/bb/valejose/valejose/revisions/bioseq2seq/"
export scripts="${dir}/experiments/scripts"
source $scripts/templates.sh
suffix=${1}

# class-only  
#bash $scripts/run.sh txt/pred_class_${suffix}.txt
# with protein
bash $scripts/run.sh txt/pred_${suffix}.txt
# with encoder-decoder attention
#bash $scripts/run.sh txt/pred_with_attn_${suffix}.txt
