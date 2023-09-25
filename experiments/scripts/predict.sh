#export BIOHOME=/nfs/stak/users/valejose/hpc-share
export BIOHOME=/home/bb/valejose/valejose/
export PYTHONPATH=$BIOHOME/bioseq2seq
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/final"
echo $CHKPT_DIR
source experiments/scripts/templates.sh
readarray -t models < lncPEP_models.txt

for m in ${models[@]}
do
    dir=$(echo "$m" | sed 's/\/_/_/g' | sed 's/.pt//g')
    if [[ $m =~ ^seq2start*|^CDS* ]];
    then
        python $PYTHONPATH/bioseq2seq/bin/seq2start.py --checkpoint ${CHKPT_DIR}/$m --input ${1} --mode start \
            --output_name ${OUT_DIR}/$dir --num_gpus 1 --max_tokens 1200 
    else
        python $PYTHONPATH/bioseq2seq/bin/translate.py --checkpoint ${CHKPT_DIR}/$m --input ${1} --mode bioseq2seq \
            --output_name ${OUT_DIR}/$dir --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200 --max_decode_len 400 
    fi
done

