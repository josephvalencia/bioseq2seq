export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=$BIOHOME/bioseq2seq
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/coding_noncoding/"
readarray model_list < "model_list.txt"
model=${model_list[$3]}
echo $model
python $PYTHONPATH/bioseq2seq/bin/translate_new.py --input $1 --mode bioseq2seq  --num_gpus 1 --beam_size 4 --n_best 4 --max_decode_len 1 --max_tokens 1200 --checkpoint ${CHKPT_DIR}${model} --output_name $2 --rank 3  
