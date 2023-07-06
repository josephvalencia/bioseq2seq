source venv/bin/activate
export BIOHOME=/home/bb/valejose/valejose/revisions/
export PYTHONPATH=$BIOHOME/bioseq2seq
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/coding_noncoding/"

readarray -t weighted_models < "weighted_models.txt"
for m in "${weighted_models[@]}"
do
a="${m//\//}"
b="${a//\.pt/}"
python $PYTHONPATH/bioseq2seq/bin/translate.py --checkpoint ${CHKPT_DIR}${m} --input ${1} --mode bioseq2seq --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200 --max_decode_len 1 --output_name $b 
done

#readarray -t cnn_models < "cnn_models.txt"
#for m in "${cnn_models[@]}"
#do
#a="${m//\//}"
#b="${a//\.pt/}"
#python $PYTHONPATH/bioseq2seq/bin/translate.py --checkpoint ${CHKPT_DIR}${m} --input ${1} --mode bioseq2seq --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200 --max_decode_len 1 --model_type CNN-Transformer --output_name $b 
#done
