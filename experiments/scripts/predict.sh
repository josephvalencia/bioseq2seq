export BIOHOME=/home/bb/valejose/valejose
export PYTHONPATH=$BIOHOME/bioseq2seq
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/coding_noncoding/"
#source templates.sh

#python $PYTHONPATH/bioseq2seq/bin/translate_new.py --checkpoint ${CHKPT_DIR}/bioseq2seq_4_Jun25_07-51-41/_step_10500.pt --input ${1} --mode bioseq2seq --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200 
python $PYTHONPATH/bioseq2seq/bin/translate_new.py --checkpoint ${CHKPT_DIR}/bioseq2seq_4_Jun25_07-51-41/_step_10500.pt --input ${1} --mode bioseq2seq --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200 --max_decode_len 1
#python $PYTHONPATH/bioseq2seq/bin/translate_new.py --checkpoint ${CHKPT_DIR}/EDC_1_Jun27_08-35-05/_step_6000.pt --input ${1} --mode bioseq2seq --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200 
