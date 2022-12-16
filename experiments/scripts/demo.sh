export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=$BIOHOME/bioseq2seq
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/coding_noncoding/"
rna=${1}_RNA.fa
prot=${1}_PROTEIN.fa
#python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --inference_mode EDC --num_gpus 1 --attribution_mode $2 --max_tokens 300 --checkpoint ${CHKPT_DIR}EDC_1_Jun27_08-35-05/_step_6000.pt --name $1 --tgt_class $3 --tgt_pos $4 
#python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --inference_mode bioseq2seq --num_gpus 1 --attribution_mode $2 --max_tokens 300 --checkpoint ${CHKPT_DIR}bioseq2seq_4_Jun25_07-51-41/_step_10500.pt --name $1 --tgt_class $3 --tgt_pos $4
python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --inference_mode bioseq2seq --num_gpus 1 --attribution_mode $2 --max_tokens 300 --checkpoint ${CHKPT_DIR}bioseq2seq_1_Jun25_07-51-41/_step_8500.pt --name $1 --tgt_class $3 --tgt_pos $4
