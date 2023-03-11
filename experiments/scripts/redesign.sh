export BIOHOME=/home/bb/valejose/valejose
export PYTHONPATH=$BIOHOME/bioseq2seq
source templates.sh
rna=${1}_RNA.fa
prot=${1}_PROTEIN.fa
#python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --inference_mode bioseq2seq --num_gpus 4 --attribution_mode langevin --max_tokens 300 --checkpoint ${CHKPT_DIR}/bioseq2seq_4_Jun25_07-51-41/_step_10500.pt --name $1 --tgt_class $2 --tgt_pos $3 --rank 0 --sample_size 32 --minibatch_size 16
python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --inference_mode EDC --num_gpus 4 --attribution_mode langevin --max_tokens 300 --checkpoint ${CHKPT_DIR}/EDC_1_Jun27_08-35-05/_step_6000.pt --name ${OUT_DIR}/EDC_1_Jun27_08-35-05_step_6000/ --tgt_class $2 --tgt_pos $3 --rank 0 --sample_size 32 --minibatch_size 16 


