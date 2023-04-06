export BIOHOME=/home/bb/valejose/valejose
export PYTHONPATH=$BIOHOME/bioseq2seq
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/coding_noncoding/"
input_prefix="${BIOHOME}/bioseq2seq/data/mammalian_200-1200_train_RNA_balanced"
rna="${input_prefix}.fa"
prot="${BIOHOME}/bioseq2seq/data/mammalian_200-1200_train_PROTEIN_balanced.fa"
output_dir="${BIOHOME}/bioseq2seq/experiments/output/bioseq2seq_4_Jun25_07-51-41_step_10500/"

python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --inference_mode bioseq2seq --num_gpus 4 --attribution_mode MDIG --max_alpha 0.5 --max_tokens 300 --checkpoint ${CHKPT_DIR}bioseq2seq_4_Jun25_07-51-41/_step_10500.pt --name $output_dir --tgt_class PC --tgt_pos 1 --rank 0 --sample_size 8 --minibatch_size 8
python experiments/analysis/combine_npz.py ${output_dir} ${input_prefix}.PC.1.wildtype_logit
python experiments/analysis/combine_npz.py ${output_dir} ${input_prefix}.PC.1.MDIG

python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --inference_mode bioseq2seq --num_gpus 4 --attribution_mode grad --max_tokens 300 --checkpoint ${CHKPT_DIR}bioseq2seq_4_Jun25_07-51-41/_step_10500.pt --name $output_dir --tgt_class PC --tgt_pos 1 --rank 0 --sample_size 8 --minibatch_size 8
python experiments/analysis/combine_npz.py ${output_dir} ${input_prefix}.PC.1.grad
python experiments/analysis/combine_npz.py ${output_dir} ${input_prefix}.PC.1.onehot
