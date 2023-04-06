export BIOHOME=/home/bb/valejose/valejose
export PYTHONPATH=$BIOHOME/bioseq2seq
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/coding_noncoding/"
input_prefix="mammalian_200-1200_test_RNA_nonredundant_80"
rna="${BIOHOME}/bioseq2seq/data/${input_prefix}.fa"
prot="${BIOHOME}/bioseq2seq/data/mammalian_200-1200_test_PROTEIN_nonredundant_80.fa"
output_dir="${BIOHOME}/bioseq2seq/experiments/output/bioseq2seq_4_Jun25_07-51-41_step_10500/"

python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --inference_mode bioseq2seq --num_gpus 4 --attribution_mode ISM --max_tokens 300 --checkpoint ${CHKPT_DIR}bioseq2seq_4_Jun25_07-51-41/_step_10500.pt --name $output_dir --tgt_class PC --tgt_pos 1 --rank 0 --minibatch_size 8
python experiments/analysis/combine_npz.py ${output_dir} ${input_prefix}.PC.1.ISM

