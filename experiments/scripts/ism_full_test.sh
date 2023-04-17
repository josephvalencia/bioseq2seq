input_prefix="mammalian_200-1200_test_RNA_nonredundant_80"
rna="data/${input_prefix}.fa"
prot="data/mammalian_200-1200_test_PROTEIN_nonredundant_80.fa"
output_dir="${OUT_DIR}/bioseq2seq_4_Jun25_07-51-41_step_10500/"

python bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot \
--inference_mode bioseq2seq --num_gpus 4 --attribution_mode ISM --max_tokens 300 \
--checkpoint ${CHKPT_DIR}bioseq2seq_4_Jun25_07-51-41/_step_10500.pt --name $output_dir \
--tgt_class PC --tgt_pos 1 --rank 0 --minibatch_size 8

python experiments/analysis/combine_npz.py ${output_dir} ${input_prefix}.PC.1.ISM

