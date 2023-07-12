source venv/bin/activate
source experiments/scripts/templates.sh

input_prefix="mammalian_200-1200_test_RNA_nonredundant_80"
rna="data/${input_prefix}.fa"
prot="data/mammalian_200-1200_test_PROTEIN_nonredundant_80.fa"
#output_dir="${OUT_DIR}/bioseq2seq_lambd_0.1_Jun23_07-01-34_step_19000/"
output_dir="${OUT_DIR}/bioseq2seq_CNN_3_Jul02_19-50-23_step_11000/"

#--checkpoint ${CHKPT_DIR}/bioseq2seq_lambd_0.1_Jun23_07-01-34/_step_19000.pt --name $output_dir \
python bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot \
--mode bioseq2seq --num_gpus 1 --attribution_mode ISM --max_tokens 300 \
--checkpoint ${CHKPT_DIR}/bioseq2seq_CNN_3_Jul02_19-50-23/_step_11000.pt --name $output_dir \
--tgt_class PC --tgt_pos 1 --rank 0 --minibatch_size 64

python experiments/analysis/combine_npz.py ${output_dir} ${input_prefix}.PC.1.ISM

