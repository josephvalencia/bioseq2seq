source venv/bin/activate
source experiments/scripts/templates.sh

#input_prefix="mammalian_200-1200_test_RNA_nonredundant_80"
input_prefix="mammalian_200-1200_train_RNA_balanced"
rna="data/${input_prefix}.fa"
#prot="data/mammalian_200-1200_test_PROTEIN_nonredundant_80.fa"
prot="data/mammalian_200-1200_train_PROTEIN_balanced.fa"
chkpt='bioseq2seq_lambd_0.1_3_Aug15_16-41-29/_step_8000.pt'
#chkpt='bioseq2seq_CNN_lambd_0.05_3_Jul13_13-53-22/_step_9000.pt'
#chkpt='bioseq2seq_CNN_3_Jul02_19-50-23/_step_11000.pt'
#chkpt='bioseq2seq_5_Aug12_18-42-39/_step_15500.pt'
dir=$(echo "$chkpt" | sed 's/\/_/_/g' | sed 's/.pt//g')

#python bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot \
#--mode bioseq2seq --num_gpus 4 --attribution_mode ISM --max_tokens 300 \
#--checkpoint ${CHKPT_DIR}/$chkpt --name ${OUT_DIR}/$dir \
#--tgt_class PC --tgt_pos 1 --rank 0 --minibatch_size 64

python bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot \
--mode bioseq2seq --num_gpus 1 --attribution_mode MDIG --sample_size 8 --max_alpha 0.25 \
--max_tokens 300 --checkpoint ${CHKPT_DIR}/$chkpt --name ${OUT_DIR}/$dir \
--tgt_class PC --tgt_pos 1 --rank 0 --minibatch_size 64 

python experiments/analysis/combine_npz.py ${output_dir} ${input_prefix}.PC.1.ISM

