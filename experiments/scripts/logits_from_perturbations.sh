export PYTHONPATH=$BIOHOME/bioseq2seq
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/final/"
wildtype="verified_test_RNA"
shuffled_5_prime="${wildtype}_2-nuc_shuffled_5-prime"
shuffled_3_prime="${wildtype}_2-nuc_shuffled_3-prime"
shuffled_5_prime_1nt="${wildtype}_1-nuc_shuffled_5-prime"
shuffled_3_prime_1nt="${wildtype}_1-nuc_shuffled_3-prime"
shuffled_CDS="${wildtype}_3-nuc_shuffled_CDS"
prot="${BIOHOME}/bioseq2seq/data/mammalian_200-1200_test_PROTEIN_nonredundant_80.fa"
BIO_dir="${BIOHOME}/bioseq2seq/experiments/output/bioseq2seq_lambd_0.1_3_Aug15_16-41-29_step_8000/"
CNN_dir="${BIOHOME}/bioseq2seq/experiments/output/bioseq2seq_CNN_lambd_0.05_3_Jul13_13-53-22_step_9000/"
EDC_dir="${BIOHOME}/bioseq2seq/experiments/output/EDC_1_Jun27_08-35-05_step_6000/"

#for input_prefix in $wildtype $shuffled_5_prime $shuffled_3_prime $shuffled_CDS "swapped_uORFs"
for input_prefix in $shuffled_5_prime_1nt $shuffled_3_prime_1nt
do
rna="${BIOHOME}/bioseq2seq/data/${input_prefix}.fa"
echo $rna
python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --mode bioseq2seq --num_gpus 1 --attribution_mode logit --max_tokens 300 --checkpoint ${CHKPT_DIR}bioseq2seq_lambd_0.1_3_Aug15_16-41-29/_step_8000.pt --name $BIO_dir --tgt_class PC --tgt_pos 1 --rank 3 --minibatch_size 8

#python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --mode EDC --num_gpus 1 --attribution_mode logit --max_tokens 300 --checkpoint ${CHKPT_DIR}EDC_1_Jun27_08-35-05/_step_6000.pt --name $EDC_dir --tgt_class PC --tgt_pos 1 --rank 3 --minibatch_size 8

python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --mode bioseq2seq --num_gpus 1 --attribution_mode logit --max_tokens 300 --checkpoint ${CHKPT_DIR}bioseq2seq_CNN_lambd_0.05_3_Jul13_13-53-22/_step_9000.pt --name $CNN_dir --tgt_class PC --tgt_pos 1 --rank 3 --minibatch_size 8
done

