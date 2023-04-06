export BIOHOME=/home/bb/valejose/valejose
export PYTHONPATH=$BIOHOME/bioseq2seq
export CHKPT_DIR="$BIOHOME/bioseq2seq/experiments/checkpoints/coding_noncoding/"
wildtype="verified_test_RNA"
#shuffled_5_prime="${wildtype}_2-nuc_shuffled_5-prime"
#shuffled_3_prime="${wildtype}_2-nuc_shuffled_3-prime"
shuffled_5_prime="${wildtype}_1-nuc_shuffled_5-prime"
shuffled_3_prime="${wildtype}_1-nuc_shuffled_3-prime"
shuffled_CDS="${wildtype}_3-nuc_shuffled_CDS"
prot="${BIOHOME}/bioseq2seq/data/mammalian_200-1200_test_PROTEIN_nonredundant_80.fa"
BIO_dir="${BIOHOME}/bioseq2seq/experiments/output/bioseq2seq_4_Jun25_07-51-41_step_10500/"
EDC_dir="${BIOHOME}/bioseq2seq/experiments/output/EDC_1_Jun27_08-35-05_step_6000/"

#for input_prefix in $wildtype $shuffled_5_prime $shuffled_3_prime $shuffled_CDS
for input_prefix in $shuffled_5_prime $shuffled_3_prime 
do
rna="${BIOHOME}/bioseq2seq/data/${input_prefix}.fa"
echo $rna
#python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --inference_mode bioseq2seq --num_gpus 1 --attribution_mode logit --max_tokens 300 --checkpoint ${CHKPT_DIR}bioseq2seq_4_Jun25_07-51-41/_step_10500.pt --name $BIO_dir --tgt_class PC --tgt_pos 1 --rank 0 --minibatch_size 8

python $PYTHONPATH/bioseq2seq/interpret/run_attribution.py --input $rna --tgt_input $prot --inference_mode EDC --num_gpus 1 --attribution_mode logit --max_tokens 300 --checkpoint ${CHKPT_DIR}EDC_1_Jun27_08-35-05/_step_500.pt --name $EDC_dir --tgt_class PC --tgt_pos 1 --rank 0 --minibatch_size 8
done

