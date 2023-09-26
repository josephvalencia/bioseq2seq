# test pre-trained rnasamba
source venv/bin/activate
source templates.sh
full_train_set="data/mammalian_200-1200_train_RNA_balanced.fa"
pc_train_set="data/mammalian_200-1200_train_PC_RNA.fa"
nc_train_set="data/mammalian_200-1200_train_NC_RNA.fa"
# re-train rnasamba on our data and test
for i in 1 2 3 4 5
do
 rnasamba train -s 5 -e 80 -v 3 rnasamba_${i}.hdf5 $pc_train_set $nc_train_set
rnasamba classify test_rnasamba_mammalian_${i}.tsv $test_set rnasamba_${i}.hdf5
done

# run CPC2,  must use python2
cd bioseq2seq/experiments/tools/CPC2-beta
export CPC_HOME="$PWD"
python2 bin/CPC2.py -i $test_set -o test_cpc2_mammalian.txt

cd bioseq2seq/experiments/
# cpat
# split our train set into balanced sets between coding and noncoding
# build necessary tables
make_hexamer_tab.py -c $pc_train_set -n $nc_train_set > mammalian_hexamer.tsv
make_logitmodel.py -x mammalian_hexamer.tsv -c $pc_train_set -n $nc_train_set -o mammals
# run on train set to get data for coding threshold script
cpat.py -x mammalian_hexamer.tsv -d mammals.logit.rdata -g $full_train_set  --top-orf=5 -o train_cpat
# run on our test set
cpat.py -x mammalian_hexamer.tsv -d mammals.logit.rdata -g $test_set --top-orf=5 -o test_cpat_mammalian


