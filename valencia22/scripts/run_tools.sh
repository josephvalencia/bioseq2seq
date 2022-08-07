# test pre-trained rnasamba
source commands.sh
full_train_set="$BIOHOME/bioseq2seq/data/mammalian_200-1200_train_RNA_balanced.fa"
pc_train_set="$BIOHOME/bioseq2seq/data/mammalian_200-1200_train_PC_RNA.fa"
nc_train_set="$BIOHOME/bioseq2seq/data/mammalian_200-1200_train_NC_RNA.fa"
# re-train rnasamba on our data and test
rnasamba train -s 3 -e 25 -v 3 rnasamba.hdf5 $pc_train_set $nc_train_set
rnasamba classify test_rnasamba_mammalian.tsv $test_set rnasamba.hdf5

# run CPC2,  must use python2
cd $BIOHOME/bioseq2seq/valencia22/tools/CPC2-beta
export CPC_HOME="$PWD"
python2 bin/CPC2.py -i $test_set -o test_cpc2_mammalian.txt

cd $BIOHOME/bioseq2seq/valencia22/
# CPAT
# split our train set into balanced sets between coding and noncoding
# build necessary tables
make_hexamer_tab.py -c $pc_train_set -n $nc_train_set > mammalian_Hexamer.tsv
make_logitModel.py -x mammalian_Hexamer.tsv -c $pc_train_set -n $nc_train_set -o Mammals
# run on train set to get data for coding threshold script
cpat.py -x mammalian_Hexamer.tsv -d Mammals.logit.RData -g $full_train_set  --top-orf=5 -o train_cpat
# run on our test set
cpat.py -x mammalian_Hexamer.tsv -d Mammals.logit.RData -g $test_set --top-orf=5 -o test_cpat_mammalian


