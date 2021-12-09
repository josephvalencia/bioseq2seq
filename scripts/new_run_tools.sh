# test pre-trained rnasamba

# re-train rnasamba on our data and test
rnasamba train -s 3 -e 25 -v 3 rnasamba.hdf5 mammalian_rnasamba_train_PC_RNA.fa mammalian_rnasamba_train_NC_RNA.fa
rnasamba classify test_rnasamba_mammalian.tsv data/mammalian_1k_test_RNA_nonredundant_80.fa rnasamba.hdf5

# run CPC2,  must use python2
cd ../tools/CPC2-beta
export CPC_HOME="$PWD"
python2 bin/CPC2.py -i ../../bioseq2seq/data/mammalian_1k_test_RNA_nonredundant_80.fa -o test_cpc2_mammalian.txt

# CPAT
cd ../../bioseq2seq
# split our train set into balanced sets between coding and noncoding
python sample_RNA.py
# build necessary tables
make_hexamer_tab.py -c mammalian_rnasamba_train_PC_RNA.fa -n mammalian_rnasamba_train_NC_RNA.fa > mammalian_Hexamer.tsv
make_logitModel.py -x mammalian_Hexamer.tsv -c mammalian_rnasamba_train_PC_RNA.fa -n mammalian_rnasamba_train_NC_RNA.fa -o Mammals
# run on train set to get data for coding threshold script
cpat.py -x mammalian_Hexamer.tsv -d Mammals.logit.RData -g mammalian_rnasamba_train_ALL_RNA.fa  --top-orf=5 -o train_cpat
# run on our test set
cpat.py -x mammalian_Hexamer.tsv -d Mammals.logit.RData -g data/mammalian_1k_test_RNA_nonredundant_80.fa --top-orf=5 -o test_cpat_mammalian


