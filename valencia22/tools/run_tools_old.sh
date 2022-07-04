# re-train rnasamba on our data and test
rnasamba train -s 3 -e 25 -v 3 rnasamba_200-1200.hdf5 mammalian_rnasamba_train_PC_RNA.fa mammalian_rnasamba_train_NC_RNA.fa
#rnasamba classify test_rnasamba_mammalian_1k.tsv data/mammalian_1k_test_RNA_nonredundant_80.fa rnasamba.hdf5
#rnasamba classify test_rnasamba_zebrafish_1k.tsv data/zebrafish_1k_RNA_nonredundant_80.fa rnasamba.hdf5

# run CPC2,  must use python2
#cd ../tools/CPC2-beta
#export CPC_HOME="$PWD"
#export CPC_HOME="../tools/CPC2-beta"
#python2 ../tools/CPC2-beta/bin/CPC2.py -i data/mammalian_1k_test_RNA_nonredundant_80.fa -o test_cpc2_mammalian_1k.txt
#python2 ../tools/CPC2-beta/bin/CPC2.py -i data/mammalian_1k-2k_RNA_nonredundant_80.fa -o test_cpc2_mammalian_1k-2k.txt
#python2 ../tools/CPC2-beta/bin/CPC2.py -i data/zebrafish_1k_RNA_nonredundant_80.fa -o test_cpc2_zebrafish_1k.txt

# CPAT
# build necessary tables
#make_hexamer_tab.py -c mammalian_rnasamba_train_PC_RNA.fa -n mammalian_rnasamba_train_NC_RNA.fa > mammalian_Hexamer.tsv
#make_logitModel.py -x mammalian_Hexamer.tsv -c mammalian_rnasamba_train_PC_RNA.fa -n mammalian_rnasamba_train_NC_RNA.fa -o Mammals
# run on train set to get data for coding threshold script
#cpat.py -x mammalian_Hexamer.tsv -d Mammals.logit.RData -g mammalian_rnasamba_train_ALL_RNA.fa  --top-orf=5 -o train_cpat
# run on our test sets
#cpat.py -x mammalian_Hexamer.tsv -d Mammals.logit.RData -g data/mammalian_1k_test_RNA_nonredundant_80.fa --top-orf=5 -o test_cpat_mammalian_1k
#cpat.py -x mammalian_Hexamer.tsv -d Mammals.logit.RData -g data/mammalian_1k-2k_RNA_nonredundant_80.fa --top-orf=5 -o test_cpat_mammalian_1k-2k
#cpat.py -x mammalian_Hexamer.tsv -d Mammals.logit.RData -g data/zebrafish_1k_RNA_nonredundant_80.fa --top-orf=5 -o test_cpat_zebrafish_1k


