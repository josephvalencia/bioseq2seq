makeblastdb -in data/train_RNA.fasta -dbtype nucl -parse_seqids -out data/train_RNA
blastn -db data/train_RNA -query data/dev_RNA.fasta -max_hsps 1 -outfmt "7 qaccver saccver pident evalue bitscore qcovs" > train_dev.blast
blastn -db data/train_RNA -query data/test_RNA.fasta -max_hsps 1 -outfmt "7 qaccver saccver pident evalue bitscore qcovs" > train_test.blast
