library("biomaRt")

ensembl <- useMart(biomart="ensembl",dataset="hsapiens_gene_ensembl")
listAttributes(ensembl)
refseq <-scan("refseq_ids.txt",character(),quote="")

xr = refseq[startsWith(refseq,"XR")]
xm = refseq[startsWith(refseq,"XM")]
nr = refseq[startsWith(refseq,"NR")]
nm = refseq[startsWith(refseq,"NM")]

mrna <- getBM(attributes=c("refseq_mrna","ensembl_transcript_id_version"),filters="refseq_mrna",values=nm,mart=ensembl)
names(mrna)[1] <- "refseq_transcript_id"

mrna_pred <- getBM(attributes=c("refseq_mrna_predicted","ensembl_transcript_id_version"),filters="refseq_mrna_predicted",values=xm,mart=ensembl) 
names(mrna_pred)[1] <- "refseq_transcript_id"

lncrna <- getBM(attributes=c("refseq_ncrna","ensembl_transcript_id_version"),filters="refseq_ncrna",values=nr,mart=ensembl)
names(lncrna)[1] <- "refseq_transcript_id"

lncrna_pred <- getBM(attributes=c("refseq_ncrna_predicted","ensembl_transcript_id_version"),filters="refseq_ncrna_predicted",values=xr,mart=ensembl)
names(lncrna_pred)[1] <- "refseq_transcript_id"

merged <- rbind(mrna,lncrna,mrna_pred,lncrna_pred)

write.table(merged,file="refseq_to_ensembl.txt",sep="\t",quote=FALSE)