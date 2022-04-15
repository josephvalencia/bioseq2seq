import pandas as pd
import gzip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO

def match_rna2protein_ID(prefix):

    mRNA = set()
    cds = {}
    protein2rna = {}
    with gzip.open(prefix+"feature_table.txt.gz",'rt') as inFile:
        lines = inFile.readlines()
        header = lines[0].split("\t")
        for l in lines:
            fields = l.split()
            if fields[0] == "mRNA":
                rna = fields[10] # product_accession
                protein = fields[11] # non-redundant_refseq
                mRNA.add(rna)
                protein2rna[protein] = rna
    return mRNA,protein2rna

def parse_CDS_from_genbank(path):

    cds_storage = {}
    genbank_file = path+"rna.gbff.gz"
    with gzip.open(genbank_file,"rt") as inFile:
        for rec in SeqIO.parse(inFile,"genbank"):
            cds_feat = [x for x in rec.features if x.type == "CDS"]
            if len(cds_feat) > 0:
                cds = cds_feat[0].location
                cds_storage[rec.id] = "{}:{}".format(cds.start,cds.end)
    return cds_storage

def get_lncRNA_ID(prefix):
    
    lnc = set()
    with gzip.open(prefix+"feature_table.txt.gz",'rt') as inFile:
        lines = inFile.readlines()
        for l in lines:
            fields = l.split()
            if fields[0] == "ncRNA" and fields[1] == "lncRNA":
                lnc.add(fields[11])
    return lnc

def parse_SARSCov2(genome):

    entries = []
    
    with open(genome,'r') as inFile:
        for rec in SeqIO.parse(inFile,"fasta"):
            entries.append((rec.id,rec.seq))

    with open("../Fa/SARSCov2_test.csv",'w') as outFile:
        header = "ID\tRNA\tProtein\tType\tCDS\n"
        outFile.write(header)
        for id,seq in entries:
            line = "{}\t{}\t?\t<PC>\t-1\n".format(id,seq)
            outFile.write(line)

def refseq_RNA(prefix,mRNA,lncRNA,cds,prot2rna,table):

    rna_fasta = prefix+"rna.fna.gz"
    prot_fasta = prefix+"protein.faa.gz"

    lengths = []
    
    with gzip.open(rna_fasta,"rt") as inFile:
        for seq_record in SeqIO.parse(inFile,"fasta"):
            tscript = seq_record.id
            if tscript.startswith("NM_") or tscript.startswith("XM_"):
                if tscript in mRNA:
                    new_entry = {"RNA" : seq_record.seq, "TYPE" : "<PC>"}
                    if tscript in cds:
                        new_entry["CDS"] = cds[tscript]
                    else:
                        new_entry["CDS"] = -1
                    table[tscript] = new_entry
            elif tscript.startswith("NR_") or tscript.startswith("XR_"):                
                if tscript in lncRNA:
                    new_entry = {"RNA" : seq_record.seq, "TYPE" : "<NC>", "PROTEIN" : "[NONE]", "CDS" : "-1"}
                    table[tscript] = new_entry
                    
    with gzip.open(prot_fasta,"rt") as inFile:
        for seq_record in SeqIO.parse(inFile,"fasta"):
            prot = seq_record.id
            if prot in prot2rna:
                tscript_match = prot2rna[prot]
                entry = table[tscript_match]
                entry["PROTEIN"] = seq_record.seq
                table[tscript_match] = entry

    return table

def to_csv(table,filename):

    linear = []
    for k in table.keys():
        if "PROTEIN" in table[k]:
            linear.append((k,table[k]["RNA"],table[k]["TYPE"],table[k]["PROTEIN"],table[k]["CDS"])) 
    
    df = pd.DataFrame(linear,columns = ['ID','RNA','Type','Protein',"CDS"])
    df = df.set_index('ID')
    df = df.sample(frac=1.0,random_state=65)
    df.to_csv(filename,sep = "\t")

def build_dataset(species_names,filename):

    table = {}

    for name,prefix in species_names.items():
        print("Processing {}".format(name))
        path = parent+prefix
        cds = parse_CDS_from_genbank(path)
        print("CDS locations parsed")
        mRNA,protein2rna = match_rna2protein_ID(path)
        print("mRNA and protein linked")
        lncRNA = get_lncRNA_ID(path)
        print("lncRNAs identified")
        table = refseq_RNA(path,mRNA,lncRNA,cds,protein2rna,table)
    print("Finishing")
    to_csv(table,filename)

if __name__ == "__main__":

    parent = "../Fa/refseq/"

    mammalian = {"gorilla":"gorilla_gorilla/GCF_008122165.1_Kamilah_GGO_v0_",
                        "cow" : "bos_taurus/GCF_002263795.1_ARS-UCD1.2_",
                        "mouse" : "mus_musculus/GCF_000001635.26_GRCm38.p6_",
                        "human" : "homo_sapiens/GCF_000001405.39_GRCh38.p13_",
                        "rhesus" : "macaca_mulatta/GCF_003339765.1_Mmul_10_",
                        "chimp" : "pan_troglodytes/GCF_002880755.1_Clint_PTRv2_",
                        "rat" : "rattus_rattus/GCF_011064425.1_Rrattus_CSIRO_v1_",
                        "orangutan" : "pongo_abelii/GCF_002880775.1_Susie_PABv2_"}

    zebrafish = {'zebrafish': 'danio_rerio/GCF_000002035.6_GRCz11_'}
    
    build_dataset(mammalian,'mammalian_refseq.csv')
    #build_dataset(zebrafish,'zebrafish_refseq.csv')

    #covid_genome_path = "/nfs0/BB/Hendrix_Lab/CoV/FASTA/SARSCoV2_genome.fasta"
    #parse_SARSCov2(covid_genome_path)
