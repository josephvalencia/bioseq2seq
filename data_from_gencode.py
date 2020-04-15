from Bio import SeqIO
import os,sys,re
import shlex
import pandas as pd
import subprocess

def download_GENCODE():

    prefix = "ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_33/"

    downloads = ["annotation.gff3","lncRNA_transcripts.fa","pc_transcripts.fa","pc_translations.fa"]

    for f in downloads:
        remote = "gencode.v33.{}.gz".format(f)
        cmd = shlex.split("wget {}{}".format(prefix,remote))
        subprocess.run(cmd)
        subprocess.run(shlex.split("gunzip "+remote))

class TranslationTable:

    def __init__(self,coding,noncoding):

        self.table = {}
        self.noncoding = noncoding
        self.coding = coding
        self.alt_count = 0
        self.pc_count = 0

    def add_RNA(self,seq_record):

        tscript_name, cds = self.get_name_cds(seq_record.id)

        if tscript_name not in self.table and tscript_name in self.coding:
            new_entry = {"RNA" : seq_record.seq, "TYPE" : "<PC>", "CDS": cds}
            self.table[tscript_name] = new_entry
            self.pc_count +=1

        elif tscript_name not in self.table and tscript_name in self.noncoding:
            new_entry = {"RNA" : seq_record.seq,"TYPE" : "<NC>", "PROTEIN" : "?", "CDS" : cds}
            self.table[tscript_name] = new_entry
            self.alt_count +=1

    def add_protein(self,seq_record):

        id_fields = seq_record.id.split("|")
        tscript_name = [x for x in id_fields if x.startswith("ENST")][0]

        if tscript_name in self.table and "PROTEIN" not in self.table[tscript_name]:
            entry = self.table[tscript_name]
            entry["PROTEIN"] = seq_record.seq
            self.table[tscript_name] = entry

    def to_csv(self,translation_file):

        table = self.table
        linear = [ (k,table[k]["RNA"],table[k]["CDS"],table[k]["TYPE"],table[k]["PROTEIN"]) for k in table.keys() ]
        df = pd.DataFrame(linear,columns = ['ID','RNA','CDS','Type','Protein'])
        df = df.set_index('ID')
        df = df.sample(frac = 1.0, random_state = 65)
        df.to_csv(translation_file,sep = "\t")

    def get_name_cds(self,id):

        bounds = "-1"
        name_pattern = re.compile("ENST(\d*).(\d*)")
        cds_pattern = re.compile("CDS:(\d*-\d*)")

        cds = cds_pattern.search(id)
        name = name_pattern.search(id).group(0)

        if not cds == None:
            bounds = cds.group(1)

        return name , bounds

def dataset_from_fasta_v1(mRNA_file,protein_file,lncRNA_file,coding_list,noncoding_list,outFile):

    """"""
    translation = TranslationTable(coding_list,noncoding_list)

    count_lnc = count_pc = count_protein = 0

    for seq_record in SeqIO.parse(lncRNA_file,"fasta"):
        translation.add_RNA(seq_record)
        count_lnc+=1

    for seq_record in SeqIO.parse(mRNA_file,"fasta"):
        translation.add_RNA(seq_record)
        count_pc +=1

    for seq_record in SeqIO.parse(protein_file,"fasta"):
        translation.add_protein(seq_record)
        count_protein +=1

    msg = "# records lnc {} , # records mRNA {} , # records protein {}".format(count_lnc,count_pc,count_protein)
    print(msg)
    msg = "# parsed ALT {}, # parsed mRNA {}".format(translation.alt_count, translation.pc_count)
    print(msg)

    translation.to_csv(outFile)

def dataset_from_fasta_v2(mRNA_file,protein_file,coding_list,outFile):

    translation = TranslationTable(coding_list,set())

    count_pc = count_protein = 0

    for seq_record in SeqIO.parse(mRNA_file,"fasta"):
        translation.add_RNA(seq_record)
        count_pc +=1

    for seq_record in SeqIO.parse(protein_file,"fasta"):
        translation.add_protein(seq_record)
        count_protein +=1

    translation.to_csv(outFile)

    print(count_pc,count_protein)

def parse_GFF(gff):

    save_attributes = {"ID", "gene_id", "gene_type", "gene_status", "transcript_id", "transcript_type",
                       "transcript_status", "protein_id"}
    tags = {"CCDS", "basic", "upstream_ATG", "downstream_ATG", "non_ATG_start", "cds_start_NF", "cds_end_NF",
            "mRNA_end_NF", "mRNA_start_NF"}
    alt_pc_types = {"nonsense_mediated_decay", "non_stop_decay", "polymorphic_pseudogene"}

    count_pc = count_alt = count_lnc = count_transcript = 0

    pc = set()
    lnc = set()
    alt = set()

    with open(gff) as inFile:
        for line in inFile:
            if not line.startswith("#"):
                fields = line.split("\t")

                if fields[2] == "transcript":

                    count_transcript +=1
                    entry = {"chr" : fields[0],"database" : fields[1],"feature" : fields[2],"start" : fields[3],"end" : fields[4],"strand" : fields[6]}
                    info = fields[8].split(";")
                    attributes = [tuple(l.split("=")) for l in info]

                    for k,v in attributes:
                        if k in save_attributes:
                            entry[k] = v
                        elif k == "tag":
                            present = v.split(",")
                            for p in present:
                                if p in tags:
                                    entry[p] = True

                    status = entry["transcript_type"] if "transcript_type" in entry else None
                    cds_start_NF = "cds_start_NF" in entry
                    cds_end_NF = "cds_end_NF" in entry

                    if status in alt_pc_types or re.match("IG_(\w*)_gene",status) or re.match("TR_(\w*)_gene",status):
                        alt.add(entry["ID"])
                        count_alt +=1
                    if status == "protein_coding" :#and not cds_start_NF and not cds_end_NF:
                        pc.add(entry["ID"])
                        count_pc+=1
                    if status == "lncRNA" or status =="retained_intron" :
                        lnc.add(entry["ID"])
                        count_lnc+=1

    msg = "# transcripts {} , # protein coding {} , # lncRNA {}, # alt coding {}"
    print(msg.format(count_transcript,count_pc,count_lnc,count_alt))

    return pc,alt,lnc

if __name__ =="__main__":

    #download_GENCODE()

    mRNA_file = sys.argv[1]
    lncRNA_file = sys.argv[2]
    protein_file = sys.argv[3]
    annotation_file = sys.argv[4]

    mRNA,alt,noncoding = parse_GFF(annotation_file)
    coding = mRNA | alt

    dataset_from_fasta_v1(mRNA_file,protein_file,lncRNA_file,coding,noncoding,"coding_nocoding_enhanced.map")
    #dataset_from_fasta_v2(mRNA_file,protein_file,mRNA,"coding.map")
