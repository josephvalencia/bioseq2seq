import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys, re, subprocess,shlex
from Bio import SeqIO
import pprint
from BCBio import GFF
from BCBio.GFF import GFFExaminer

def validCDS(cds):
    return len(cds) % 3 == 0 and cds.startswith("AUG") and cds.endswith(("UAG", "UAA", "UGA"))

def alternateStart(cds):
    return not cds.startswith("AUG")

def alternateStop(cds):
    return not cds.endswith(("UAG","UAA", "UGA"))

def parse_GFF(gff):

    save_attributes = set(["gene_id","gene_type","gene_status","transcript_id","transcript_type","transcript_status","protein_id"])
    tags = set(["CCDS","basic","upstream_ATG","downstream_ATG","non_ATG_start","cds_start_NF","cds_end_NF","mRNA_end_NF","mRNA_start_NF"])

    count_pc = count_basic = count_ccds = 0

    with open(gff) as inFile:

        line = inFile.readline()

        while line is not None:

            line = inFile.readline().rstrip()

            if not line.startswith("##"):

                fields = line.split("\t")

                if len(fields) >= 9 and fields[2] == "transcript":

                    entry = {"chr":fields[0],"database":fields[1],"feature":fields[2],"start":fields[3],"end":fields[4],"strand":fields[6]}

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

                    if "basic" in entry:
                        count_basic+=1
                    if status == "protein_coding":
                        count_pc+=1
                    if "CCDS" in entry:
                        count_ccds +=1

    print("# protein coding {} , # basic {} , # CCDS {}".format(count_pc,count_basic,count_ccds))

def parse_mrna(fastaFile):

    cdsPattern = re.compile('CDS:(\d+)-(\d+)')
    fivePattern = re.compile('UTR5:(\d+)-(\d+)')
    threePattern = re.compile('UTR3:(\d+)-(\d+)')

    sequences = SeqIO.parse(fastaFile,"fasta")

    n_total = 0
    n_cds = 0
    n_alt_start = 0
    n_alt_stop = 0
    n_both = 0
    n_noncanonical = 0
    n_left_chopped = 0
    n_right_chopped = 0

    entries = []

    for i,record in enumerate(sequences):

        description =  record.description
        transcript = record.seq.upper().transcribe()

        id_fields = record.id.split("|")
        transcript_id = [x for x in id_fields if x.startswith("ENST")][0]

        n_total+=1

        has_cds = False

        left_chopped = False
        right_chopped = False

        alt_start = False
        alt_stop = False

        has_three = False
        has_five = False

        malformed = False

        five_match = fivePattern.search(description)
        three_match = threePattern.search(description)
        cds_match = cdsPattern.search(description)

        if not five_match is None:
            has_five = True

        if not three_match is None:
            has_three = True

        if not cds_match is None:

            match = cdsPattern.search(description)
            start = int(match.group(1))-1
            end = int(match.group(2))-1
            cds = transcript[start:end+1]

            if validCDS(cds):
                n_cds +=1
                has_cds = True
            else:

                right_chopped = end > len(transcript) - 4  # has_three # fewer than 3 nucleotides -> right truncation
                left_chopped = start < 3 # and has_five # fewer than 3 nucleotides before -> left truncation

                if left_chopped:
                    n_left_chopped +=1
                if right_chopped:
                    n_right_chopped +=1

                if len(cds) % 3 == 0 and not left_chopped or right_chopped:

                    n_noncanonical +=1

                    if alternateStop(cds):
                        n_alt_stop +=1
                        alt_stop =True
                    if alternateStart(cds):
                        n_alt_start +=1
                        alt_start = True
                    if alternateStop(cds) and alternateStart(cds):
                        n_both += 1

                elif len(cds) % 3 != 0 or (right_chopped and has_three) or (left_chopped and has_five):
                    malformed = True

        record = {"ID" : transcript_id,
                  "UTR_5" : has_five,
                  "CDS" : has_cds,
                  "UTR_3" : has_three,
                  "ALT_START" : alt_start,
                  "ALT_STOP" : alt_stop,
                  "MALFORMED" :malformed,
                  "TRANSCRIPT":transcript}

        entries.append(record)

    df = pd.DataFrame.from_records(entries,index = "ID")

    msg = "# Total {} , # w/ valid CDS {} , # w/ non-AUG start {} , # w/ alt stop {}, # Both {} , # Legal non-canonical {}, # Left truncated {} , # Right truncated {}"
    print(msg.format(n_total,n_cds,n_alt_start,n_alt_stop,n_both,n_noncanonical,n_left_chopped,n_right_chopped))

    return df

def emboss_getorf(nucleotide_seq):

    '''Find longest Open Reading Frame (START-STOP) using getorf from needle package.
    See http://bioinf-hpc.ibun.unal.edu.co/cgi-bin/emboss/help/getorf'''

    cmd = "getorf -sequence={} -find=1 -noreverse -stdout -auto".format("asis::"+nucleotide_seq)
    response = subprocess.check_output(shlex.split(cmd),universal_newlines=True)

    orfs = response.rstrip().split("\n")
    orfs = [x.rstrip() for x in orfs if not x.startswith(">asis")]

    # ORFs sorted by size descending
    orfs = sorted(orfs,key = lambda x : len(x),reverse = True)
    return orfs

def parse_results(align_results):

    best_pattern = re.compile("(BEST: )(\D*)( PCT: )(1|0\.\d*)")
    best_n_pattern = re.compile("(BEST_N: )(\D*)( PCT )(1|0\.\d*)")

    with open(align_results,'r') as inFile:
        lines = inFile.readlines()

    all_id = []
    all_best = []
    all_best_pct = []
    all_best_n = []
    all_best_n_pct = []
    all_gold = []

    for i in range(0,len(lines),5):

        id = lines[i].split("NAME: ")[1].rstrip()

        best_match = best_pattern.search(lines[i+1])
        best = best_match.group(2)
        best_pct = float(best_match.group(4))

        best_n_match = best_n_pattern.search(lines[i+2])
        best_n = best_n_match.group(2)
        best_n_pct = float(best_n_match.group(4))

        gold = lines[i+3].split("GOLD: ")[1].rstrip()

        all_id.append(id)

        all_best.append(best)
        all_best_pct.append(best_pct)

        all_best_n.append(best_n)
        all_best_n_pct.append(best_n_pct)
        all_gold.append(gold)

    return all_id, all_best, all_best_pct, all_best_n, all_best_n_pct, all_gold

if __name__ == "__main__":

    align_results = sys.argv[1]
    mRNA_fasta = sys.argv[2]
    gff = sys.argv[3]

    all_names,all_best,all_best_pct,all_best_n,all_best_n_pct,all_gold = parse_results(align_results)

    parse_GFF(gff)

    quit()

    df = parse_mrna(mRNA_fasta)

    df_dev = df.loc[all_names]
    df_dev['BEST'] = all_best
    df_dev['BEST_N'] = all_best_n
    df_dev['BEST_PCT'] = all_best_pct
    df_dev['BEST_N_PCT'] = all_best_n_pct
    df_dev["GOLD"] = all_gold
    df_dev["INCOMPLETE"] = [ True if not x.startswith("M") else False for x in all_gold]

    bad = df_dev.query("BEST_PCT < 0.4")

    ids = bad.index.values
    transcripts = bad["TRANSCRIPT"].values
    preds = bad["BEST"].values
    golds = bad["GOLD"].values

    for id,mRNA,pred, gold in zip(ids,transcripts,preds,golds):

        print(id)
        print("mRNA:")
        print(mRNA)
        orfs = emboss_getorf(mRNA)
        print("PRED:")
        print(pred)
        print("GOLD:")
        print(gold)

        for i, orf in enumerate(orfs):
            print("ORF_{} : {}".format(i,orf))
        print("")

    '''well_behaved = df_dev.query("CDS & UTR_3 & UTR_5 ")
    cds = df_dev.query("CDS")
    three_prime = df_dev.query("UTR_3")
    no_three_prime = df_dev.query("not UTR_3")
    five_prime = df_dev.query("UTR_5")
    no_five_prime = df_dev.query("not UTR_5")
    incomplete = df_dev.query("INCOMPLETE")
    malformed = df_dev.query("MALFORMED")
    alt_start = df_dev.query("ALT_START")
    alt_stop = df_dev.query("ALT_STOP")

    print("ALL")
    print(df_dev.shape)
    print(df_dev.mean())

    print("WELL BEHAVED")
    print(well_behaved.shape)
    print("MEAN)
    print(well_behaved.mean())

    print("CDS")
    print(cds.shape)
    print(cds.mean())

    print("With 3'UTR")
    print(three_prime.shape)
    print("MEAN")
    print(three_prime.mean())

    print("Missing 3'UTR")
    print(no_three_prime.shape)
    print("MEAN")
    print(no_three_prime.mean())

    print("With 5'UTR")
    print(five_prime.shape)
    print("MEAN")
    print(five_prime.mean())

    print("Missing 5'UTR")
    print(no_five_prime.shape)
    print("MEAN")
    print(no_five_prime.mean())

    print("Missing 5'UTR")
    print(no_five_prime.shape)
    print("MEAN")
    print(no_five_prime.mean())

    print("Incomplete")
    print(incomplete.shape)
    print("MEAN")
    print(incomplete.mean())

    print("Alt START")
    print(alt_start)
    print("MEAN")
    print(alt_start.mean())
    print()

    print("Alt STOP")
    print(alt_stop)
    print("MEAN")
    print(alt_stop.mean())
    print()
    '''


