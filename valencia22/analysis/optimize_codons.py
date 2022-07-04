import json
from Bio import SeqIO
import re
from CAI import CAI
import pandas as pd

def getLongestORF(mRNA):
    ORF_start = -1
    ORF_end = -1
    longestORF = 0
    for startMatch in re.finditer('ATG',mRNA):
        remaining = mRNA[startMatch.start():]
        if re.search('TGA|TAG|TAA',remaining):
            for stopMatch in re.finditer('TGA|TAG|TAA',remaining):
                ORF = remaining[0:stopMatch.end()]
                if len(ORF) % 3 == 0:
                    if len(ORF) > longestORF:
                        ORF_start = startMatch.start()
                        ORF_end = startMatch.start()+stopMatch.end()
                        longestORF = len(ORF)
                    break
    return ORF_start,ORF_end

def load_optimization_dict(filename):
    with open(filename,'r') as inFile:
        strategy = json.loads(inFile.read())
    return strategy

def optimize(opt_strategy,seq,bounds=None):
    
    optimized = []

    if bounds is None:
        cds_start,cds_end = getLongestORF(seq)
    else:
        cds_start,cds_end = bounds

    # preserve 5' UTR
    optimized.append(seq[:cds_start])
    #optimize CDS
    for i in range(cds_start,cds_end,3):
        codon = seq[i:i+3]
        opt_codon = opt_strategy[codon] if codon in opt_strategy else codon
        optimized.append(opt_codon)
    # preserve 3' UTR
    optimized.append(seq[cds_end:])
    
    optimized = ''.join(optimized)
    return optimized

if __name__ == "__main__":

    by_center = load_optimization_dict('optimal_codons_by_center.json')
    by_edge = load_optimization_dict('optimal_codons_by_transition.json')

    reference_file = 'ecol.heg.fasta' 
    references =[]
    for record in SeqIO.parse(reference_file,'fasta'):
        references.append(str(record.seq))

    storage = []
    for record in SeqIO.parse('data/mammalian_1k_test_RNA_nonredundant_80.fa','fasta'):
        coding = record.id.startswith('XM') or record.id.startswith('NM')
        legal_chars = {'A','C','G','T'}
        allowed = lambda s : all([x in legal_chars for x in s])
        rna = str(record.seq)
        if coding and allowed(rna):
            # two optimization methods
            center_optimized = optimize(by_center,rna)
            edge_optimized = optimize(by_edge,rna)
            cds_start,cds_end = getLongestORF(rna)
            
            # original CAI
            cds_original = rna[cds_start:cds_end]
            original_CAI = CAI(cds_original,reference=references)

            # CAI with graph center strategy
            cds_center_optimized = center_optimized[cds_start:cds_end]
            center_CAI = CAI(cds_center_optimized,reference=references)

            # CAI with max edge strategy
            cds_edge_optimized = edge_optimized[cds_start:cds_end]
            edge_CAI = CAI(cds_edge_optimized,reference=references)
            
            entry = {'ID' : record.id , 'CAI_original' : original_CAI, 'CAI_center' : center_CAI , 'CAI_edge' : edge_CAI} 
            storage.append(entry)

    df = pd.DataFrame(storage)
    df.to_csv('CAI_mRNA_test.csv',sep='\t',index=False)
