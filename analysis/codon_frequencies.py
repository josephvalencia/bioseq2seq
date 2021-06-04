import sys,re,random
import pandas as pd
from math import log
 
def count_codons(df,K):

    codonCount = {}
    codonTotal = 0.0
    kMerCount = {}
    kMerTotal = 0.0
    extra = []
    tscript_list = df.index.tolist()
    
    for id in tscript_list:
        seq = df.loc[id,'RNA']
        tscript_type = df.loc[id,'Type']
        if tscript_type == "<PC>":               
            # use provided CDS
            cds = df.loc[id,'CDS']
            if cds != "-1":
                splits = cds.split(":")
                clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                cds_start,cds_end = tuple([int(clean(x)) for x in splits])
            else:
                cds_start,cds_end = getLongestORF(seq)
        else:
            # use start and end of longest ORF
            cds_start,cds_end = getLongestORF(seq)
        
        legal_chars = {'A','C','G','T'}
        allowed = lambda codon : all([x in legal_chars for x in codon])
       
        coding = True if (id.startswith('NM') or id.startswith('XM')) else False
        codonPos = {}
        
        if coding:
            # inside CDS
            for i in range(cds_start,cds_end-3,3):
                codon = seq[i:i+K]
                if allowed(codon):
                    if len(codon) == K:
                        if codon not in codonCount:
                            codonCount[codon] = 0.0
                        codonCount[codon] += 1.0
                        codonTotal += 1.0
                        codonPos[i] = 1.0
                    else:
                        print("Found {} position {} has K-mer {}".format(id,i,codon))
                        quit()
        else: 
            shuffled_seq = ''.join(random.sample(seq,len(seq)))
            for i in range(len(seq)-K):
                if i not in codonPos:
                    kMer = seq[i:i+K]
                    if allowed(kMer):
                        if kMer not in kMerCount:
                            kMerCount[kMer] = 0.0
                        # update the count
                        kMerCount[kMer] += 1.0
                        kMerTotal += 1.0
        '''
        for i in range(cds_start,cds_end-1,3):
            codon = seq[i:i+K]
            #print i, codon, start, end
            if len(codon) == K:
                if allowed(codon):
                    if codon not in codonCount:
                        codonCount[codon] = 0.0
                    codonCount[codon] += 1.0
                    codonTotal += 1.0
                    codonPos[i] = 1.0
        for i in range(len(seq)-K):
            if i not in codonPos:
                kMer = seq[i:i+K]
                if allowed(kMer):
                    if kMer not in kMerCount:
                        kMerCount[kMer] = 0.0
                    # update the count
                    kMerCount[kMer] += 1.0
                    kMerTotal += 1.0
        ''' 
    for codon in codonCount:
        codonCount[codon] /= codonTotal
    for kMer in kMerCount:
        kMerCount[kMer] /= kMerTotal
   
    codons = sorted(codonCount.keys(), key=lambda x: codonCount[x], reverse=True)

    codonFile = "codonCountK"+str(K)+"PC.txt"
    with open(codonFile,'w') as outFile:
        outFile.write("codon\tPC_prob\tNC_prob\tenrichment\n")
        for codon in codons:
            score = log(codonCount[codon]/kMerCount[codon],2)
            outFile.write("{}\t{}\t{}\t{}\n".format(str(codon),codonCount[codon],kMerCount[codon],score))

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

def validCDS(seq,start,end):
    
    startCodon = re.compile('ATG')
    stopCodon = re.compile('TGA|TAG|TAA')
    #print "checking ", seq[start:start+3]
    if re.match(startCodon,seq[start:start+3]):
        #print "now checking ", seq[end-2:end+1]
        if re.match(stopCodon,seq[end-2:end+1]):
            return True
        else:
            return False
    else:
        return False

if __name__ == "__main__":

    df = pd.read_csv(sys.argv[1],sep='\t')
    K = int(sys.argv[2])
    df = df.set_index('ID')
    count_codons(df,K)
