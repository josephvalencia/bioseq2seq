import sys, re
from Bio import SeqIO
from math import log

## subroutines
def countCodons(fastaFile,K):
    codonCount = {}
    codonTotal = 0.0
    kMerCount = {}
    kMerTotal = 0.0
    sequences = SeqIO.parse(fastaFile,'fasta')
    for record in sequences:
        seq = record.seq
        transcriptID = record.id
        transcriptID = transcriptID.split('|')[0]
        transcriptID = re.sub('>','',transcriptID)
        desc = record.description
        match = re.search('CDS:(\d+)-(\d+)',desc)
        start = int(match.group(1))
        end = int(match.group(2))
        start = start - 1
        end = end - 1
        codonPos = {}
        if validCDS(str(seq),start,end):
            for i in range(start,end-1,3):
                codon = seq[i:i+K]
                #print i, codon, start, end
                if len(codon) == K:
                    if codon not in codonCount:
                        codonCount[codon] = 0.0
                    codonCount[codon] += 1.0
                    codonTotal += 1.0
                    codonPos[i] = 1.0
                else:
                    print("Found {} position {} has K-mer {}".format(transcriptID,i,codon))
                    sys.exit()
            for i in range(len(seq)-K):
                if i not in codonPos:
                    kMer = seq[i:i+K]
                    if kMer not in kMerCount:
                        kMerCount[kMer] = 0.0
                    # update the count
                    kMerCount[kMer] += 1.0
                    kMerTotal += 1.0
    for codon in codonCount:
        codonCount[codon] /= codonTotal
    for kMer in kMerCount:
        kMerCount[kMer] /= kMerTotal
    return (codonCount,kMerCount)

def validCDS(seq,start,end):
    if(ALLCDS):
        return True    
    else:
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

############
##  MAIN  ##
############

usage = 'Usage: python ' + sys.argv[0] + ' <fasta file> <K>'

if len(sys.argv) != 3:
    print(usage)
    sys.exit()

ALLCDS = True
fastaFile = sys.argv[1]
K = int(sys.argv[2])

codonProb,kMerProb = countCodons(fastaFile,K)
codons = sorted(codonProb.keys(), key=lambda x: codonProb[x], reverse=True)

codonFile = "codonCountK"+str(K)+".txt"
CF = open(codonFile,'w')
for codon in codons:
    score = log(codonProb[codon]/kMerProb[codon],2)
    CF.write("%s\t%f\t%f\t%f\n" % (str(codon),codonProb[codon],kMerProb[codon],score))
CF.close()
