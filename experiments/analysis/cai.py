from bioseq2seq.bin.transforms import CodonTable
import math
from Bio.SeqUtils.CodonUsage import CodonAdaptationIndex
from CAI import CAI as external_CAI

class CAI:

    def __init__(self):
        self.table = CodonTable()
        self.rscu = None
    
    def parse_entry(self,entry):

        if any([c == ' ' for c in entry]):
            return None
        else:
            return entry[0], float(entry[1])

    def build_codon_table(self,filename):
       
        raw_frequencies = {}
        with open(filename,'r') as inFile:
            fields = inFile.read().replace('\n','\t').split('\t')
        
        for i in range(0,len(fields),3):
            entry = fields[i:i+3]
            result = self.parse_entry(entry)
            if result:
                raw_frequencies[result[0]] = result[1]
        
        rscu = {}
        weights = {}
        for aa,codons in self.table.aa_to_codon_dict.items():
            codon_freqs = [raw_frequencies[c] for c in codons]
            max_freq = max(codon_freqs)
            mean_freq = sum(codon_freqs) / len(codons)
            for i,c in enumerate(codons):
                rscu[c] = codon_freqs[i] / mean_freq
            for i,c in enumerate(codons):
                weights[c] = codon_freqs[i] / max_freq 
        
        self.rscu = rscu        
        self.weights = weights

    def calculate(self,cds):
        
        CAI = 0.0 
        L = 0
        for i in range(0,len(cds),3):
            codon = cds[i:i+3]
            if codon in self.weights: 
                if not (codon == 'AUG' or codon == 'TGG'):
                    CAI += math.log(self.weights[codon])
                    L+=1
            else:
                print(f'codon {codon} does not exist')
                CAI += math.log(0.5)
                L+=1
        external = external_CAI(cds,weights=self.weights) 
        alt_cai = CodonAdaptationIndex()
        alt_cai.set_cai_index(self.weights)
        CAI = math.exp(CAI / (L-1))
        print(f'BioPython ={alt_cai.cai_for_gene(cds)}, external = {external}, mine = {CAI}')
        return CAI
