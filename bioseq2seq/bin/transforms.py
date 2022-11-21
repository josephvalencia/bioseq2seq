from bioseq2seq.transforms import Transform
import random
import re
import numpy as np

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

class CodonTable(object):

    def __init__(self):
        
        self.aa_to_codon_dict = {'A' : ['GCT','GCC','GCA','GCG'],
                                'C' : ['TGT','TGC'],
                                'D' : ['GAT','GAC'],
                                'E' : ['GAA','GAG'],
                                'F' : ['TTT','TTC'],
                                'G' : ['GGT','GGC','GGA','GGG'],
                                'H' : ['CAT','CAC'],
                                'I' : ['ATT','ATC','ATA'],
                                'K' : ['AAA','AAG'],
                                'L' : ['TTA','TTG','CTT','CTC','CTA','CTG'],
                                'N' : ['AAT','AAC'],
                                'M' : ['ATG'],
                                'P' : ['CCT','CCC','CCA','CCG'],
                                'Q' : ['CAA','CAG'],
                                'R' : ['CGT','CGC','CGA','CGG','AGA','AGG'],
                                'S' : ['TCT','TCC','TCA','TCG','AGT','AGC'],
                                'T' : ['ACT','ACC','ACA','ACG'],
                                'V' : ['GTT','GTC','GTA','GTG'],
                                'W' : ['TGG'],
                                'Y' : ['TAT','TAC'],
                                '*' : ['TAA','TAG','TGA']}

        self.codon_to_aa_dict = {'TTT':'F', 'TTC':'F', 'TTA':'L', 'TTG':'L', 'TCT':'S', 
                                'TCC':'S', 'TCA':'S', 'TCG':'S', 'TAT':'Y', 'TAC':'Y', 
                                'TAA':'*', 'TAG':'*', 'TGT':'C', 'TGC':'C', 'TGA':'*', 
                                'TGG':'W', 'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L',
                                'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P', 'CAT':'H', 
                                'CAC':'H', 'CAA':'Q', 'CAG':'Q', 'CGT':'R', 'CGC':'R', 
                                'CGA':'R', 'CGG':'R', 'ATT':'I', 'ATC':'I', 'ATA':'I', 
                                'ATG':'M', 'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
                                'AAT':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K', 'AGT':'S', 
                                'AGC':'S', 'AGA':'R', 'AGG':'R', 'GTT':'V', 'GTC':'V', 
                                'GTA':'V', 'GTG':'V', 'GCT':'A', 'GCC':'A', 'GCA':'A', 
                                'GCG':'A', 'GAT':'D', 'GAC':'D', 'GAA':'E', 'GAG':'E',
                                'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G',}
    
    def codon_to_aa(self,codon):
        if codon in self.codon_to_aa_dict:
            return self.codon_to_aa_dict[codon] 
        else:
            return None

    def aa_to_codon(self,aa):
        if aa in self.aa_to_codon_dict:
            return self.aa_to_codon_dict[aa] 
        else:
            return None

    def random_synonymous_codon(self,codon):
        if 'N' in codon or 'R' in codon:
            return codon
        else:
            aa = self.codon_to_aa(codon)
            codon_list = self.aa_to_codon(aa)
            return random.choice(codon_list)
    
    def synonymous_codon_list(self,codon):
        if 'N' in codon or 'R' in codon:
            return codon
        else:
            aa = self.codon_to_aa(codon)
            codon_list = self.aa_to_codon(aa)
            return codon_list
    
    def enforce_synonymous(self,new_codon,old_codon):
        if 'N' in codon or 'R' in codon:
            return (False,3)
        else:
            aa1 = self.codon_to_aa(codon1)
            aa2 = self.codon_to_aa(codon2)
            for c1,c2 in zip(aa1,aa2):
                if c1 != c2:
                    mismatches+=1
            synonymous = True if aa1 == aa2 else False
            return (synonymous,mismatches) 
            
    def synonymous_codon_by_max_score(self,codon,scores):
        
        if 'N' in codon or 'R' in codon:
            return codon
        else:
            aa = self.codon_to_aa(codon)
            codon_list = self.aa_to_codon(aa)

            max_score = sum(scores)
            best_codon = codon
            for alt_codon in codon_list: 
                candidate_score = 0.0
                for base_curr,base_alt,s in zip(codon,alt_codon,scores):
                    if base_alt == base_curr:
                        candidate_score +=s
                    else:
                        candidate_score -= s 
                if candidate_score > max_score:
                    max_score = candidate_score
                    best_codon = alt_codon

            return best_codon

    def synonymous_codon_by_min_score(self,codon,scores):
        
        if 'N' in codon or 'R' in codon:
            return codon
        else:
            aa = self.codon_to_aa(codon)
            codon_list = self.aa_to_codon(aa)

            min_score = sum(scores)
            best_codon = codon
            for alt_codon in codon_list: 
                candidate_score = 0.0
                for base_curr,base_alt,s in zip(codon,alt_codon,scores):
                    if base_alt == base_curr:
                        candidate_score +=s
                    else:
                        candidate_score -= s 
                if candidate_score < min_score:
                    min_score = candidate_score
                    best_codon = alt_codon

            return best_codon

class AttachClassLabel(Transform):
    '''Pre-pend class label based on FASTA sequence '''
    def apply(self, example, is_train=False, stats=None, **kwargs):
        
        curr_tgt = example['tgt'] 
        if curr_tgt[0] == '[NONE]':
            example['tgt'] = ['<NC>']
        else:
            example['tgt'] = ['<PC>'] + curr_tgt
        return example

class OmitPeptide(Transform):
    '''Remove amino acid sequence'''
    
    def apply(self, example, is_train=False, stats=None, **kwargs):
        
        curr_tgt = example['tgt']
        example['tgt'] = [curr_tgt[0]]
        return example

class ShuffleCopies(Transform):
    '''Create fully shuffled copies of src'''
    
    def apply(self, example, is_train=False, stats=None, **kwargs):
        
        src = example['src']
        num_copies = 50
        src_shuffled = [] 
        for i in range(num_copies):
            s = src.copy()
            random.shuffle(s)
            seq_name = f'src_shuffled_{i}'
            example[seq_name] = {seq_name : ' '.join(s)}
        return example

class GenPointMutations(Transform):
    '''Randomly swap synonymous codons inside the longest ORF'''
   
    def apply(self, example, is_train=False, stats=None, **kwargs):
        
        src = example['src']
        src_shuffled = []
        s = ''.join(src) 
        start,end = getLongestORF(''.join(s))
      
        rel_start = -12
        rel_end = 60
        abs_start = start+rel_start
        abs_end = start+rel_end
        
        if abs_start >=0 and abs_end <=end:
            for base in ['A','C','G','T']:
                for abs_loc,rel_loc in zip(range(abs_start,abs_end),range(-12,60)):
                    c = src[abs_loc]
                    if base != c:
                        s_mutated = src[:abs_loc]+[base]+src[abs_loc:] 
                        seq_name = f'src_{rel_loc}->{base}'
                        example[seq_name] = {seq_name : ' '.join(s_mutated)}
        return example

class SynonymousCopies(Transform):
    '''Randomly swap synonymous codons inside the longest ORF'''
   
    def apply(self, example, is_train=False, stats=None, **kwargs):
        
        src = example['src']
        num_copies = self.opts.num_copies
        src_shuffled = []
        s = ''.join(src) 
        start,end = getLongestORF(s)
        t = CodonTable()    
        threshold = self.opts.mutation_prob if hasattr(self.opts,'mutation_prob') else 1.0
        
        for n in range(num_copies):
            # traverse ORF
            s_mutated = src[:start]
            for i in range(start,end,3):
                c = str(s[i:i+3])
                p = np.random.rand()
                if p < threshold:
                    alt = t.random_synonymous_codon(c)
                    s_mutated.extend(list(alt))
                else:
                    s_mutated.extend(list(c))
            s_mutated+=src[end:]
            seq_name = f'src_shuffled_{n}'
            example[seq_name] = {seq_name : ' '.join(s_mutated)}
        
        return example
