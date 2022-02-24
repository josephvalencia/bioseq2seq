
def gen_all_mutations(rna):

    vocab = ['A','C','G','T']

    variant_list = [rna]
    
    for i,c in enumerate(rna):
        for nuc in vocab:
            if nuc != c:
                variant = rna[:i]+nuc+rna[i+1:]
                variant_list.append(variant)
    return variant_list



test_seq = 'ACGTCGTCTCCCCACG'

print(gen_all_mutations(test_seq))
        
