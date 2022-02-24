import pandas as pd
import re,sys
import random

def get_CDS_start(cds,rna):

    if cds != "-1": 
        splits = cds.split(":")
        clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
        splits = [clean(x) for x in splits]
        start,end = tuple([int(x) for x in splits])
    else:
        start,end = getLongestORF(rna)
    
    return start
        
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

def augment(filename):

    df = pd.read_csv(filename,sep='\t')
    rnas = df['RNA'].tolist()
    ids = df['ID'].tolist()
    cds = df['CDS'].tolist()
    df = df.set_index('ID')
    cds_list = [get_CDS_start(c,s) for c,s in zip(cds,rnas)]
    
    ids_with_upstream = [t for t,s in zip(ids,cds_list) if s>2]
    df_mutated = df.loc[ids_with_upstream]
    df = df.reset_index()
    rnas_with_upstream = df_mutated['RNA'].tolist()
    cds_unparsed = df_mutated['CDS'].tolist()
    cds_with_upstream = [get_CDS_start(c,s) for c,s in zip(cds_unparsed,rnas_with_upstream)]

    df_mutated = df_mutated.reset_index()
    nucs = ['A','C','G','T']
    all_data = [df]

    # single insertions
    for i in range(4):
        df_single_ins = df_mutated.copy()
        mutated_rnas = []
        mutated_ids = []
        for s,r,t in zip(cds_with_upstream,rnas_with_upstream,ids_with_upstream):
            idx = random.randint(0,s-1)
            char = random.choice(nucs)
            seq = insertion(r,idx,char)
            name = f'{t}-1ins-{idx}-{char}'
            assert len(seq) == len(r)+1
            mutated_rnas.append(seq)
            mutated_ids.append(name)
        df_single_ins['RNA'] = mutated_rnas
        df_single_ins['ID'] = mutated_ids
        all_data.append(df_single_ins)

    # single deletions
    for i in range(4):
        df_single_del = df_mutated.copy()
        mutated_rnas = []
        mutated_ids = []
        for s,r,t in zip(cds_with_upstream,rnas_with_upstream,ids_with_upstream):
            idx = random.randint(0,s-1)
            seq = deletion(r,idx)
            name = f'{t}-1del-{idx}'
            assert len(seq) == len(r) -1
            mutated_rnas.append(seq)
            mutated_ids.append(name)
        df_single_del['RNA'] = mutated_rnas
        df_single_del['ID'] = mutated_ids
        all_data.append(df_single_del)

    # double insertions
    for i in range(4):
        df_double_ins = df_mutated.copy()
        mutated_rnas = []
        mutated_ids = []
        for s,r,t in zip(cds_with_upstream,rnas_with_upstream,ids_with_upstream):
            idx = random.randint(0,s-2)
            char1 = random.choice(nucs)
            char2 = random.choice(nucs)
            name = f'{t}-2ins-{idx}-{char1+char2}'
            seq = insertion(r,idx,char1+char2)
            assert len(seq) == len(r) +2
            mutated_rnas.append(seq)
            mutated_ids.append(name)
        df_double_ins['RNA'] = mutated_rnas
        df_double_ins['ID'] = mutated_ids
        all_data.append(df_double_ins)

   # double deletions
    for i in range(4):
        df_double_del = df_mutated.copy()
        mutated_rnas = []
        mutated_ids = []
        for s,r,t in zip(cds_with_upstream,rnas_with_upstream,ids_with_upstream):
            idx = random.randint(0,s-2)
            name = f'{t}-2del-{idx}'
            seq = deletion(r,idx)
            seq = deletion(seq,idx)
            assert len(seq) == len(r) -2
            mutated_rnas.append(seq)
            mutated_ids.append(name)
        df_double_del['RNA'] = mutated_rnas
        df_double_del['ID'] = mutated_ids
        all_data.append(df_double_del)

    augmented_data = pd.concat(all_data).sample(frac=1.0)
    augmented_data.to_csv(filename+'.AUGMENTED',sep='\t',index=False)

def substitution(seq,i,nuc):
    return seq[:i]+nuc+seq[i+1:]

def insertion(seq,i,chars):
    return seq[:i]+chars+seq[i:]

def deletion(seq,i):
    return seq[:i]+seq[i+1:]

def parse_cds(loc):

    cds = loc.split(':')
    s = cds[0]
    e = cds[1]
    
    if s.startswith('<'):
        s = s[1:]
    if e.startswith('>'):
        e = e[1:]

    return int(s),int(e)

def swap_UTRs(df):
    
    n_has_5 = 0
    n_has_3 = 0
    n_canonical = 0

    for loc,rna in zip(df['CDS'].tolist(),df['RNA'].tolist()):
        if loc != "-1":
            s,e = parse_CDS(loc)
            has_5 =  s >0
            has_3 =  e < len(rna)-1
            if has_5:
                n_has_5+=1
            if has_3:
                n_has_3+=1
            if has_5 and has_3:
                n_canonical+=1

    return n_has_5, n_has_3, n_canonical
    
if __name__ == "__main__":

   augment(sys.argv[1]) 
