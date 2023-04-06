import sys,os,re
from scipy.stats import pearsonr,spearmanr
import pandas as pd

def parse_entry(entry):

    if any([c == ' ' for c in entry]):
        return None
    else:
        return entry[0], float(entry[1])

def build_codon_table(filename):
   
    raw_frequencies = {}
    search_string = re.search('(.*)/(.*)_codon_table.txt',filename)
    if search_string is not None:
        species = search_string.group(2)
    else:
        species = filename

    with open(filename,'r') as inFile:
        fields = inFile.read().rstrip().replace('\n','\t').split('\t')
    
    for i in range(0,len(fields),3):
        entry = fields[i:i+3]
        result = parse_entry(entry)
        if result:
            raw_frequencies[result[0]] = result[1]
    return species,raw_frequencies

def make_vector(dict_a,dict_b):
    return [dict_a[x] for x,y in dict_b.items()] 

if __name__ == "__main__":
    
    parent = sys.argv[1]
    all_species = [os.path.join(parent,x) for x in os.listdir(parent)]
   
    tai_df = pd.read_csv('human_TAI_Tuller.csv')
    tai_dict = {x : y for x,y in zip(tai_df['Codon'],tai_df['Human'])} 
    storage = {}
    for f in all_species:
        species,freqs = build_codon_table(f)
        storage[species] = freqs
   
    human_freqs = storage['homo_sapiens']
    human_usage = [x for x in human_freqs.values()]
    tai_vector = make_vector(tai_dict,human_freqs)
    pearson = pearsonr(human_usage,tai_vector) 
    spearman = spearmanr(human_usage,tai_vector) 
    print(f'Corr with tAI, r= {pearson[0]}, rho= {spearman[0]}')

    for species,freqs in storage.items():
        species_usage = make_vector(freqs,human_freqs)
        pearson = pearsonr(human_usage,species_usage) 
        spearman = spearmanr(human_usage,species_usage) 
        print(f'Corr with {species}, r={pearson[0]}, rho = {spearman[0]}')



