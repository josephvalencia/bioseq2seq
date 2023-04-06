from Bio import motifs
import sys,re,os
import numpy as np

''' This script is modified from https://github.com/vagarwal87/saluki_paper/blob/main/Fig5_S6/modisco2meme.py '''

def parse_cisbprna(filename):
    
    storage = []
    with open(filename) as inFile:
        data = inFile.read()
        entries = data.split('\n\n')
        for entry in entries:
            result = process_entry(entry)
            if 'PWM' in result: 
                storage.append(result)
    
    out_file = filename.replace('.txt','.meme') 
    write_to_meme(storage,out_file)

def process_entry(entry):

    results = {}
    fields = ['RBP','RBP Name','Gene','Motif','Family','Species']
    field_regex = r'RBP\t(.*)\nRBP Name\t(.*)\nGene\t(.*)\nMotif\t(.*)\nFamily\t(.*)\nSpecies\t(.*)'
    match = re.search(field_regex,entry)
    if match:
        for f,g in zip(fields,match.groups()): 
            results[f] = g
        remainder = entry[match.end():]
        pwm = process_pwm(remainder)
        if pwm is not None:
            results['PWM'] = pwm
    return results

def process_pwm(remainder):

    lines = remainder.split('\n')
    storage = []
    if len(lines) > 2: 
        for l in lines[2:]:
            fields = [float(x) for x in l.split('\t')[1:]]
            storage.append(fields)
        return np.stack(storage)
    else:
        return None

def write_to_meme(all_pwms,meme_file):
    # save to meme
    with open(meme_file,'w+') as f:
        f.write('MEME version 5\n')
        f.write('\n')
        f.write('ALPHABET= ACGT\n')
        f.write('\n')
        f.write('strands: + -\n')
        f.write('\n')
        f.write('Background letter frequencies\n')
        f.write('A 0.27 C 0.23 G 0.23 T 0.27\n')
        f.write('\n')

    for entry in all_pwms:
        with open(meme_file,'a') as f:
            alt_name = ' {}_{}_{}'.format(entry['RBP Name'],entry['Species'],entry['RBP'])
            f.write('MOTIF '+entry['Motif']+alt_name+'\n')
            f.write('letter-probability matrix: alength= 4 w= {}\n'.format(entry['PWM'].shape[0]))
        with open(meme_file,'ab') as f:
            print(entry['PWM'].shape,entry['PWM'].sum(axis=1)) 
            np.savetxt(f, entry['PWM'],fmt='%.6f')
        with open(meme_file,'a') as f:
            f.write('\n')

if __name__ == "__main__":
    
    parse_cisbprna(sys.argv[1])
