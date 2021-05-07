import re,sys
from Bio import SeqIO
'''
with open(sys.argv[1]) as inFile:

    nc=pc =0

    for l in inFile:
        match = re.search('(ID: )((\w)*.\d)',l)
        if match is not None:
            tscript = match.group(2)
            if tscript.startswith('XR_') or tscript.startswith('NR_'):
                nc+=1
            else:
                pc+=1
'''

with open(sys.argv[1]) as inFile:
    
    nc=pc=0
    for seq in SeqIO.parse(inFile,'fasta'):
        tscript = seq.id
        #print(tscript)
        if tscript.startswith('XR_') or tscript.startswith('NR_'):
            nc+=1
        else:
            pc+=1

print("total = {}, NC = {} , PC = {}".format(nc+pc,nc,pc))
