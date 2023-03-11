#! /usr/bin/env python

# altschulEriksonDinuclShuffle.py
# P. Clote, Oct 2003
# NOTE: One cannot use function "count(s,word)" to count the number
# of occurrences of dinucleotide word in string s, since the built-in
# function counts only nonoverlapping words, presumably in a left to
# right fashion.


import sys,string,random
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import re

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

def computeCountAndLists(s):
  #WARNING: Use of function count(s,'TT') returns 1 on word TTT
  #since it apparently counts only nonoverlapping words TT
  #For this reason, we work with the indices.

  #Initialize lists and mono- and dinucleotide dictionaries
  List = {} #List is a dictionary of lists
  List['A'] = []; List['C'] = [];
  List['G'] = []; List['T'] = [];
  List['N'] = []; List['R'] = [];
  nuclList   = ["A","C","G","T","N","R"]
  s       = s.upper()
  nuclCnt    = {}  #empty dictionary
  dinuclCnt  = {}  #empty dictionary
  for x in nuclList:
    nuclCnt[x]=0
    dinuclCnt[x]={}
    for y in nuclList:
      dinuclCnt[x][y]=0

  #Compute count and lists
  nuclCnt[s[0]] = 1
  nuclTotal     = 1
  dinuclTotal   = 0
  for i in range(len(s)-1):
    x = s[i]; y = s[i+1]
    List[x].append( y )
    nuclCnt[y] += 1; nuclTotal  += 1
    dinuclCnt[x][y] += 1; dinuclTotal += 1
  assert (nuclTotal==len(s))
  assert (dinuclTotal==len(s)-1)
  return nuclCnt,dinuclCnt,List
 
 
def chooseEdge(x,dinuclCnt):
  numInList = 0
  for y in ['A','C','G','T']:
    numInList += dinuclCnt[x][y]
  z = random.random()
  denom=dinuclCnt[x]['A']+dinuclCnt[x]['C']+dinuclCnt[x]['G']+dinuclCnt[x]['T']
  numerator = dinuclCnt[x]['A']
  if z < float(numerator)/float(denom):
    dinuclCnt[x]['A'] -= 1
    return 'A'
  numerator += dinuclCnt[x]['C']
  if z < float(numerator)/float(denom):
    dinuclCnt[x]['C'] -= 1
    return 'C'
  numerator += dinuclCnt[x]['G']
  if z < float(numerator)/float(denom):
    dinuclCnt[x]['G'] -= 1
    return 'G'
  dinuclCnt[x]['T'] -= 1
  return 'T'

def connectedToLast(edgeList,nuclList,lastCh):
  D = {}
  for x in nuclList: D[x]=0
  for edge in edgeList:
    a = edge[0]; b = edge[1]
    if b==lastCh: D[a]=1
  for i in range(2):
    for edge in edgeList:
      a = edge[0]; b = edge[1]
      if D[b]==1: D[a]=1
  ok = 0
  for x in nuclList:
    if x!=lastCh and D[x]==0: return 0
  return 1

 

def eulerian(s):
  nuclCnt,dinuclCnt,List = computeCountAndLists(s)
  #compute nucleotides appearing in s
  nuclList = []
  for x in ["A","C","G","T"]:
    if x in s: nuclList.append(x)
  #compute numInList[x] = number of dinucleotides beginning with x
  numInList = {}
  for x in nuclList:
    numInList[x]=0
    for y in nuclList:
      numInList[x] += dinuclCnt[x][y]
  #create dinucleotide shuffle L 
  firstCh = s[0]  #start with first letter of s
  lastCh  = s[-1]
  edgeList = []
  for x in nuclList:
    if x!= lastCh: edgeList.append( [x,chooseEdge(x,dinuclCnt)] )
  ok = connectedToLast(edgeList,nuclList,lastCh)
  return ok,edgeList,nuclList,lastCh


def shuffleEdgeList(L):
  n = len(L); barrier = n
  for i in range(n-1):
    z = int(random.random() * barrier)
    tmp = L[z]
    L[z]= L[barrier-1]
    L[barrier-1] = tmp
    barrier -= 1
  return L

def dinuclShuffle(s):
  ok = 0
  while not ok:
    ok,edgeList,nuclList,lastCh = eulerian(s)
  nuclCnt,dinuclCnt,List = computeCountAndLists(s)

  #remove last edges from each vertex list, shuffle, then add back
  #the removed edges at end of vertex lists.
  for [x,y] in edgeList: List[x].remove(y)
  for x in nuclList: shuffleEdgeList(List[x])
  for [x,y] in edgeList: List[x].append(y)

  #construct the eulerian path
  L = [s[0]]; prevCh = s[0]
  for i in range(len(s)-2):
    ch = List[prevCh][0] 
    L.append( ch )
    del List[prevCh][0]
    prevCh = ch
  L.append(s[-1])
  t = ''.join(L)
  return t

def safe_dinuc_shuffle(cds):

    stops = {'TAG','TAA','TGA'}
    if len(cds) > 0:
        start_codon = cds[:3]
        stop_codon = cds[-3:]
        unrestricted = cds[3:-3] 
        unrestricted = dinuclShuffle(unrestricted) if len(unrestricted) > 0 else '' 
        edited = [] 
        count = 0
        for i in range(0,len(unrestricted),3):
            codon = unrestricted[i:i+3]
            if codon in stops:
                edited.append('CTG')
                count+=1
            else:
                edited.append(codon)
        cds = start_codon + ''.join(edited) + stop_codon 
    return cds

def safe_codon_shuffle(cds):

    stops = {'TAG','TAA','TGA'}
    if len(cds) > 0:
        start_codon = cds[:3]
        stop_codon = cds[-3:]
        unrestricted = cds[3:-3] 
        edited = [] 
        for i in range(0,len(unrestricted),3):
            codon = unrestricted[i:i+3]
            edited.append(codon)
        random.shuffle(edited) 
        cds = start_codon + ''.join(edited) + stop_codon 
    return cds

def safe_mononuc_shuffle(cds):

    stops = {'TAG','TAA','TGA'}
    if len(cds) > 0:
        start_codon = cds[:3]
        stop_codon = cds[-3:]
        unrestricted = cds[3:-3] 
        edited = [] 
        for i in range(0,len(unrestricted),3):
            codon = unrestricted[i:i+3]
            edited.append(codon)
        unrestricted = ''.join(edited) 
        edited = [] 
        count = 0
        for i in range(0,len(unrestricted),3):
            codon = unrestricted[i:i+3]
            if codon in stops:
                edited.append('CTG')
                count+=1
            else:
                edited.append(codon)
        cds = start_codon + ''.join(edited) + stop_codon 
    return cds

def shuffle_all_fasta(fasta_file,mode,k,num_shuffles=1):

    storage = []
    for record in SeqIO.parse(fasta_file,'fasta'):
        tscript = record.id
        for i in range(num_shuffles):
            src = str(record.seq) 
            if mode == '5-prime' or mode == '3-prime':
                s,e = getLongestORF(src)
                fiveprime = src[:s]
                cds = src[s:e]
                threeprime = src[e:]
                if len(fiveprime) > 0 and mode == '5-prime':
                    fiveprime =  dinuclShuffle(fiveprime)
                if len(threeprime) > 0 and mode == '3-prime':
                    threeprime = dinuclShuffle(threeprime)
                shuffled = fiveprime + cds + threeprime
            elif mode == 'CDS':
                s,e = getLongestORF(src)
                fiveprime = src[:s]
                cds = src[s:e]
                threeprime = src[e:]
                shuffle_fn = safe_dinuc_shuffle 
                if k == 1:
                    shuffle_fn = safe_mononuc_shuffle
                elif k == 3:
                    shuffle_fn = safe_codon_shuffle
                shuffled = fiveprime + shuffle_fn(cds) + threeprime
                s2,e2 = getLongestORF(shuffled)
            else:
                shuffled = dinuclShuffle(src)
            variant_name=f'{tscript}-{k}-nuc_shuffled_{mode}-{i+1}'
            record = SeqRecord(Seq(shuffled),
                                id=variant_name)
            storage.append(record)
   
    fields = fasta_file.split('.')
    saved_name = ''.join(fields[:-1]) +f'_{k}-nuc_shuffled_{mode}.fa'
     
    with open(saved_name,'w') as outFile:
        SeqIO.write(storage, outFile, "fasta")
    print(f'saved {saved_name}')
if __name__ == '__main__':

    shuffle_all_fasta(sys.argv[1],sys.argv[2],sys.argv[3])
