#!/usr/bin/env python
import random
import sys
from Bio import SeqIO
from scipy import mean, std

if len(sys.argv) < 3 or "-h" in sys.argv or "--help" in sys.argv:
    sys.stderr.write("Script breaks a dataset into a n-sized sample dataset with equal length distributions. Does not provide the remaining portion after sampling. To use this in a train/test pipeline, you would separate out your test set first, then pull out data from this dataset. This is logical since you want your test set to match the data distriubtion while the train set can be whatever you want.\n")
    sys.stderr.write("\n")
    sys.stderr.write("generate_train_test.py\t<transcripts1.fa>\t<transcripts2.fa>\t<num_sample>\n")
    sys.exit()

"""
Given two distributions, sort and index by length

Choose a sample from A, choose an identical sample from B
if there is no such example, choose the nearest (+1 -1)
repeat until finished
"""

prefix_a = ".".join(sys.argv[1].split(".")[:-1])
prefix_b = ".".join(sys.argv[2].split(".")[:-1])
cur_title = ""

def load_dataset(filename):
    cur = ""
    seqs = []
    dataset = {}
    for line in open(filename, 'r'):
        if ">" in line:
            if cur != "":
                key = len(cur)
                if key not in dataset:
                   dataset[key] = []
                dataset[key].append((cur_name, cur))
                seqs.append((cur_name, cur))
                cur = ""
            cur_name = line.strip()
        else:
            cur += line.strip()
    dataset[key].append((cur_name, cur))
    seqs.append((cur_name, cur))
    return dataset, seqs

def load_dataset_v2(filename):
    seqs = []
    dataset = defaultdict(list)

    for r in SeqIO.parse(filename,'fasta'):
        key = len(r.seq)
        dataset[key].append((r.id,r.seq))
        seqs.append(r.id,r.seq)
    
    return dataset,seqs

def match_length_distribution(sample_keys,dataset_b):
    #list of keys
    samples_b = []
    for key in sample_keys:
        found_neighbor = False
        sign = 1
        dist = 1
        s_key = key
        while not found_neighbor:
            try:
                samples_b.append(dataset_b[key][0])
                found_neighbor = True
                del dataset_b[key][0]
                if len(dataset_b[key]) == 0:
                    del dataset_b[key]
            except KeyError:
                key += dist * sign
                dist += 1
                sign *= -1
    return samples_b

dataset_a, seqs = load_dataset_v2(sys.argv[1])
dataset_b, _ = load_dataset(sys.argv[2])
print(len(dataset_a),len(dataset_b))
quit()
samples_a = random.sample(seqs, len(seqs)) #list of keys
sample_keys = [len(seq) for name, seq in samples_a]

samples_b = match_length_distribution(sample_keys,dataset_b)
print("means")
print('A',mean([len(seq) for name, seq in samples_a]))
print('B',mean([len(seq) for name, seq in samples_b]))
print("stds")
print('A',std([len(seq) for name, seq in samples_a]))
print('B',std([len(seq) for name, seq in samples_b]))
print("max")
print('A',max([len(seq) for name, seq in samples_a]))
print('B',max([len(seq) for name, seq in samples_b]))
print("min")
print('A',min([len(seq) for name, seq in samples_a]))
print('B',min([len(seq) for name, seq in samples_b]))


fd = open(prefix_a + ".a.samples"+".fa", 'w')
for n, s in samples_a: 
    fd.write(n+'\n')
    fd.write(s+'\n')
fd.close()

fd = open(prefix_b + ".b.samples"+".fa", 'w')
for n, s in samples_b: 
    fd.write(n+'\n')
    fd.write(s+'\n')
fd.close()

