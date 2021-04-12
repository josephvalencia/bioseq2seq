import sys
import re
from bioseq2seq.bin.evaluator import Evaluator
import pandas as pd

def process(filename,query_file,db_file):

    evaluator = Evaluator() 
    
    query_df = pd.read_csv(query_file,sep="\t").set_index("ID")
    db_df = pd.read_csv(db_file,sep="\t").set_index("ID")
   
    with open(filename) as inFile:
        results = inFile.read()

    storage = []
    entries = re.split("# BLASTN",results)

    for entry in entries:
        lines = entry.split("\n")
        test  = re.search("# Query: (\w*.\d) ",entry)        
        
        match = test.group(1) if not test is None else None

        for l in lines:
            if l.startswith("NR") or l.startswith("XR") or l.startswith("NM") or l.startswith("XM"):
                match = l
                break
        if not match is None:
            storage.append(match)

    for line in storage:
        fields = line.split("\t")
        if len(fields) > 1:
            query,subject = fields[0],fields[1]
            query_seq = query_df.loc[query,'RNA']
            db_seq = db_df.loc[subject,'RNA']
            match,total = evaluator.emboss_needle(query_seq,db_seq)
            print(query+ "\t"+str(match/total))    
        else:
            print(line+"\t0.0")

if __name__ == "__main__":
    
    process(sys.argv[1],sys.argv[2],sys.argv[3])
