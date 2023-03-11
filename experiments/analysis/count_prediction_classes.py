import re,sys
import pandas as pd

def parse_predictions(record):
    
    transcript = record[0].split('ID: ')[1]
    pred_match  = re.search('PRED: (<PC>|<NC>)(\S*)?',record[1])
    
    if pred_match is not None:
        pred_class = pred_match.group(1)
        pred_peptide = pred_match.group(2)
    else:
        raise ValueError('Bad format')

    entry = {'ID' : transcript,
            'pred_class' : pred_class, 
            'pred_seq': pred_peptide}
    return entry

def evaluate(pred_file):

    storage = []
    with open(pred_file,"r") as inFile:
        lines = inFile.read().split("\n")
        for i in range(0,len(lines)-6,6):
            entry = parse_predictions(lines[i:i+6])
            storage.append(entry)

    df = pd.DataFrame(storage)
    print(df.groupby('pred_class').count())

if __name__ == "__main__":
    
    evaluate(sys.argv[1])
