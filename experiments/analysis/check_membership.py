import pandas as pd

id_list = []

with open('lncPEP_ids.txt','r') as outFile:
    id_list = [x.rstrip() for x in outFile.readlines()]

print(id_list)
datasets = ["data/mammalian_200-1200_test.csv","data/mammalian_200-1200_val.csv","data/mammalian_200-1200_train_balanced.csv","data/mammalian_refseq.csv"]

for split in datasets:
    print(split)
    df = pd.read_csv(split,sep="\t")
    try: 
        overlap = df.loc[df['ID'].isin(id_list)]
        print(overlap) 
    except:
        print(f'None found in {split}')
