import pandas as pd

id_list = []

with open('validated_lncPEP_ids.txt','r') as outFile:
    id_list = [x.rstrip().split('.')[0] for x in outFile.readlines()]

datasets = ["data/mammalian_200-1200_test.csv","data/mammalian_200-1200_val.csv","data/mammalian_200-1200_train_balanced.csv","data/mammalian_200-1200_train.csv"]

for split in datasets:
    print(split)
    df = pd.read_csv(split,sep="\t")
    try:
        # check if any version of transcript is in dataset
        overlap = df.loc[df['ID'].str.startswith(tuple(id_list))] 
        print(overlap) 
    except:
        print(f'None found in {split}')
