import numpy as np
import sys,re,os

def combine(parent,model_string):

    storage = {}
    print(parent)
    print(model_string)
    portions = [os.path.join(parent,x) for x in os.listdir(parent) if x.startswith(model_string)]
    print(portions)
    for portion in portions:
        print(f'loading {portion}')
        saved = np.load(portion)
        for tscript,attr in saved.items():
            storage[tscript] = attr
    
    filename = os.path.join(parent,model_string) 
    print(f'combined file saved at {filename}')
    np.savez_compressed(filename,**storage)

if __name__ == "__main__":

    combine(sys.argv[1],sys.argv[2])
