import orjson,json
import numpy as np

def combine(in_file_list,combined_file):

    storage = {}

    for f in in_file_list:
        with open(f) as inFile:
            for l in inFile:
                fields = orjson.loads(l)
                id = fields['ID']
                # IG has padding, strip it out
                src = fields['src'].split('<pad>')[0]
                scores = [float(x) for x in fields['summed_attr']]
                scores = np.asarray(scores[:len(src)])
                
                if id not in storage:
                    entry = {'ID' : id , 'summed_attr' : [scores] , 'src' : src} 
                    storage[id] = entry
                else :
                    storage[id]['summed_attr'].append(scores)

    chars = {0 :'A' , 1 :'C' , 2 : 'G' ,3 :'T'}

    with open(combined_file,'w') as outFile:
        for tscript,entry in storage.items():
            all_MDIG = np.stack(entry['summed_attr'])
            coding = tscript.startswith('XM') or tscript.startswith('NM')
            
            if coding:
                maxes = all_MDIG.max(axis=0)
                arg_maxes = all_MDIG.argmax(axis=0)
                top_base = [chars[x] for x in arg_maxes.tolist()]
                entry['summed_attr'] = maxes.tolist()
                entry['top_bases']  = ''.join(top_base)
            else:
                mins = all_MDIG.min(axis=0)
                arg_mins = all_MDIG.argmin(axis=0)
                top_base = [chars[x] for x in arg_mins.tolist()]
                entry['summed_attr'] = mins.tolist()
                entry['top_bases']  = ''.join(top_base)
            
            outFile.write(json.dumps(entry)+'\n') 

if __name__ == "__main__":
    
    seq_bases = ['A','C','G','T']
    seq_three_test = ['new_output/IG/seq2seq_3_'+b+'_pos_test.ig' for b in seq_bases]
    combine(seq_three_test,'max_MDIG.ig')
