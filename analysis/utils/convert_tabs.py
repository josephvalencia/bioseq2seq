import os
import orjson

def convert_enc_dec(saved_file,tgt_field):

    id_field = "TSCRIPT_ID" 

    # delete pre-existing files
    with open(saved_file) as inFile:
        first = inFile.readline()
        fields = orjson.loads(first)
        keys = list(fields.keys())
        tscript = keys.pop(0)
        for head in keys:
            tab_file = saved_file.split(".")[0]+"."+head+".enc_dec_attns.tabs"
            if os.path.isfile(tab_file):
                os.remove(tab_file)

    with open(saved_file) as inFile:
        for l in inFile:
            fields = orjson.loads(l)
            keys  = list(fields.keys())
            id  = keys.pop(0)
            for head in keys:
                tab_file = saved_file.split(".")[0]+"."+head+".enc_dec_attns.tabs"
                tscript = fields[id]
                array = fields[head]
                with open(tab_file,'a') as outFile:
                    for i,s in enumerate(array):
                        outFile.write("{}\t{}\t{}\n".format(tscript,i,s))

def convert_IG(saved_file):

    id_field = "ID"

    # delete pre-existing files
    with open(saved_file) as inFile:
        for head in ['normed_attr','summed_attr']:
            tab_file = saved_file+"."+head+".tabs"
            if os.path.isfile(tab_file):
                os.remove(tab_file)

    with open(saved_file,'r') as inFile:
        for l in inFile:
            fields = orjson.loads(l)
            for head in ['summed_attr','normed_attr']:
                tab_file = saved_file+"."+head+".tabs"
                tscript = fields[id_field]
                array = fields[head]
                with open(tab_file,'a') as outFile:
                    for i,s in enumerate(array):
                        outFile.write("{}\t{}\t{}\n".format(tscript,i,s/1000))

if __name__ == "__main__":
    
    for l in range(4):
        saved_file = "results/val/best_seq2seq/best_seq2seq_layer"+str(l)+".enc_dec_attns"
        for h in range(8):
            tgt_field = "layer{}head{}".format(l,h)
            convert_enc_dec(saved_file,tgt_field)

    quit()
    bases = ['A','C','G','T','avg','zero']
    
    ED_prefix = "results/best_ED_classify/best_ED_classify_"
    ED_files = [ED_prefix+b +'_pos.ig' for b in bases] 
    for f in ED_files:
        convert_IG(f)

    seq_prefix = "results/best_seq2seq/seq2seq_3_"
    seq_files = [seq_prefix+b+'_pos.ig' for b in bases]
    for f in seq_files:
        convert_IG(f)
