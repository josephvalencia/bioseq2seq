import os
import json

def convert(saved_file,tgt_field,mode="attn"):

    id_field = "TSCRIPT_ID" if mode == "attn" else "ID"

    storage = []

    # delete pre-existing files
    with open(saved_file) as inFile:
        first = inFile.readline()
        fields = json.loads(first)
        keys = list(fields.keys())
        tscript = keys.pop(0)
        for head in keys:
            tab_file = saved_file.split(".")[0]+"."+head+".enc_dec_attns.tabs"
            if os.path.isfile(tab_file):
                os.remove(tab_file)

    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            keys  = list(fields.keys())
            id  = keys.pop(0)
            for head in keys:
                tab_file = saved_file.split(".")[0]+"."+head+".enc_dec_attns.tabs"
                tscript = fields[id]
                array = fields[head]
                with open(tab_file,'a') as outFile:
                    for i,s in enumerate(array):
                        outFile.write("{}\t{}\t{}\n".format(tscript,i,s))

if __name__ == "__main__":
   
    for l in range(4):
        saved_file = "best_ED_classify/best_ED_classify_layer"+str(l)+".enc_dec_attns"
        for h in range(8):
            tgt_field = "layer{}head{}".format(l,h)
            convert(saved_file,tgt_field,"attn")

