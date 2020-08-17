import sys, os
import json

def json2tabs(json_filename):

    with open(json_filename) as inFile:
        for l in inFile:
            fields = json.loads(l)
            tscript_id = fields["TSCRIPT_ID"]
            array = fields["layer_0_pos_0"] 

            # print tab-delimited attention for each transcript
            tscript_file = tscript_id+"_"+os.path.split(json_filename)[1]
            with open(tscript_file,'w') as outFile:
                for k,v in enumerate(array):
                    outFile.write("{}\t{}\n".format(k,v))
            print("Wrote to "+tscript_file)
                    
if __name__ == "__main__":

    json2tabs(sys.argv[1])