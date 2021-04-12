import orjson
import sys

def parse(saved_file):

    print("ID")
    with open(saved_file) as inFile:
        for l in inFile:
            fields = orjson.loads(l)
            id_field = "TSCRIPT_ID"
            id = fields[id_field]
            print(id)

if __name__ == "__main__":

    parse(sys.argv[1])
