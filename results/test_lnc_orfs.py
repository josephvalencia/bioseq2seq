import subprocess,sys,shlex
from Bio import SeqIO

def emboss_getorf(nucleotide_seq):

    '''Find longest Open Reading Frame (START-STOP) using getorf from needle package.
    See http://bioinf-hpc.ibun.unal.edu.co/cgi-bin/emboss/help/getorf'''

    cmd = "getorf -sequence={} -find=1 -noreverse -stdout -auto".format("asis::"+nucleotide_seq)
    response = subprocess.check_output(shlex.split(cmd),universal_newlines=True)

    orfs = response.rstrip().split("\n")
    orfs = [x.rstrip() for x in orfs if not x.startswith(">asis")]

    # ORFs sorted by size descending
    orfs = sorted(orfs,key = lambda x : len(x),reverse = True)
    return orfs

if __name__ == "__main__":

    RNA = sys.argv[1]

    longest_lengths = []

    for record in SeqIO.parse(RNA,"fasta"):

        orfs = emboss_getorf(record.seq)
        longest_lengths.append(len(orfs[0]))

        #print("lncRNA: {}\n".format(record.seq))
        #for i,o in enumerate(orfs):
        #    print("ORF {} : {}".format(i,o))
        #print("")

    avg_len = sum(longest_lengths) / len(longest_lengths)

    print("AVG_LEN : {}".format(avg_len))
