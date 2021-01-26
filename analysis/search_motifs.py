import sys, re
from Bio import motifs

class Motif:

    def __init__(self):

        self.data = []

    def update(self,entry):

        weights = [float(x) for x in entry.split()]
        alphabet = ['A','C','G','T']
        scores = {c : w for c,w in zip(alphabet,weights)}
        self.data.append(scores)

    def score_word(self,word):

        assert len(word) == len(self.data)

        score = 0.0
        for i,c in enumerate(word):
            score += self.data[i][c]
        
        print(score)

    def score(self,seq):

        scores = []
        step = len(self.data)
        
        for i in range(0,len(seq)-step):
            word = seq[i:i+step]
            s = self.score_word(word)
            scores.append(s)
        
        return scores

def make_motifs(filename):

    with open(filename) as inFile:
        
        motif_list = []
        lines = re.split(r'MOTIF',inFile.read())

        for l in lines[1:]:
            rows = l.split('\n')
            width = re.search(r'w= (\d*)',rows[1])
            if width is not None:
                w = int(width.group(1))
                motif = Motif()
                for r in rows[2:2+w]:
                   motif.update(r) 
                motif_list.append(motif)
                
    return motif_list

if __name__ == "__main__":

    meme_file = sys.argv[1]
    search_motifs(meme_file)
