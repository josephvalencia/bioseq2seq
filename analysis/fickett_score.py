from Bio import SeqIO

def findPositionProbability(position_x, base):
    '''Calculate the Position probablity of a base in codon'''
    coding_probs = []
    codeProb = 0
    if (base == "A"):
        coding_probs = [.22, .20, .34, .45, .68, .58, .93, .84, .68, .94]
    if (base == "C"):
        coding_probs = [.23, .30, .33, .51, .48, .66, .81, .70, .70, .80]
    if (base == "G"):
        coding_probs = [.08, .08, .16, .27, .48, .53, .64, .74, .88, .90]
    if (base == "T"):
        coding_probs = [.09, .09, .20, .54, .44, .69, .68, .91, .97, .97]

    if (position_x >= 0 and position_x < 1.1):
        test = coding_probs[0]
        code_prob = coding_probs[0]
    elif (position_x >= 1.1 and position_x < 1.2):
        code_prob = coding_probs[1]
    elif (position_x >= 1.2 and position_x < 1.3):
        code_prob = coding_probs[2]
    elif (position_x >= 1.3 and position_x < 1.4):
        code_prob = coding_probs[3]
    elif (position_x >= 1.4 and position_x < 1.5):
        code_prob = coding_probs[4]
    elif (position_x >= 1.5 and position_x < 1.6):
        code_prob = coding_probs[5]
    elif (position_x >= 1.6 and position_x < 1.7):
        code_prob = coding_probs[6]
    elif (position_x >= 1.7 and position_x < 1.8):
        code_prob = coding_probs[7]
    elif (position_x >= 1.8 and position_x < 1.9):
        code_prob = coding_probs[8]
    elif (position_x > 1.9):
        code_prob = coding_probs[9]
    return code_prob

def readfile(filename):
    f = open(filename)
    sl = list(SeqIO.parse(f, 'fasta'))
    ids = []
    seqs = []
    for s in sl:
        id = s.id
        seq = str(s.seq)
        ids.append(id)
        seqs.append(seq.upper())

    return ids, seqs

def findContentProbability(position_x, base):
    ''' Find the composition probablity of base in codon '''
    coding_probs = []
    code_prob = 0
    if (base == "A"):
        coding_probs = [.21, .81, .65, .67, .49, .62, .55, .44, .49, .28]
    if (base == "C"):
        coding_probs = [.31, .39, .44, .43, .59, .59, .64, .51, .64, .82]
    if (base == "G"):
        coding_probs = [.29, .33, .41, .41, .73, .64, .64, .47, .54, .40]
    if (base == "T"):
        coding_probs = [.58, .51, .69, .56, .75, .55, .40, .39, .24, .28]

    if (position_x >= 0 and position_x < .17):
        code_prob = coding_probs[0]
    elif (position_x >= .17 and position_x < .19):
        code_prob = coding_probs[1]
    elif (position_x >= .19 and position_x < .21):
        code_prob = coding_probs[2]
    elif (position_x >= .21 and position_x < .23):
        code_prob = coding_probs[3]
    elif (position_x >= .23 and position_x < .25):
        code_prob = coding_probs[4]
    elif (position_x >= .25 and position_x < .27):
        code_prob = coding_probs[5]
    elif (position_x >= .27 and position_x < .29):
        code_prob = coding_probs[6]
    elif (position_x >= .29 and position_x < .31):
        code_prob = coding_probs[7]
    elif (position_x >= .31 and position_x < .33):
        code_prob = coding_probs[8]
    elif (position_x > .33):
        code_prob = coding_probs[9]
    return code_prob

def ficketTestcode(seq):
    ''' The driver function '''
    baseOne = [0, 0, 0, 0]
    baseTwo = [0, 0, 0, 0]
    baseThree = [0, 0, 0, 0]
    seq.upper()

    for pos_1, pos_2, pos_3 in zip(range(0, len(seq), 3), range(1, len(seq), 3), range(2, len(seq), 3)):

        # Base one
        if (seq[pos_1] == "A"):
            baseOne[0] = baseOne[0] + 1
        elif (seq[pos_1] == "C"):
            baseOne[1] = baseOne[1] + 1
        elif (seq[pos_1] == "G"):
            baseOne[2] = baseOne[2] + 1
        elif (seq[pos_1] == "T"):
            baseOne[3] = baseOne[3] + 1

        # Base two
        if (seq[pos_2] == "A"):
            baseTwo[0] = baseTwo[0] + 1
        elif (seq[pos_2] == "C"):
            baseTwo[1] = baseTwo[1] + 1
        elif (seq[pos_2] == "G"):
            baseTwo[2] = baseTwo[2] + 1
        elif (seq[pos_2] == "T"):
            baseTwo[3] = baseTwo[3] + 1

        # Base two
        if (seq[pos_3] == "A"):
            baseThree[0] = baseThree[0] + 1
        elif (seq[pos_3] == "C"):
            baseThree[1] = baseThree[1] + 1
        elif (seq[pos_3] == "G"):
            baseThree[2] = baseThree[2] + 1
        elif (seq[pos_3] == "T"):
            baseThree[3] = baseThree[3] + 1

    position_A = max(baseOne[0], baseTwo[0], baseThree[0]) / (min(baseOne[0], baseTwo[0], baseThree[0]) + 1)
    position_C = max(baseOne[1], baseTwo[1], baseThree[1]) / (min(baseOne[1], baseTwo[1], baseThree[1]) + 1)
    position_G = max(baseOne[2], baseTwo[2], baseThree[2]) / (min(baseOne[1], baseTwo[2], baseThree[2]) + 1)
    position_T = max(baseOne[3], baseTwo[3], baseThree[3]) / (min(baseOne[3], baseTwo[3], baseThree[3]) + 1)

    content_A = (baseOne[0] + baseTwo[0] + baseThree[0]) / len(seq)
    content_C = (baseOne[1] + baseTwo[1] + baseThree[1]) / len(seq)
    content_G = (baseOne[2] + baseTwo[2] + baseThree[2]) / len(seq)
    content_T = (baseOne[3] + baseTwo[3] + baseThree[3]) / len(seq)

    position_A_prob = findPositionProbability(position_A, "A")
    position_C_prob = findPositionProbability(position_C, "C")
    position_G_prob = findPositionProbability(position_G, "G")
    position_T_prob = findPositionProbability(position_T, "T")

    content_A_prob = findContentProbability(content_A, "A")
    content_C_prob = findContentProbability(content_C, "C")
    content_G_prob = findContentProbability(content_G, "G")
    content_T_prob = findContentProbability(content_T, "T")
    ficket_score = position_A_prob * .26 + content_A_prob * .11 + position_C_prob * .18 + content_C_prob * .12 + position_G_prob * .31 + content_G_prob * .15 + position_T_prob * .33 + content_T_prob * .14
    return ficket_score

if __name__ == '__main__':
    '''gives the verdict'''
    ids, seqs = readfile("test.fasta")
    for id, seq in zip (ids, seqs ):
        ficket_score = ficketTestcode(seq)
        if (ficket_score < .74):
            print("non-conding \nID: "+ id)
        elif (ficket_score >= .74 and ficket_score < .95):
            print("No decision \nID: " +id)
        elif (ficket_score >= .95):
            print("protein-coding \nID: " + id)
