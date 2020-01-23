import pandas as pd
import sys

with open(sys.argv[1]) as inFile:
    data = inFile.read()

lines = data.split("\n")[:-1]
pcts = []
print(lines)
for i in range(0,len(lines),3):
    pcts.append(float(lines[i]))

df = pd.DataFrame()
df['id'] = pcts

print("Summary of {} ".format(sys.argv[1]))
print(df.describe())
