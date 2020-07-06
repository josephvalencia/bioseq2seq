import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

inFile = sys.argv[1]
df = pd.read_csv(inFile,names=["TSCRIPT","ID"])
align_ids = np.asarray(df["ID"].values)

plt.hist(align_ids,density=True,bins=20)
plt.title("Old GENCODE Alignment ID")
plt.xlabel("% ID")
plt.ylabel("Density")
plt.savefig("align_id_hist_old.png")



