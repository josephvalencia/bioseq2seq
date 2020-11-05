import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

trials_data = sys.argv[1]

df = pd.read_csv(trials_data)
sns.lineplot(data=df,x="Step",y="Value",hue="model",palette="tab10")
plt.title("ED_classifier Validation Accuracy")
plt.ylabel("Accuracy")
plt.savefig(trials_data.split(".")[0]+".pdf")
