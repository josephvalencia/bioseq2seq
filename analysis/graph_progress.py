import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def smooth_array(array,window_size):

    half_size = window_size // 2
    running_sum = sum(array[:half_size])
    smoothed_scores = [0.0]*len(array)

    print("running_sum",running_sum)

    trail_idx = 0
    lead_idx = half_size

    for i in range(len(array)):
        gap_size = lead_idx - trail_idx + 1
        smoothed_scores[i] = running_sum / gap_size

        # advance lead until it reaches end
        if lead_idx < len(array)-1:
            running_sum += array[lead_idx]
            lead_idx +=1

        # advance trail when the gap is big enough or the lead has reached the end        
        if gap_size == window_size or lead_idx == len(array) -1:
            running_sum -= array[trail_idx]
            trail_idx+=1
    
    return smoothed_scores


trials_data = sys.argv[1]

df = pd.read_csv(trials_data)
step = df[" step"].values.tolist()
train = df[" train_accuracy"].values.tolist()
val = df[" val_accuracy"].values.tolist()

plt.plot(step,train,label="train")
plt.plot(step,val,label="validation")

plt.title("ED_classifier  Accuracy")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(trials_data.split(".")[0]+".eps")
