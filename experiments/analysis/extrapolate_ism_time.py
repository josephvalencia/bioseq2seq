import pandas as pd

def format_time(minutes):
    
    days = int(minutes // (24*60))
    remainder = minutes % (24*60)
    hours = int(remainder // (60))
    remainder2 = remainder % (60)
    minutes = int(round(remainder2))
    return f'{days}D:{hours}Hr:{minutes}m'

# was run on one GPU
verified_data_size = 440
verified_ism_time = 7.5 * 60 
# was split over four GPUs
train_pred_time = 9 * 4

df = pd.read_csv('data/mammalian_200-1200_train_balanced.csv',sep='\t')
print(df.groupby('Type').count())
total_ism_evals = 3 * sum([len(x) for x in df['RNA'].tolist()])
print(f'Total test set forward passes = {total_ism_evals}')

# extrapolate based on our small ISM sample
estimate_ism_based = verified_ism_time * len(df) / verified_data_size 
# extrapolate based on full train set prediction-only time , i.e. assume improvements in parallelism
estimate_pred_based = train_pred_time * total_ism_evals / len(df)
print(f'Extrapolating from train set pred-only = {format_time(estimate_pred_based)}, Extrapolating from small ISM test set = {format_time(estimate_ism_based)}')
