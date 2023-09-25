import pandas as pd

def format_time(minutes):
    
    days = int(minutes // (24*60))
    remainder = minutes % (24*60)
    hours = int(remainder // (60))
    remainder2 = remainder % (60)
    minutes = int(round(remainder2))
    return f'{days}D:{hours}Hr:{minutes}m'

def estimate(test_ism_time,train_mdig_time):
    
    test_data_size = 4491
    df = pd.read_csv('data/mammalian_200-1200_train_balanced.csv',sep='\t')
    # extrapolate based on our small ISM sample
    estimate_ism_based = test_ism_time * len(df) / test_data_size 
    print(f'MDIG runtime was {format_time(train_mdig_time)}') 
    print(f'Extrapolating from small ISM test set {format_time(test_ism_time)} -> {format_time(estimate_ism_based)}')
    savings = estimate_ism_based / train_mdig_time
    print(f'Savings is a factor of {savings}')

# cascade 

test_ism = 22.33*4*60
train_mdig = 48*60 
print('CASCADE')
estimate(test_ism,train_mdig)

# eecs hpc
train_mdig = 8*4*60
test_ism = 19.5*60
print('EECS HPC')
estimate(test_ism,train_mdig)

