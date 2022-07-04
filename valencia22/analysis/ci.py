from scipy import stats
import numpy as np

def test_significance(a,b,a_num,b_num,confidence=0.95):

    a_mean = np.mean(a)
    b_mean = np.mean(b)
    
    rounds = len(a)
    t_value = stats.t.ppf((1 + confidence) / 2.0, df=rounds- 1)
    a_sd = np.std(a,ddof=1)
    b_sd = np.std(b,ddof=1)
    factor = np.sqrt(a_sd**2/a_num+b_sd**2/b_num)
    diff = a_mean - b_mean
    return (diff-t_value*factor,diff+t_value*factor)

test_accuracies_a = [0.957,0.955,0.961,0.952,0.956]
test_accuracies_b = [0.937,0.942,0.944,0.933,0.939]
lower,upper = test_significance(test_accuracies_a,test_accuracies_b,4959,4959)
print(lower,upper)
test_mean = np.mean(test_accuracies)

confidence = 0.95  # Change to your desired confidence level

rounds = len(test_accuracies)
t_value = stats.t.ppf((1 + confidence) / 2.0, df=rounds- 1)

sd = np.std(test_accuracies, ddof=1)
se = sd / np.sqrt(rounds)

ci_length = t_value * se

ci_lower = test_mean - ci_length
ci_upper = test_mean + ci_length

print(ci_lower, ci_upper)
