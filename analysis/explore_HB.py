from math import ceil,log

max_iter = 20000  # maximum iterations/epochs per configuration
eta = 3 # defines downsampling rate (default=3)
logeta = lambda x: log(x)/log(eta)
s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

print(f'max_iter = {max_iter} , s_max = {s_max} , B = {B}')
#### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
for s in reversed(range(s_max+1)):
    n = int(ceil(int(B/max_iter/(s+1))*eta**s)) # initial number of configurations
    r = max_iter*eta**(-s) # initial number of iterations to run configurations for
    print(f'n : {n} r: {r} s={s}')
    print("---------------------")
    #### Begin Finite Horizon Successive Halving with (n,r)
    for i in range(s+1):
        # Run each of the n_i configs for r_i iterations and keep best n_i/eta
        n_i = n*eta**(-i)
        r_i = r*eta**(i)
        keep_best = int(n_i/eta)
        print(f'n_{i} : {n_i} r_{i} : {r_i} keep_best : {keep_best}')
    print("---------------------")
