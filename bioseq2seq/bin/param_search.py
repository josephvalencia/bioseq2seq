import optuna

def get_cnn_searchspace():
    
    ''' Return optuna.Trial object describing hyperparams for trial'''
   
    # Optimizer hyperparams
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True) # 4
    learning_rate_schedule = trial.suggest_categorical("learning_rate_scheduler", ["Vaswani","CosineAnnealing","ReduceOnPlateau"]) # 3

    # Encoder/Decoder hyperparams
    num_enc_layers = trial.suggest_int("num_enc_layers", 2,16,2) # 8 
    num_dec_layers = trial.suggest_int("num_dec_layers", 2,16,2) # 8
    model_dim = trial.suggest_int("model_dim", 32, 256, log=True) # 4
    enc_kernel_width = trial.suggest_int("enc_kernel_width",3,15,3) # 5
    dec_kernel_width = trial.suggest_int("dec_kernel_width",3,15,3) # 5
   
    
    hyperparams = {'learning_rate' : learning_rate,
                    'learning_rate_schedule' : learning_rate_schedule,
                    'num_enc_layers' : num_enc_layers,
                    'num_dec_layers' : num_dec_layers,
                    'model_dim' : model_dim,
                    'enc_kernel_width' : enc_kernel_width,
                    'dec_kernel_width' : dec_kernel_width}
    return hyperparams

    
    
'''
N_TRIALS = 50
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)
'''


for i in range(10):
    print(get_cnn_searchspace())
