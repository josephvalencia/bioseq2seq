import ConfigSpace as CS  
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

def base_config():
    ''' hyperparameters that are independent of architecture''' 
    
    config = {"model_dim": tune.choice([32,64,128]),
        "n_enc_layers": tune.choice([1,2,4,8,16]),
        "n_dec_layers": tune.choice([1,2,4,8,16]),
        "learning_rate" : tune.qloguniform(1e-5,1e-1,1e-5),
        "dropout" : tune.quniform(0.1,0.5,0.1)}
    return config

def cnn_config():
    ''' hyperparameters for CNN+CNN architecture''' 
    
    config = {"dilation_multipler" : tune.choice([1,2]),
            "kernel_size" : tune.choice([3,6])}
    return config

def mixer_config():
    ''' hyperparameters for GFNet+Transformer architecture''' 
    
    config = {"filter_size" : tune.choice([50,100,500]),
             "lambda_L1" : tune.choice([0,1e-3,1e-2,1e-1,1])}
    return config

if __name__ == "__main__":

    model = "mixer"
    config = base_config()
    model_config = mixer_config() if model == "mixer" else cnn_config() 
    config.update(model_config)
    print(config)
    bohb_hyperband = HyperBandForBOHB(time_attr="training_iteration",
                                        max_t=100,
                                        reduction_factor=4,
                                        stop_last_trials=False)
    bohb_search = TuneBOHB()
    bohb_search = tune.suggest.ConcurrencyLimiter(bohb_search, max_concurrent=4)

    name = f"BOHB_search{model}"
    analysis = tune.run(MyTrainableClass,
                        name=name,
                        config=config,
                        scheduler=bohb_hyperband,
                        search_alg=bohb_search,
                        num_samples=10,
                        stop={"training_iteration": 100},
                        metric="class_accuracy",
                        mode="max")
    
    print("Best hyperparameters found were: ", analysis.best_config)

