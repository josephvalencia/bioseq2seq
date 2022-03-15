import ConfigSpace as CS  
import argparse
from argparse import Namespace

import ray
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback

from bioseq2seq.bin.train_utils import train_seq2seq, parse_train_args

def base_config():
    ''' hyperparameters that are independent of architecture''' 
    
    config = {"model_dim": tune.choice([32,64,128]),
        "n_enc_layers": tune.choice([1,2,4,8,12,16]),
        "n_dec_layers": tune.choice([1,2,4,8,12,16]),
        "dropout" : tune.quniform(0.1,0.5,0.1),
        "lr_warmup_steps" : tune.quniform(2000,10000,2000)}
    return config

def cnn_config():
    ''' hyperparameters for CNN+CNN architecture''' 
    
    config = {"dilation_multipler" : tune.choice([1,2]),
            "kernel_size" : tune.choice([3,6])}
    return config

def mixer_config():
    ''' hyperparameters for GFNet+Transformer architecture''' 
    
    config = {"filter_size" : tune.choice([50,100,500])}
    return config

def train_protein_coding_potential(config,cmd_args):
    ''' implements the Ray Tune API '''
    
    args = vars(cmd_args) 
    # override some cmd line args with Ray Tune
    args.update(config)
    args = Namespace(**args)
    train_seq2seq(args,tune=True)

if __name__ == "__main__":

    cmd_args = parse_train_args()
    config = base_config()
    model_config = mixer_config() if cmd_args.model_type == "GFNet" else cnn_config() 
    config.update(model_config)
    
    ray.init(num_cpus=32,_temp_dir='/tmp/ray_tune') 
    bohb_hyperband = HyperBandForBOHB(time_attr="valid_step",
                                        max_t=10000,
                                        reduction_factor=4,
                                        stop_last_trials=False)
    bohb_search = TuneBOHB()

    name = f"BOHB_search{cmd_args.model_type}"
    train_wrapper = tune.with_parameters(train_protein_coding_potential,cmd_args = cmd_args)
    
    wandb_callback = WandbLoggerCallback(project="GFNET Hyperparam Search",api_key="415f7e8e731deb5b2aeb354cb62f72a2c4556657",log_config=False)
    total_time_allowed = 10*24*60*60 # 10 days in seconds 
    analysis = tune.run(train_wrapper,
                        name=name,
                        config=config,
                        scheduler=bohb_hyperband,
                        search_alg=bohb_search,
                        num_samples=20,
                        time_budget_s=total_time_allowed,
                        resources_per_trial={'gpu': 0, 'cpu' : 1},
                        stop={"valid_step": 10000},
                        metric="valid_class_accuracy",
                        mode="max",
                        callbacks=[wandb_callback])
    
    print("Best hyperparameters found were: ", analysis.best_config)

