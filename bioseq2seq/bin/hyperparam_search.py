import ConfigSpace as CS  
import argparse
from argparse import Namespace
import os
import ray
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.stopper import CombinedStopper, TrialPlateauStopper, MaximumIterationStopper
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
    
    config = {"encoder_dilation_factor" : tune.choice([1,2]),
            "encoder_kernel_size" : tune.choice([3,6,9]),
            "decoder_dilation_factor" : tune.choice([1,2]),
            "decoder_kernel_size" : tune.choice([3,6,9])}
    return config

def cnn_transformer_config():
    ''' hyperparameters for CNN+CNN architecture''' 
    
    config = {"encoder_dilation_factor" : tune.choice([1,2]),
            "encoder_kernel_size" : tune.choice([3,6,9])}
    return config

def mixer_config():
    ''' hyperparameters for GFNet+Transformer architecture''' 
    
    config = {"window_size" : tune.choice([100,150,200,250,300,350,400]),
              "lambd_L1" : tune.qloguniform(1e-3,1.0,1e-3)}
    return config

def lfnet_cnn_config():
    ''' hyperparameters for GFNet+Transformer architecture''' 
    
    config = {"window_size" : tune.choice([100,150,200,250,300,350,400]),
              "lambd_L1" : tune.qloguniform(1e-3,1.0,1e-3),
              "decoder_dilation_factor" : tune.choice([1,2]),
                "decoder_kernel_size" : tune.choice([3,6,9])}
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
    model_config = cnn_transformer_config() if cmd_args.model_type == "CNN-Transformer" else mixer_config() 
    config.update(model_config)
    #config['pos_decay_rate'] = tune.qloguniform(1e-3,1.0,1e-3)

    metric = "valid_class_accuracy"
    time_attr = "valid_step"
    
    max_time = 20000
    max_iter = 40
    n_samples = 32 

    ray.init(num_cpus=16,_temp_dir='/tmp/ray_tune') 
    bohb_hyperband = HyperBandForBOHB(time_attr=time_attr,
                                        max_t=max_time)

    # best setting found via manual experimentation
    seed_config = {"model_dim": 64,
                    "n_enc_layers": 12,
                    "n_dec_layers": 12,
                    "dropout" : 0.2,
                    "lr_warmup_steps" : 4000,
                    "window_size" : 200,
                    "lambd_L1" : 0.5}

    #bohb_search = TuneBOHB(points_to_evaluate=[seed_config])
    bohb_search = TuneBOHB()

    name = f"new_BOHB_search_{cmd_args.mode}_{cmd_args.model_type}"
    train_wrapper = tune.with_parameters(train_protein_coding_potential,cmd_args=cmd_args)
    
    wandb_callback = WandbLoggerCallback(project=f"full_tune{cmd_args.mode}-{cmd_args.model_type} Hyperparam Search",\
                    api_key=os.environ["WANDB_KEY"],log_config=False)
    
    total_time_allowed = 7*24*60*60 # 7 days in seconds 
  
    # stop trial on convergence or after max iterations
    stopper = CombinedStopper(MaximumIterationStopper(max_iter=max_iter),
                            TrialPlateauStopper(metric=metric))

    analysis = tune.run(train_wrapper,
                        name=name,
                        config=config,
                        scheduler=bohb_hyperband,
                        search_alg=bohb_search,
                        num_samples=n_samples,
                        time_budget_s=total_time_allowed,
                        resources_per_trial={'gpu': 1, 'cpu' : 4},
                        stop=stopper,
                        metric=metric,
                        mode="max",
                        callbacks=[wandb_callback])
    
    print("Best hyperparameters found were: ", analysis.best_config)

