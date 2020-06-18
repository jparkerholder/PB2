
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import argparse
import pandas as pd
from datetime import datetime

import ray
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining, AsyncHyperBandScheduler
from pb2 import PB2
#from ppbt_noreward import PPBT

# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    config['target_delay'] = int(config['target_delay'])
    return config

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=3000000)
    parser.add_argument("--algo", type=str, default='IMPALA')
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--freq", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--perturb", type=float, default=0.25)
    parser.add_argument("--env_name", type=str, default="SpaceInvadersNoFrameskip-v4")
    parser.add_argument("--criteria", type=str, default="timesteps_total") # "training_iteration"
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--method", type=str, default="pb2") # ['pbt', 'pb2', 'asha']
    
    args = parser.parse_args()
    ray.init()
    
    args.dir = "{}_{}_{}_Size{}_{}_{}".format(args.algo, args.filename, args.method, str(args.num_samples), args.env_name, args.criteria)
    if not(os.path.exists('data/'+args.dir)):
        os.makedirs('data/'+args.dir)

    pbt = PopulationBasedTraining(
        time_attr= args.criteria,
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=args.freq,
        resample_probability=args.perturb,
        quantile_fraction = args.perturb, # copy bottom % with top %
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "epsilon":  lambda: random.uniform(0.01, 0.5), # 0.1
            "entropy_coeff": lambda: random.uniform(0.001, 0.1), # 0.01
            "lr": lambda: random.uniform(1e-5, 1e-2), # 5e-3
       },
        custom_explore_fn=explore)
    
    pb2 = PB2(
        time_attr= args.criteria,
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=args.freq,
        resample_probability=0,
        quantile_fraction = args.perturb, # copy bottom % with top %
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "epsilon":  lambda: random.uniform(0.01, 0.5), # 0.1
            "entropy_coeff": lambda: random.uniform(0.001, 0.1), # 0.01
            "lr": lambda: random.uniform(1e-5, 1e-2), # 5e-3
       },
        custom_explore_fn=explore)

    asha = AsyncHyperBandScheduler(
        time_attr=args.criteria,
        metric="episode_reward_mean",
        mode="max",
        grace_period=args.freq,
        max_t=args.max)

    
    methods = {'pbt': pbt,
               'pb2': pb2,
               'asha': asha}
    
    timelog = str(datetime.date(datetime.now())) + '_' + str(datetime.time(datetime.now()))
    
    analysis = run(
        args.algo,
        name="{}_{}_{}_seed{}_{}".format(timelog, args.method, args.env_name, str(args.seed), args.filename),
        scheduler=methods[args.method],
        verbose=1,
        num_samples= args.num_samples,
        stop= {args.criteria: args.max},
        
        config= {
            "env": args.env_name,
            "log_level": "INFO",
            "seed": args.seed,
            "num_gpus": 0,
            "num_workers": args.num_workers,
            "horizon": args.horizon,
            "rollout_fragment_length": 50,
            "train_batch_size": 500,
            "num_envs_per_worker": 5,
            "epsilon": sample_from(
                    lambda spec: random.uniform(0.1, 0.5)),
            "entropy_coeff": sample_from(
                    lambda spec: random.uniform(0.001, 0.1)),
            "lr": sample_from(
                    lambda spec: random.uniform(1e-5, 1e-2)),
            }
        )
            

    
    all_dfs = analysis.trial_dataframes
    names = list(all_dfs.keys())
    
    results = pd.DataFrame()    
    for i in range(args.num_samples):
        df = all_dfs[names[i]]
        df = df[['timesteps_total', 'episodes_total', 'episode_reward_mean']]
        df['Agent'] = i
        results = pd.concat([results, df]).reset_index(drop=True)
    
    results.to_csv("data/{}/seed{}.csv".format(args.dir, str(args.seed)))


