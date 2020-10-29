## Population-Based Bandits (PB2)


Code for the Population-Based Bandits (PB2) Algorithm, from the paper *Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits*.

The framework is based on a union of [ray](https://github.com/ray-project/ray) (using rllib and tune) and [GPy](https://github.com/SheffieldML/GPy). Heavily inspired by the ray tune pbt_ppo example. 

*NOTE* We are working to include PB2 into the ``ray.tune`` library, which will become the official supported implementation. This repo is a semi-buggy version to share with the community while we work on that. It is expected to be released in ray v1.0.1.


#### Running the Code

To run the IMPALA experiment, use command:

``
 python run_impala.py 
``

To run the PPO experiment, use command:

``
 python run_ppo.py 
``

#### Config

Within that function, there are multiple ways to mix it up. You can choose the following:

-env_name: for example BreakoutNoFrameSkip-v4. \
-method: either pb2 or pbt (or asha for PPO).  \
-freq: the frequency of updating hyperparams, we use 500,000 for IMPALA and 50,000 for PPO.  \
-seed: we used 0 1 2 3 4 5 6... and plan to add more seeds.  \
-max: the maximum number of timesteps, we used 10,000,000 for IMPALA and 1,000,000 for PPO.  

It should also be possible to adapt this code to run other ray tune schedulers. We used it for ASHA in our PPO experiments. We are also working to include a BOHB baseline. 

Please get in touch for all questions.
jackph [at] robots [dot] ox [dot] ac [dot] uk

#### Citing PB2

Finally, if you found this repo useful, please consider citing us:

```
@article{pb2_neurips,  
    title={Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits},  
    author={Parker-Holder, Jack and Nguyen, Vu and Roberts, Stephen},  
    journal={Advances in Neural Information Processing Systems},  
    year={2020}   
}
```
