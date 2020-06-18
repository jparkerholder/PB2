# PB2
Code for the Population-Based Bandits (PB2) Algorithm, from the paper *Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits*.

The framework is based on a union of [ray](https://github.com/ray-project/ray) (using rllib and tune) and [GPy](https://github.com/SheffieldML/GPy)

This is currently a work in progress. The code will be regularly updated during June/July 2020.

To run the IMPALA experiment, use command:

``
 python run_impala.py 
``

Within that function, there are multiple ways to mix it up. You can choose the following:

-env_name: for example BreakoutNoFrameSkip-v4
-method: either pb2 or pbt.
-freq: the frequency of updating hyperparams, we use 500,000 in the paper.
-seed: we used 0 1 2 3 4 5 6... and will continue to run up to 10 for the camera ready.
-max: the maximum number of timesteps, we used 10,000,000.

It should also be possible to adapt this code to run other ray tune schedulers. We used it for ASHA in our PPO experiments. 

Please get in touch for all questions.
jackph [at] robots [dot] ox [dot] ac [dot] uk

