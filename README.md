## Population-Based Bandits (PB2)


Code for the Population-Based Bandits (PB2) Algorithm, from the paper *Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits*.

The framework is based on a union of [ray](https://github.com/ray-project/ray) (using rllib and tune) and [GPy](https://github.com/SheffieldML/GPy). Heavily inspired by the ray tune pbt_ppo example. 

*NOTE* PB2 is included in the ``ray.tune`` library, which is the official supported implementation. The link to the code is [here](https://docs.ray.io/en/master/_modules/ray/tune/schedulers/pb2.html), and the accompanying blog post is [here](https://www.anyscale.com/blog/population-based-bandits).


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
@inproceedings{NEURIPS2020_c7af0926,
 author = {Parker-Holder, Jack and Nguyen, Vu and Roberts, Stephen J},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {17200--17211},
 publisher = {Curran Associates, Inc.},
 title = {Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits},
 url = {https://proceedings.neurips.cc/paper/2020/file/c7af0926b294e47e52e46cfebe173f20-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
