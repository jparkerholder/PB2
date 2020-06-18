from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from copy import deepcopy
import itertools
import logging
import json
import math
import os
import random
import shutil
import GPy
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from dppy.finite_dpps import FiniteDPP

from ray.tune.error import TuneError
from ray.tune.result import TRAINING_ITERATION
from ray.tune.logger import _SafeFallbackEncoder
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.suggest.variant_generator import format_vars
from ray.tune.trial import Trial, Checkpoint

from kernel import TV_SquaredExp

logger = logging.getLogger(__name__)


class PBTTrialState(object):
    """Internal PBT state tracked per-trial."""

    def __init__(self, trial):
        self.orig_tag = trial.experiment_tag
        self.last_score = None
        self.last_checkpoint = None
        self.last_perturbation_time = 0

    def __repr__(self):
        return str((self.last_score, self.last_checkpoint,
                    self.last_perturbation_time))


def normalize(data, wrt):
    # data = data to normalize
    # wrt = data will be normalized with respect to this
    return (data - np.min(wrt, axis=0))/(np.max(wrt,axis=0) - np.min(wrt,axis=0))

def standardize(data):
    data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-8)
    return np.clip(data, -2, 2)

def UCB(m, m1, x, fixed, kappa=0.5):
    
    c1 = 0.2 # from TV-GP-UCB
    c2 = 0.4
    beta_t = c1 * np.log(c2 * m.X.shape[0])
    kappa = np.sqrt(beta_t)
    
    xtest = np.concatenate((fixed.reshape(-1, 1), np.array(x).reshape(-1,1))).T
    
    preds = m.predict(xtest)
    mean = preds[0][0][0] 
    
    preds = m1.predict(xtest)
    var = preds[1][0][0]
    return mean + kappa * var


def optimize_acq(func, m, m1, fixed, num_f):
    
    print("Optimizing Acquisition Function...\n")
    
    opts = {'maxiter':200, 'maxfun':200, 'disp':False}
    
    T=10
    best_value=-999
    best_theta = m1.X[0,:]
    
    bounds = [(0,1) for _ in range(m.X.shape[1]-num_f)]
    
    for ii in range(T):
        x0 = np.random.uniform(0,1, m.X.shape[1]-num_f)
        
        res = minimize(lambda x: -func(m, m1, x, fixed), x0, bounds=bounds, method="L-BFGS-B", options=opts)
        
        val = func(m, m1, res.x, fixed)
        if val > best_value:
            best_value=val
            best_theta =res.x
    
    return(np.clip(best_theta, 0, 1))

def get_diverse(X, y, size=10):

    # This is a heuristic if the covariance matrix is ill conditioned. 
    
    if X.shape[0] > 10:
        size = int(X.shape[0]/2)
    else:
        return(X, y)

    K = rbf_kernel(X, X)    
    
    L = np.matmul(K, np.linalg.inv(np.eye(K.shape[0]) - K))

    DPP = FiniteDPP('likelihood', **{'L': L})
    DPP.flush_samples()
    try:
        DPP.sample_exact_k_dpp(size=size)
    except ValueError:
        return(X, y)
    newX = X[DPP.list_of_samples[0], :]
    newy = y[DPP.list_of_samples[0], :]
    return(newX, newy)

def select_length(Xraw, yraw, current, newpoint, bounds, num_f, num):

    # block size.
    
    min_len = 200
    
    if Xraw.shape[0] < min_len:
        return(Xraw.shape[0])
    else:
        length = min_len-10   
        scores = []
        while length+10 <= Xraw.shape[0]:
            length += 10
            
            base_vals = np.array(list(bounds.values())).T
            X_len = Xraw[-length:, :]
            y_len = yraw[-length:]
            oldpoints = X_len[:, :num_f]
            old_lims = np.concatenate((np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))).reshape(2, oldpoints.shape[1])
            limits = np.concatenate((old_lims, base_vals),axis=1)
            
            X = normalize(X_len, limits)
            y = standardize(y_len).reshape(y_len.size, 1)
            
            kernel = TV_SquaredExp(input_dim=X.shape[1], variance=1., lengthscale=1., epsilon=0.1)
            try:
                m = GPy.models.GPRegression(X, y, kernel)
            except np.linalg.LinAlgError:
                X, y = get_diverse(X, y)
                m = GPy.models.GPRegression(X, y, kernel)
            try:
                m.optimize(messages=True)
            except np.linalg.LinAlgError:
                X, y = get_diverse(X, y)
                m = GPy.models.GPRegression(X, y, kernel)
                m.optimize(messages=True)
            scores.append(m.log_likelihood())
        idx = np.argmax(scores)
        length = (idx+int((min_len/10))) * 10
        return(length)
    
def select_config(Xraw, yraw, current, newpoint, bounds, num_f, num):
    
    length = select_length(Xraw, yraw, current, newpoint, bounds, num_f, num)
    print("\n\nUsing length = {}\n\n".format(length))
    
    Xraw = Xraw[-length:, :]
    yraw = yraw[-length: ]
        
    base_vals = np.array(list(bounds.values())).T
    oldpoints = Xraw[:, :num_f]
    old_lims = np.concatenate((np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))).reshape(2, oldpoints.shape[1])
    limits = np.concatenate((old_lims, base_vals),axis=1)
    
    X = normalize(Xraw, limits)
    y = standardize(yraw).reshape(yraw.size, 1)
    
    fixed = normalize(newpoint, oldpoints)

    kernel = TV_SquaredExp(input_dim=X.shape[1], variance=1., lengthscale=1., epsilon=0.1)
    try:
        m = GPy.models.GPRegression(X, y, kernel)
    except np.linalg.LinAlgError:
        X, y = get_diverse(X, y)
        m = GPy.models.GPRegression(X, y, kernel)
    
    try:
        m.optimize(messages=True)
    except np.linalg.LinAlgError:
        X, y = get_diverse(X, y)
        m = GPy.models.GPRegression(X, y, kernel)
        m.optimize(messages=True)
        
    m.kern.lengthscale.fix(m.kern.lengthscale.clip(1e-5,1))
    
    if current is None:
        m1 = deepcopy(m)
    else:
        padding = np.array([fixed for _ in range(current.shape[0])])
        current = normalize(current, base_vals)
        current = np.hstack((padding, current))

        Xnew = np.vstack((X, current))
        ypad = np.zeros(current.shape[0])
        ypad = ypad.reshape(-1,1)
        ynew = np.vstack((y, ypad))
        
        kernel = TV_SquaredExp(input_dim=X.shape[1], variance=1., lengthscale=1., epsilon=0.1)
        m1 = GPy.models.GPRegression(Xnew, ynew, kernel)
        m1.optimize()
    
    time_param = m1.time_se.epsilon[0]
    lengthscale = m1.time_se.lengthscale[0]
    xt = optimize_acq(UCB, m, m1, fixed, num_f)
    
    # check dist vs other points:
    dists = euclidean_distances(X[:, num_f:], X[:, num_f:])
    meandist = np.mean(dists)
    min_new = np.min(euclidean_distances(X[:, num_f:], xt.reshape(-1,1).T))
    
    # convert back...
    xt = xt * (np.max(base_vals,axis=0) - np.min(base_vals,axis=0)) + np.min(base_vals, axis=0)
    
    print("\n**\n\n New Point: {} \n\n**\n".format(str(xt)))
    print("\n**\n\n Length Scale: {} \n\n**\n".format(str(lengthscale)))
    print("\n**\n\n Min Dist: {} \n\n**\n".format(str(min_new)))
    print("\n**\n\n vs. Mean Dist: {} \n\n**\n".format(str(meandist)))
    
    xt = xt.astype(np.float32)
    return(xt, lengthscale, min_new, meandist)

def explore(data, bounds, current, base, old, config, mutations, resample_probability):
    
    print("\n\nGP Bandit Time :) \n\n")
    
    data.to_csv("checks.csv", index=False)
    data['Trial'] = data['Trial'].astype(str)
    
    # df = pd.read_csv("checks.csv")

    df = data.sort_values(by='T').reset_index(drop=True)
    df['y'] = df.groupby('Trial')['Reward'].diff()
    df['t_change'] = df.groupby('Trial')['T'].diff()
    df = df[df['t_change'] > 0].reset_index(drop=True)
    df['R_before'] = df.Reward - df.y
    df['y'] = df.y / df.t_change
    df = df[~df.y.isna()].reset_index(drop=True)
    df = df.sort_values(by='T').reset_index(drop=True)
    
    # replace previous data for the trial being replaced
    # otherwise we have a huge jump
    marker=0
    done = False
    while not done:
        old_rename = str(old) +'_*_'+ str(marker)
        if data[data['Trial'] == old_rename].empty:
            data = data.replace(str(old), old_rename)
            done = True
        else:
            marker += 1
    
    dfnewpoint = df[df['Trial']==str(base)]
    
    if not dfnewpoint.empty:
        
        # override previous entries of this trial
        # we don't want to compare against them as the weights were overriden
        
        ref = dfnewpoint[[c for c in data.columns]]
        ref.Trial = str(old)
        ref = ref.tail(1).reset_index(drop=True)
        data = pd.concat([data, ref]).reset_index(drop=True)
        
        # now specify the dataset for the GP
                
        y = np.array(df.y.values)
        # meta data we keep -> episodes and reward (TODO: convert to curve)
        t_r = df[['T', 'R_before']] # df[['T', 'R_before']]
        h = df[[key for key in bounds.keys()]]
        X = pd.concat([t_r, h], axis=1).values 
        newpoint = df[df['Trial']==str(base)].iloc[-1, :][['T', 'R_before']].values
        new, lengthscale, mindist, meandist = select_config(X,y, current, newpoint, bounds, num_f = len(t_r.columns), num=len(h.columns))
        new_config = config.copy()
        for i in range(len(h.columns)):
            if type(config[h.columns[i]]) is int:
                new_config[h.columns[i]] = int(new[i])
            else:
                new_config[h.columns[i]] = new[i]
    else:
        new_config = config
        lengthscale = -1
        mindist = -1
        meandist = -1

    return new_config, lengthscale, mindist, meandist, data


def make_experiment_tag(orig_tag, config, mutations):
    """Appends perturbed params to the trial name to show in the console."""

    resolved_vars = {}
    for k in mutations.keys():
        resolved_vars[("config", k)] = config[k]
    return "{}@perturbed[{}]".format(orig_tag, format_vars(resolved_vars))


class PB2(FIFOScheduler):
    """
    Below text is from PBT... which this was adapted from.

    Implements the Population Based Training (PBT) algorithm.

    https://deepmind.com/blog/population-based-training-neural-networks

    PBT trains a group of models (or agents) in parallel. Periodically, poorly
    performing models clone the state of the top performers, and a random
    mutation is applied to their hyperparameters in the hopes of
    outperforming the current top models.

    Unlike other hyperparameter search algorithms, PBT mutates hyperparameters
    during training time. This enables very fast hyperparameter discovery and
    also automatically discovers good annealing schedules.

    This Tune PBT implementation considers all trials added as part of the
    PBT population. If the number of trials exceeds the cluster capacity,
    they will be time-multiplexed as to balance training progress across the
    population. To run multiple trials, use `tune.run(num_samples=<int>)`.

    In {LOG_DIR}/{MY_EXPERIMENT_NAME}/, all mutations are logged in
    `pbt_global.txt` and individual policy perturbations are recorded
    in pbt_policy_{i}.txt. Tune logs: [target trial tag, clone trial tag,
    target trial iteration, clone trial iteration, old config, new config]
    on each perturbation step.

    Args:
        time_attr (str): The training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        metric (str): The training result objective value attribute. Stopping
            procedures will use this attribute.
        mode (str): One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        perturbation_interval (float): Models will be considered for
            perturbation at this interval of `time_attr`. Note that
            perturbation incurs checkpoint overhead, so you shouldn't set this
            to be too frequent.
        hyperparam_mutations (dict): Hyperparams to mutate. The format is
            as follows: for each key, either a list or function can be
            provided. A list specifies an allowed set of categorical values.
            A function specifies the distribution of a continuous parameter.
            You must specify at least one of `hyperparam_mutations` or
            `custom_explore_fn`.
        quantile_fraction (float): Parameters are transferred from the top
            `quantile_fraction` fraction of trials to the bottom
            `quantile_fraction` fraction. Needs to be between 0 and 0.5.
            Setting it to 0 essentially implies doing no exploitation at all.
        resample_probability (float): The probability of resampling from the
            original distribution when applying `hyperparam_mutations`. If not
            resampled, the value will be perturbed by a factor of 1.2 or 0.8
            if continuous, or changed to an adjacent value if discrete.
        custom_explore_fn (func): You can also specify a custom exploration
            function. This function is invoked as `f(config)` after built-in
            perturbations from `hyperparam_mutations` are applied, and should
            return `config` updated as needed. You must specify at least one of
            `hyperparam_mutations` or `custom_explore_fn`.
        log_config (bool): Whether to log the ray config of each model to
            local_dir at each exploit. Allows config schedule to be
            reconstructed.

    Example:
        >>> pbt = PopulationBasedTraining(
        >>>     time_attr="training_iteration",
        >>>     metric="episode_reward_mean",
        >>>     mode="max",
        >>>     perturbation_interval=10,  # every 10 `time_attr` units
        >>>                                # (training_iterations in this case)
        >>>     hyperparam_mutations={
        >>>         # Perturb factor1 by scaling it by 0.8 or 1.2. Resampling
        >>>         # resets it to a value sampled from the lambda function.
        >>>         "factor_1": lambda: random.uniform(0.0, 20.0),
        >>>         # Perturb factor2 by changing it to an adjacent value, e.g.
        >>>         # 10 -> 1 or 10 -> 100. Resampling will choose at random.
        >>>         "factor_2": [1, 10, 100, 1000, 10000],
        >>>     })
        >>> tune.run({...}, num_samples=8, scheduler=pbt)
    """

    def __init__(self,
                 time_attr="time_total_s",
                 reward_attr=None,
                 metric="episode_reward_mean",
                 mode="max",
                 perturbation_interval=60.0,
                 hyperparam_mutations={},
                 quantile_fraction=0.25,
                 resample_probability=0.25,
                 custom_explore_fn=None,
                 log_config=True):
        if not hyperparam_mutations and not custom_explore_fn:
            raise TuneError(
                "You must specify at least one of `hyperparam_mutations` or "
                "`custom_explore_fn` to use PBT.")

        if quantile_fraction > 0.5 or quantile_fraction < 0:
            raise TuneError(
                "You must set `quantile_fraction` to a value between 0 and"
                "0.5. Current value: '{}'".format(quantile_fraction))

        assert mode in ["min", "max"], "`mode` must be 'min' or 'max'!"

        if reward_attr is not None:
            mode = "max"
            metric = reward_attr
            logger.warning(
                "`reward_attr` is deprecated and will be removed in a future "
                "version of Tune. "
                "Setting `metric={}` and `mode=max`.".format(reward_attr))

        FIFOScheduler.__init__(self)
        self._metric = metric
        if mode == "max":
            self._metric_op = 1.
        elif mode == "min":
            self._metric_op = -1.
        self._time_attr = time_attr
        self._perturbation_interval = perturbation_interval
        self._hyperparam_mutations = hyperparam_mutations
        self._quantile_fraction = quantile_fraction
        self._resample_probability = resample_probability
        self._trial_state = {}
        self._custom_explore_fn = custom_explore_fn
        self._log_config = log_config
        
        self.meta = {'timesteps': [],'lengthscales': [], 'closest': [], 'meandist': []}
        self.latest = 0 # when we last did bayesopt
        self.data = pd.DataFrame()
        
        self.bounds = {}
        for key, distribution in self._hyperparam_mutations.items():
            self.bounds[key] = [np.min([distribution() for _ in range(999999)]),np.max([distribution() for _ in range(999999)])]

        # Metrics
        self._num_checkpoints = 0
        self._num_perturbations = 0

    def on_trial_add(self, trial_runner, trial):
        self._trial_state[trial] = PBTTrialState(trial)

    def on_trial_result(self, trial_runner, trial, result):
        
        score = self._metric_op * result[self._metric]

        names = []
        values = []
        for key, distribution in self._hyperparam_mutations.items():
            names.append(str(key))
            values.append(trial.config[key])
    
        lst = [[trial, result["timesteps_total"]] + values + [score]]
        cols = ['Trial', 'T'] + names + ['Reward']
        entry = pd.DataFrame(lst, columns = cols) 

        self.data = pd.concat([self.data, entry]).reset_index(drop=True)
        self.data.Trial = self.data.Trial.astype('str')

        if self._time_attr not in result or self._metric not in result:
            return TrialScheduler.CONTINUE
        time = result[self._time_attr]
        state = self._trial_state[trial]

        if time - state.last_perturbation_time < self._perturbation_interval:
            return TrialScheduler.CONTINUE  # avoid checkpoint overhead

        score = self._metric_op * result[self._metric]
        state.last_score = score
        state.last_perturbation_time = time
        lower_quantile, upper_quantile = self._quantiles()

        
        if trial in upper_quantile:
            state.last_checkpoint = trial_runner.trial_executor.save(
                trial, Checkpoint.MEMORY)
            self._num_checkpoints += 1
        else:
            state.last_checkpoint = None  # not a top trial

        if trial in lower_quantile:
            trial_to_clone = random.choice(upper_quantile)
            assert trial is not trial_to_clone
            self._exploit(trial_runner.trial_executor, trial, trial_to_clone)

        for trial in trial_runner.get_trials():
            if trial.status in [Trial.PENDING, Trial.PAUSED]:
                return TrialScheduler.PAUSE  # yield time to other trials

        return TrialScheduler.CONTINUE

    def _log_config_on_step(self, trial_state, new_state, trial,
                            trial_to_clone, new_config):
        """Logs transition during exploit/exploit step.

        For each step, logs: [target trial tag, clone trial tag, target trial
        iteration, clone trial iteration, old config, new config].

        """
        trial_name, trial_to_clone_name = (trial_state.orig_tag,
                                           new_state.orig_tag)
        trial_id = "".join(itertools.takewhile(str.isdigit, trial_name))
        trial_to_clone_id = "".join(
            itertools.takewhile(str.isdigit, trial_to_clone_name))
        trial_path = os.path.join(trial.local_dir,
                                  "pbt_policy_" + trial_id + ".txt")
        trial_to_clone_path = os.path.join(
            trial_to_clone.local_dir,
            "pbt_policy_" + trial_to_clone_id + ".txt")
        policy = [
            trial_name, trial_to_clone_name,
            trial.last_result.get(TRAINING_ITERATION, 0),
            trial_to_clone.last_result.get(TRAINING_ITERATION, 0),
            trial_to_clone.config, new_config
        ]
        # Log to global file.
        with open(os.path.join(trial.local_dir, "pbt_global.txt"), "a+") as f:
            print(json.dumps(policy, cls=_SafeFallbackEncoder), file=f)
        # Overwrite state in target trial from trial_to_clone.
        if os.path.exists(trial_to_clone_path):
            shutil.copyfile(trial_to_clone_path, trial_path)
        # Log new exploit in target trial log.
        with open(trial_path, "a+") as f:
            f.write(json.dumps(policy, cls=_SafeFallbackEncoder) + "\n")

    def _exploit(self, trial_executor, trial, trial_to_clone):
        """Transfers perturbed state from trial_to_clone -> trial.

        If specified, also logs the updated hyperparam state.

        """

        trial_state = self._trial_state[trial]
        new_state = self._trial_state[trial_to_clone]
        
        if not new_state.last_checkpoint:
            logger.info("[pbt]: no checkpoint for trial."
                        " Skip exploit for Trial {}".format(trial))
            return
        
        # if we are at a new timestep, we dont want to penalise for trials still going
        if self.data['T'].max() > self.latest:
            self.current = None
        
        print("\n\n\n\n Copying: \n{} \n with:{} \n\n".format(str(trial), str(trial_to_clone)))
        new_config, lengthscale, mindist, meandist, data = explore(self.data, self.bounds,
                             self.current,
                             trial_to_clone,
                             trial,
                             trial_to_clone.config,
                             self._hyperparam_mutations,
                             self._resample_probability)
        
        # important to replace the old values, since we are copying across
        self.data = data.copy()
        
        # if the current guy youre selecting is at a point youve already done, 
        # then append the data to the "current" which is the points in the current batch
        
        new = []
        for key in self._hyperparam_mutations.keys():
            new.append(new_config[key])
    
        new  = np.array(new)
        new = new.reshape(1, new.size)
        if self.data['T'].max() > self.latest:
            self.latest = self.data['T'].max()
            self.current = new.copy()
        else:
            self.current = np.concatenate((self.current, new), axis=0)
            print("\n\n\n\n\n Currently Evaluating \n\n\n\n\n")
            print(self.current)
            print("\n\n\n\n\n")
        
        # log the lengthscale
        self.meta['timesteps'].append(self.data['T'].values[-1])
        self.meta['lengthscales'].append(lengthscale)
        self.meta['closest'].append(mindist)
        self.meta['meandist'].append(meandist)
        meta = pd.DataFrame({'timesteps': self.meta['timesteps'], 
                             'lengthscales': self.meta['lengthscales'],
                             'closest': self.meta['closest'],
                             'meandist': self.meta['meandist']})
        meta.to_csv('meta_data.csv')
        
        logger.info("[exploit] transferring weights from trial "
                    "{} (score {}) -> {} (score {})".format(
                        trial_to_clone, new_state.last_score, trial,
                        trial_state.last_score))

        if self._log_config:
            self._log_config_on_step(trial_state, new_state, trial,
                                     trial_to_clone, new_config)

        new_tag = make_experiment_tag(trial_state.orig_tag, new_config,
                                      self._hyperparam_mutations)
        reset_successful = trial_executor.reset_trial(trial, new_config,
                                                      new_tag)
        if reset_successful:
            trial_executor.restore(
                trial, Checkpoint.from_object(new_state.last_checkpoint))
        else:
            trial_executor.stop_trial(trial, stop_logger=False)
            trial.config = new_config
            trial.experiment_tag = new_tag
            trial_executor.start_trial(
                trial, Checkpoint.from_object(new_state.last_checkpoint))

        self._num_perturbations += 1
        # Transfer over the last perturbation time as well
        trial_state.last_perturbation_time = new_state.last_perturbation_time

    def _quantiles(self):
        """Returns trials in the lower and upper `quantile` of the population.

        If there is not enough data to compute this, returns empty lists.

        """

        trials = []
        for trial, state in self._trial_state.items():
            if state.last_score is not None and not trial.is_finished():
                trials.append(trial)
        trials.sort(key=lambda t: self._trial_state[t].last_score)

        if len(trials) <= 1:
            return [], []
        else:
            num_trials_in_quantile = int(
                math.ceil(len(trials) * self._quantile_fraction))
            if num_trials_in_quantile > len(trials) / 2:
                num_trials_in_quantile = int(math.floor(len(trials) / 2))
            return (trials[:num_trials_in_quantile],
                    trials[-num_trials_in_quantile:])

    def choose_trial_to_run(self, trial_runner):
        """Ensures all trials get fair share of time (as defined by time_attr).

        This enables the PBT scheduler to support a greater number of
        concurrent trials than can fit in the cluster at any given time.

        """

        candidates = []
        for trial in trial_runner.get_trials():
            if trial.status in [Trial.PENDING, Trial.PAUSED] and \
                    trial_runner.has_resources(trial.resources):
                candidates.append(trial)
        candidates.sort(
            key=lambda trial: self._trial_state[trial].last_perturbation_time)
        return candidates[0] if candidates else None

    def reset_stats(self):
        self._num_perturbations = 0
        self._num_checkpoints = 0

    def last_scores(self, trials):
        scores = []
        for trial in trials:
            state = self._trial_state[trial]
            if state.last_score is not None and not trial.is_finished():
                scores.append(state.last_score)
        return scores

    def debug_string(self):
        return "PopulationBasedTraining: {} checkpoints, {} perturbs".format(
            self._num_checkpoints, self._num_perturbations)
