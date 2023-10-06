from optuna.samplers import RandomSampler
import optuna
import random
import torch
import numpy as np
import utils.utils as utils
from data.data_generation import get_datasets
from models.stage1 import CNF_stage1
from models.propensity import PropensityNet

def run_hyper_tuning(config_hyper, config_data):
    # Load data
    seed = config_hyper["seed"]
    _ = set_seeds(seed)
    datasets = get_datasets(config_data)
    models = config_hyper["models"]
    tuning_ranges = config_hyper["tuning_ranges"]
    hyper_path = "/experiments/" + config_hyper["name"] + "/model_configs/"

    # Tune
    for model in models:
        tune_sampler = set_seeds(seed)
        obj = get_objective(model, datasets, tuning_ranges[model])
        study = tune_objective(obj, study_name=model, num_samples=config_hyper["num_samples"], sampler=tune_sampler)
        best_params = study.best_trial.params
        # Save params
        path = hyper_path + model
        utils.save_yaml(path, best_params)
    print("Done tuning")

def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tune_sampler = RandomSampler(seed=seed)
    return tune_sampler


def get_objective(model_name, datasets, tuning_ranges):
    data_dims = datasets["d_train"].get_dims(["x", "a", "y"])
    def obj(trial):
        #Sample params
        params = sample_params(trial, tuning_ranges)
        if model_name == "stage1":
            config_base = {"dim_context": data_dims["x"] + data_dims["a"], "dim_y": data_dims["y"], "neptune": True}
            model = CNF_stage1(config_base | params)
        else:
            config_base = {"dim_input": data_dims["x"], "dim_output": data_dims["a"],
                           "out_type": datasets["d_train"].datatypes["a_type"], "neptune": True}
            model = PropensityNet(config_base | params)
        train_results = model.fit(datasets["d_train"], datasets["d_val"], name="propensity")
        # Fit

        return train_results[0]["val_obj"]

    return obj


def tune_objective(objective, study_name, num_samples=10, sampler=None):
    if sampler is not None:
        study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler)
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective, n_trials=num_samples)

    print("Finished. Best trial:")
    trial_best = study.best_trial

    print("  Value: ", trial_best.value)

    print("  Params: ")
    for key, value in trial_best.params.items():
        print("    {}: {}".format(key, value))
    return study

def sample_params(trial, tuning_ranges):
    params = {}
    for param in tuning_ranges.keys():
        params[param] = trial.suggest_categorical(param, tuning_ranges[param])
    params["neptune"] = False
    return params