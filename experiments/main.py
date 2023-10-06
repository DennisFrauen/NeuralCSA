import torch
import numpy as np
import utils.utils as utils
from data.data_generation import get_datasets
import wandb
import random
from models.auxillary import JointModel
from data.data_structures import CausalDataset


def run_experiment(config_run, exp_function, end_function=None):
    if config_run["run"]["validation"]:
        wandb.login(key="f2a65b64221c6eda048b770aa5a599e538934e2a")
    seed = config_run["run"]["seed"]
    result_path = utils.get_project_path() + "/experiments/" + config_run["run"]["name"] + "/results/"
    results = []


    for i in range(config_run["run"]["n_runs"]):
        print("Starting run " + str(i + 1) + " of " + str(config_run["run"]["n_runs"]))
        utils.set_seed(seed)
        seed = random.randint(0, 1000000)
        if config_run["run"]["run_start_index"] <= i+1:
            # Load data
            datasets = get_datasets(config_run["data"])
            # Create model config
            config_models = create_config(config_run, datasets)
            # Train or load models
            joint_model = get_models(config_run["run"], config_models, datasets, run=i, seed=seed)
            # Run experiment, result should be dictionary of pandas dataframes
            result = exp_function(config_run["run"], datasets, joint_model, run=i)
            # Save results
            if "save" in config_run:
                if config_run["save"]:
                    if result is not None:
                        for key in result.keys():
                            result[key].to_pickle(result_path + key + "_run_" + str(i) + ".pkl")
            results.append(result)
    if end_function is not None:
        if "scm" in datasets.keys():
            end_function(config_run["run"], results, datasets["scm"])
        else:
            end_function(config_run["run"], results)
    print("Experiment finished")


def get_models(config_run, config_models, datasets, run=0, seed=0):
    savepath = "/experiments/" + config_run["name"] + "/saved_models/run_" + str(run) + "/"

    joint_model = JointModel(config_models)
    utils.set_seed(seed)
    # Propensity
    if "train_propensity" in config_run.keys():
        if config_run["train_propensity"] and datasets["d_train"].datatypes["a_type"] == "discrete":
            print("Fitting propensity model")
            joint_model.fit_propensity(datasets["d_train"], datasets["d_val"])
            joint_model.save_propensity(savepath + "propensity")
        elif datasets["d_train"].datatypes["a_type"] == "discrete":
            joint_model.load_propensity(savepath + "propensity")


    utils.set_seed(seed)
    # Stage 1
    if config_run["train_stage1"]:
        print("Fitting stage 1")
        joint_model.fit_stage1(datasets["d_train"], datasets["d_val"])
        if config_run["save_stage1"]:
            joint_model.save_stage1(savepath + "stage1")
    else:
        joint_model.load_stage1(savepath + "stage1")

    utils.set_seed(seed)
    # Stage 2
    if config_run["train_stage2_all"]:
        print("Fitting stage 2 models")

        if config_models["stage2_upper"][0]["dim_context"] == 0:
            data_test = CausalDataset(x=np.full((1, 1), 0.5), a=np.zeros((1, 1)), y=None)
        else:
            data_test = datasets["d_train"]
        joint_model.fit_stage2(data_test, seed=seed, savepath=savepath + "stage2/")
        #if config_run["save_stage2_all"]:
        #    joint_model.save_stage2(savepath + "stage2/")
    else:
        print("Loading stage 2 models")
        joint_model.load_stage2(savepath + "stage2/")

    return joint_model


def create_config(config_run, datasets):
    config_path = "/experiments/" + config_run["run"]["name"] + "/model_configs/"
    data_dims = datasets["d_train"].get_dims(["x", "a", "y"])

    # Base config for propensity
    config_base = {"dim_input": data_dims["x"], "dim_output": data_dims["a"],
                   "out_type": datasets["d_train"].datatypes["a_type"], "neptune": config_run["run"]["validation"]}
    # Propensity config
    if "train_propensity" in config_run["run"].keys():
        if datasets["d_train"].datatypes["a_type"] == "discrete":
            config_propensity = utils.load_yaml(config_path + "propensity") | config_base
        else:
            config_propensity = None
    else:
        config_propensity = None
    # Stage 1 config
    config_stage1 = {"dim_context": data_dims["x"] + data_dims["a"], "dim_y": data_dims["y"],
                     "neptune": config_run["run"]["validation"]} | utils.load_yaml(config_path + "stage1")
    # Stage 2 config
    config_stage2_base = {"neptune": config_run["run"]["validation"], "dim_y": data_dims["y"],
                          "bounds": config_run["run"]["bounds"], "a_type": datasets["d_train"].datatypes["a_type"]}
    config_stage2_upper =[]
    config_stage2_lower =[]
    for query in config_run["run"]["queries"]:
        if config_run["run"]["sensitivity_models"] is not None:
            for sensitivity_model in config_run["run"]["sensitivity_models"]:
                for gamma in sensitivity_model["gammas"]:
                    config_stage2_upper.append(config_stage2_base | utils.load_yaml(config_path + "stage2/upper/" + query["name"] +
                                                    "_" + sensitivity_model["name"] + "_" + str(gamma)))
                    config_stage2_lower.append(config_stage2_base | utils.load_yaml(config_path + "stage2/lower/" + query["name"] +
                                                    "_" + sensitivity_model["name"] + "_" + str(gamma)))
                    if config_stage2_upper[-1]["cnf"] == True:
                        config_stage2_upper[-1]["dim_context"] = data_dims["x"] + data_dims["a"]
                        config_stage2_lower[-1]["dim_context"] = data_dims["x"] + data_dims["a"]
                    else:
                        config_stage2_upper[-1]["dim_context"] = 0
                        config_stage2_lower[-1]["dim_context"] = 0

    # Overall config
    model_config = {"propensity": config_propensity, "stage1": config_stage1, "stage2_upper": config_stage2_upper, "stage2_lower": config_stage2_lower}
    return model_config

# def save_scale_params(gmsm, savepath):
#    scaling_params = gmsm.scaling_params.copy()
#    if torch.is_tensor(scaling_params["mean"]):
#        scaling_params["mean"] = scaling_params["mean"].item()
#        scaling_params["sd"] = scaling_params["sd"].item()
#    utils.save_yaml(savepath, scaling_params)
