from data.scms import SCM_binary, SCM_binary_1D, SCM_continuous_1D
import numpy as np
import torch
from data.MIMIC.mimic_semi_synthetic import semi_synthetic_data, real_world_data
from data.IHDP.data_ihdp import load_data_ihdp

def get_datasets(config_data):
    if config_data["name"] == "sim_binary":
        scm = SCM_binary(config_data)
        # Trainig data
        d_train  = scm.generate_dataset(n_samples=config_data["n_train"])
        # Validation data
        d_val = scm.generate_dataset(n_samples=config_data["n_val"])
        return {"d_train": d_train, "d_val": d_val, "scm": scm}

    if config_data["name"] in ["sim_binary_1D", "sim_binary_1D_f"]:
        scm = SCM_binary_1D(config_data)
        # Trainig data
        d_train = scm.generate_dataset(n_samples=config_data["n_train"])
        # Validation data
        d_val = scm.generate_dataset(n_samples=config_data["n_val"])
        return {"d_train": d_train, "d_val": d_val, "scm": scm}

    if config_data["name"] in ["sim_continuous_1D", "sim_continuous_1D_f"]:
        scm = SCM_continuous_1D(config_data)
        # Trainig data
        d_train = scm.generate_dataset(n_samples=config_data["n_train"])
        # Validation data
        d_val = scm.generate_dataset(n_samples=config_data["n_val"])
        return {"d_train": d_train, "d_val": d_val, "scm": scm}

    if config_data["name"] == "semi_synthetic":
        d_train, d_val, d_test, Y_1_test, Y_0_test, y_mean, y_std = semi_synthetic_data(config_data)
        return {"d_train": d_train, "d_val": d_val, "d_test": d_test, "Y_1_test": Y_1_test, "Y_0_test": Y_0_test,
                "y_mean": y_mean, "y_std": y_std}

    if config_data["name"] == "real_world":
        d_train, d_val, d_test, scaling_y = real_world_data(config_data)
        return {"d_train": d_train, "d_val": d_val, "d_test": d_test, "scaling_y": scaling_y}

    if config_data["name"] == "ihdp":
        d_train, d_val, d_test, Y_1_test, Y_0_test, y_mean, y_std = load_data_ihdp(config_data)
        return {"d_train": d_train, "d_val": d_val, "d_test": d_test, "Y_1_test": Y_1_test, "Y_0_test": Y_0_test,
                "y_mean": y_mean, "y_std": y_std}