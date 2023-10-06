import utils.utils as utils  # Import module
import numpy as np
from data.data_generation import semi_synthetic_data

# Compute oracle gammas of simulated SCM for different sensitvity models
if __name__ == "__main__":
    q = 0.5
    np.random.seed(0)
    config_run = utils.load_yaml("/experiments/semi_synthetic/config")
    d_train, d_val, d_test, Y_1_test, Y_0_test, y_mean, y_std = semi_synthetic_data(config_run["data"])
    overlap_idx = (d_test.data["propensity"] < 0.3) | (d_test.data["propensity"] > 0.7)
    for sensitivity_model in config_run["run"]["sensitivity_models"]:
        sensitivity_name = sensitivity_model["name"]
        if sensitivity_name == "rosenbaum_binary":
            gamma = d_test.data["rosenbaum"].detach().numpy()[~overlap_idx[:, 0], :]
        else:
            gamma = d_test.data[sensitivity_name].detach().numpy()[~overlap_idx[:, 0], :]
        gamma = np.quantile(gamma, q=q, axis=0)

        print(sensitivity_name)
        print(gamma)
