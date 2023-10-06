import utils.utils as utils  # Import module
import numpy as np
from data.scms import SCM_binary_1D

# Compute oracle gammas of simulated SCM for different sensitvity models
if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/sim_binary_1D_f/config")
    scm = SCM_binary_1D(config_run["data"])
    x = np.expand_dims(np.arange(-1, 1, 0.1), 1)
    # MSM
    sensitivity_values = scm.get_msm_sensitivity_value(x, a=None)
    print("MSM")
    print(sensitivity_values)
    for sensitivity_model in config_run["run"]["sensitivity_models"]:
        sensitivity_name = sensitivity_model["name"]
        if sensitivity_name == "rosenbaum_binary":
            sensitivity_values = scm.get_rosenbaum_sensitivity_value(x, a=None)
        elif sensitivity_name == "msm":
            sensitivity_values = scm.get_msm_sensitivity_value(x, a=None)
        else:
            sensitivity_values = scm.get_f_sensitivity_value(x, utils.get_f_by_name(sensitivity_name), a=None)

        print(sensitivity_name)
        print(sensitivity_values)
