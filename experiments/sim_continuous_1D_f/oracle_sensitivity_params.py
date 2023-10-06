import utils.utils as utils  # Import module
import numpy as np
from data.scms import SCM_continuous_1D

# Compute oracle gammas of simulated SCM for different sensitvity models
if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/sim_continuous_1D_f/config")
    scm = SCM_continuous_1D(config_run["data"])
    x = np.expand_dims(np.arange(-0.7, 0.7, 0.1), 1)
    a = 0.5
    # CMSM
    sensitivity_values = scm.get_msm_sensitivity_value(x=x, a=a)
    print("CMSM")
    print(sensitivity_values.mean())

    for sensitivity_model in config_run["run"]["sensitivity_models"]:
        sensitivity_name = sensitivity_model["name"]
        if sensitivity_name == "rosenbaum_continuous":
            sensitivity_values = scm.get_rosenbaum_sensitivity_value(x=x, a=a)
        elif sensitivity_name == "cmsm":
            sensitivity_values = scm.get_msm_sensitivity_value(x=x, a=a)
        else:
            sensitivity_values = scm.get_f_sensitivity_value(x=x, f=utils.get_f_by_name(sensitivity_name), a=a)

        print(sensitivity_name)
        print(sensitivity_values.mean())
