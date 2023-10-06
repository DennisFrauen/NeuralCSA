import copy
import os
import sys
# main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(main_dir)
import utils.utils as utils  # Import module
from experiments.main import run_experiment
import numpy as np
import utils.plotting as plotting
import torch
from data.data_structures import CausalDataset
import seaborn as sns
import matplotlib.pyplot as plt


def exp_function(config_run, datasets, joint_model, run=None):
    path_rel = "/experiments/real_world/results/"
    data_test_a1 = CausalDataset(x=datasets["d_test"].data["x"][0:1, :].detach().numpy(), a=np.ones((1, 1)), y=None)
    # Scaling
    scaling = {}
    scaling["mean"] = torch.tensor([datasets["scaling_y"][i][0] for i in range(len(datasets["scaling_y"]))], dtype=datasets["d_test"].data["x"].dtype)
    scaling["std"] = torch.tensor([datasets["scaling_y"][i][1] for i in range(len(datasets["scaling_y"]))], dtype=datasets["d_test"].data["x"].dtype)
    #scaling = None

    # Plotting
    #Treatment 1 upper bounds
    x_range = [0, 3.2]
    y_range = [-1.3, 3.3]
    if scaling is not None:
        x_range = x_range * scaling["std"][0].numpy() + scaling["mean"][0].numpy()
        y_range = y_range * scaling["std"][1].numpy() + scaling["mean"][1].numpy()
    l_min = 0
    l_max = 0.0041


    plotting.plot_2d_density_stage1(joint_model, data_test=data_test_a1, xl=x_range[0], xu=x_range[1], yl=y_range[0], yu=y_range[1], stepsize=0.1,
                                    path=path_rel + "/plot_real_stage1.pdf", scaling=scaling, l_min=l_min, l_max=l_max)
    plotting.plot_2d_density_stage2(joint_model.stage2_upper[0].test_likelihood, data_test=data_test_a1,xl=x_range[0], xu=x_range[1],
                                    yl=y_range[0], yu=y_range[1], stepsize=0.1,
                                    path=path_rel + "/plot_real_stage2_upper_gamma2.pdf", scaling=scaling, l_min=l_min, l_max=l_max)
    plotting.plot_2d_density_stage2(joint_model.stage2_upper[1].test_likelihood, data_test=data_test_a1, xl=x_range[0], xu=x_range[1],
                                    yl=y_range[0], yu=y_range[1], stepsize=0.1,
                                    path=path_rel + "/plot_real_stage2_upper_gamma4.pdf", scaling=scaling, l_min=l_min, l_max=l_max)

    # Treatment 1 lower bounds
    x_range = [0, 3.2]
    y_range = [-1.3, 3.3]
    if scaling is not None:
        x_range = x_range * scaling["std"][0].numpy() + scaling["mean"][0].numpy()
        y_range = y_range * scaling["std"][1].numpy() + scaling["mean"][1].numpy()
    l_min = 0
    l_max = 0.0035

    plotting.plot_2d_density_stage2(joint_model.stage2_lower[0].test_likelihood, data_test=data_test_a1, xl=x_range[0],
                                    xu=x_range[1],
                                    yl=y_range[0], yu=y_range[1], stepsize=0.1,
                                    path=path_rel + "/plot_real_stage2_lower_gamma2.pdf", scaling=scaling, l_min=l_min,
                                    l_max=l_max)
    plotting.plot_2d_density_stage2(joint_model.stage2_lower[1].test_likelihood, data_test=data_test_a1, xl=x_range[0],
                                    xu=x_range[1],
                                    yl=y_range[0], yu=y_range[1], stepsize=0.1,
                                    path=path_rel + "/plot_real_stage2_lower_gamma4.pdf", scaling=scaling, l_min=l_min,
                                    l_max=l_max)




    return None



if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/real_world/config")
    run_experiment(config_run, exp_function=exp_function)
