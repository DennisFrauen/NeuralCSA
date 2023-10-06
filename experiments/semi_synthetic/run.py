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

    n_samples = config_run["monte_carlo_samples"]
    d_test = datasets["d_test"]
    x_test = d_test.data["x"].detach().numpy()
    # Propensity scores
    #prop_test = d_test.data["propensity"].detach().numpy()
    prop_test_fitted = joint_model.propensity.predict(d_test.data["x"]).detach().numpy()
    # Exclude overlap violations / regions with high empirical uncertainty -> bounds may not be valid
    overlap_idx = (prop_test_fitted < 0.7) & (prop_test_fitted > 0.3)
    x_test = x_test[overlap_idx[:, 0], :]
    #oracle_gammas_test = {}
    #for sensitivity_model in d_test.oracle_gammas:
    #    oracle_gammas_test[sensitivity_model] = d_test.data[sensitivity_model].detach().numpy()#[overlap_idx[:, 0], :]

    data_test_a1 = CausalDataset(x=x_test, a=np.full((x_test.shape[0], 1), 1), y=None)#, gammas=oracle_gammas_test)
    data_test_a2 = CausalDataset(x=x_test, a=np.full((x_test.shape[0], 1), 0), y=None)#, gammas=oracle_gammas_test)
    all_queries = joint_model.get_all_bounds_diff_avg(data_test_a1, data_test_a2, n_samples=n_samples,
                                                      bound_type="both")
    cate_upper = all_queries["upper"]
    cate_lower = all_queries["lower"]

    # cate_oracle = (2* np.mean(x_test, axis=1, keepdims=True) + 2) / datasets["y_std"]
    cate_oracle = 2 * np.mean(x_test, axis=1, keepdims=True) / datasets["y_std"]
    # Compute coverage
    coverage = []
    for i in range(len(cate_upper)):
        coverage.append(np.mean(
            (cate_lower[i].squeeze() < cate_oracle.squeeze()) & (cate_upper[i].squeeze() > cate_oracle.squeeze())))
        # Check if cate_upper or cate_lower is contains NaNs
        if np.isnan(cate_upper[i]).any() or np.isnan(cate_lower[i]).any():
            coverage[-1] = np.nan

    lengths = []
    for i in range(len(cate_upper)):
        lengths.append(np.median(cate_upper[i].squeeze() - cate_lower[i].squeeze()))


    print(coverage)
    print(lengths)

    # Plot densities for first run (upper bound)
    if run == 0:
        x_test_density = x_test[0:1, :]
        data_test = CausalDataset(x=x_test_density, a=np.full((1, 1), 1), y=None)
        y_grid = np.expand_dims(np.arange(-1, 1, 0.05), 1)
        stage1 = joint_model.stage1.test_likelihood(data_test=data_test, y=torch.tensor(y_grid, dtype=torch.float32))
        sns.set(font_scale=1.25)
        sns.set_style("whitegrid")
        sns.lineplot(x=y_grid.squeeze(), y=stage1.detach().numpy(), label="Stage 1", color="darkred", linewidth=3, linestyle="--")
        gammas = [5.48, 0.25, 0.38, 0.18]
        palette = ["darkblue", "darkgreen", "purple"]
        stage2_labels = ["MSM", "KL", "TV", "HE"]
        stage2_labels = [stage2_labels[i] + fr" $\Gamma^\ast = {gammas[i]}$" for i in range(len(gammas))]
        for i in range(min(2, len(joint_model.stage2_upper))):
            stage2_i = joint_model.stage2_upper[i].test_likelihood(data_test=data_test,
                                                                       y=torch.tensor(y_grid, dtype=torch.float32))
            sns.lineplot(x=y_grid.squeeze(), y=stage2_i.detach().numpy(), label=stage2_labels[i], color=palette[i],
                         linewidth=2)
        plt.savefig(utils.get_project_path() + "/experiments/semi_synthetic/results/plt_densities.pdf", bbox_inches='tight')
        # Legend to lower left
        plt.legend(loc="upper left").set_visible(True)
        plt.show()
    #x_test_density = x_test[0:1, :]
    #data_test = CausalDataset(x=x_test_density, a=np.full((1, 1), 1), y=None)
    #y_grid = np.expand_dims(np.arange(-3, 3, 0.05), 1)
    #stage1 = joint_model.stage1.test_likelihood(data_test=data_test, y=torch.tensor(y_grid, dtype=torch.float32)).detach().numpy()
    #stage2 = []
    #for i in range(len(joint_model.stage2_lower)):
    #    stage2.append(joint_model.stage2_lower[i].test_likelihood(data_test=data_test,
    #                                                              y=torch.tensor(y_grid, dtype=torch.float32)).detach().numpy())

    return {"coverage": coverage, "lengths": lengths}#, "density_stage1": stage1, "density_stage2": stage2}


def end_function(config_run, results, scm=None):
    results_coverage = [result["coverage"] for result in results]
    results_lengths = [result["lengths"] for result in results]
    # Set Nan for failed run (Rosenbaum, run 2)
    #results_coverage_msm = [result["coverage_msm"] for result in results]
    #results_length_msm = [result["length_msm"] for result in results]
    coverage_mean = np.nanmean(results_coverage, axis=0).squeeze()
    coverage_std = np.nanstd(results_coverage, axis=0).squeeze()
    lengths_mean = np.nanmean(results_lengths, axis=0).squeeze()
    lengths_std = np.nanstd(results_lengths, axis=0).squeeze()
    #coverage_msm_mean = np.nanmean(results_coverage_msm, axis=0).squeeze()
    #coverage_msm_std = np.nanstd(results_coverage_msm, axis=0).squeeze()
    #length_msm_mean = np.nanmean(results_length_msm, axis=0).squeeze()
    #length_msm_std = np.nanstd(results_length_msm, axis=0).squeeze()
    print("Coverage mean: " + str(coverage_mean))
    print("Coverage std: " + str(coverage_std))
    print("Lengths mean: " + str(lengths_mean))
    print("Lengths std: " + str(lengths_std))


if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/semi_synthetic/config")
    run_experiment(config_run, exp_function=exp_function, end_function=end_function)
