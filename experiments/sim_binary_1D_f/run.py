import os
import sys
#main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys.path.append(main_dir)
import utils.utils as utils  # Import module
from experiments.main import run_experiment
import numpy as np
import utils.plotting as plotting
import torch
from data.data_structures import CausalDataset
import seaborn as sns
import matplotlib.pyplot as plt

def exp_function(config_run, datasets, joint_model, run):
    scm = datasets["scm"]
    if config_run["plotting"]:
        path_rel = "/experiments/sim_binary_1D/results/"
        # Propensity fit
        #plotting.plot_propensity_fit(scm, joint_model)
        # Plot observed vs intervened distribution
        d_obs = scm.sample_y(10000, int_x=0.5, cond_a=0)
        d_int = scm.sample_y(10000, int_x=0.5, int_a=0)
        #plotting.plot_1d_histogram(d_obs)
        #plotting.plot_1d_histogram(d_int)
        # Test data for two different contexts (x, a)
        data_test = CausalDataset(x=np.full((1, 1), 0.5), a=np.zeros((1, 1)), y=None)
        data_test2 = CausalDataset(x=np.full((1, 1), -0.3), a=np.ones((1, 1)), y=None)
        # Context 1
        plotting.plot_1d_density_stage1(joint_model, data_test, l=-3, r=3, stepsize=0.1)
        plotting.plot_1d_density_stage2(joint_model.stage2_lower[0].test_likelihood, data_test, l=-3.5, r=3.5, stepsize=0.001)
        plotting.plot_1d_density_ratio_y(joint_model.stage2_lower[0].test_density_ratio_y, data_test=data_test, l=-3.5, r=3.5, stepsize=0.001)
        # Plot samples from stage 2
        #plotting.plot_2d_samples_stage2(joint_model, data_test=data_test, n_samples=1000)
        # Test sensitivity value
        test_sens = joint_model.stage2_lower[0].test_sensitivity_violation(data_test, n_samples=1000).squeeze().detach().numpy()
        test_query = joint_model.stage2_lower[0].test_query(data_test, n_samples=1000).squeeze().detach().numpy()
        print("Sensitivity violation: " + str(test_sens))
        print("Query: " + str(test_query))

    # Propensity fit
    #plotting.plot_propensity_fit(scm, joint_model)
    # Plot shifted densities
    #plotting.plot_1d_stage2_densities(joint_model, scm,x=0, a=1, l=-3, r=3, stepsize=0.1)

    # Plot CATE and bounds
    #plotting.plot_1d_CATE_f(joint_model, scm, a1=1, a2=0, l=-0.7, r=0.7, stepsize=0.01, gmsm_bounds=True, bound_type=config_run["bounds"], legend=True,
    #                             path="/experiments/sim_binary_1D_f/results/plot_binary_1D_f.pdf", gmsm_type="msm")

    #plotting.plot_sensitivity_value(scm, "rosenbaum", l=-0.7, r=0.7, stepsize=0.01)
    n_samples = config_run["monte_carlo_samples"]
    x = np.expand_dims(np.arange(-0.7, 0.7, 0.01), 1)
    data_test_a1 = CausalDataset(x=x, a=np.full((x.shape[0], 1), 1), y=None)
    data_test_a2 = CausalDataset(x=x, a=np.full((x.shape[0], 1), 0), y=None)
    # Get upper bounds for all stage 2 models
    all_queries = joint_model.get_all_bounds_diff_avg(data_test_a1, data_test_a2, n_samples=n_samples,
                                                      bound_type="both")
    cate_upper = all_queries["upper"]
    cate_lower = all_queries["lower"]

    # MSM bounds
    Q_gmsm_a1 = joint_model.GMSM_bounds(data_test_a1, n_samples=n_samples, gammas=torch.tensor(2),
                              sensitivity_model="msm")
    msm_bounds_upper = Q_gmsm_a1["Q_plus"].detach().numpy()
    msm_bounds_lower = Q_gmsm_a1["Q_minus"].detach().numpy()
    Q_gmsm_a2 = joint_model.GMSM_bounds(data_test_a2, n_samples=n_samples, gammas=torch.tensor(2),
                                  sensitivity_model="msm")
    msm_bounds_upper = msm_bounds_upper - Q_gmsm_a2["Q_minus"].detach().numpy()
    msm_bounds_lower = msm_bounds_lower - Q_gmsm_a2["Q_plus"].detach().numpy()

    # GMSM bounds
    #all_gmsm_bounds = joint_model.get_all_bounds_diff_avg_gmsm(data_test_a1, data_test_a2, n_samples=n_samples)
    #gmsm_bounds_upper = all_gmsm_bounds["upper"]
    #gmsm_bounds_lower = all_gmsm_bounds["lower"]

    return {"cate_upper": cate_upper, "cate_lower": cate_lower, "msm_bounds_upper": msm_bounds_upper,
            "msm_bounds_lower": msm_bounds_lower}


def end_function(config_run, results, scm=None):
    n_samples = config_run["monte_carlo_samples"]
    results_lower = [result["cate_lower"] for result in results]
    results_upper = [result["cate_upper"] for result in results]
    lower_mean = np.nanmean(results_lower, axis=0).squeeze()
    upper_mean = np.nanmean(results_upper, axis=0).squeeze()
    lower_std = np.nanstd(results_lower, axis=0).squeeze()
    upper_std = np.nanstd(results_upper, axis=0).squeeze()

    results_lower_msm = [result["msm_bounds_lower"] for result in results]
    results_upper_msm = [result["msm_bounds_upper"] for result in results]
    lower_mean_msm = np.nanmean(results_lower_msm, axis=0).squeeze()
    upper_mean_msm = np.nanmean(results_upper_msm, axis=0).squeeze()
    lower_std_msm = np.nanstd(results_lower_msm, axis=0).squeeze()
    upper_std_msm = np.nanstd(results_upper_msm, axis=0).squeeze()

    # Oracle CATE
    x = np.expand_dims(np.arange(-0.7, 0.7, 0.01), 1)
    cate_oracle = scm.get_true_query(x, n_samples=n_samples, int_a=1)
    cate_a2 = scm.get_true_query(x, n_samples=n_samples, int_a=0)
    cate_oracle = cate_oracle - cate_a2


    # Plot cate, upper and lower bounds over x
    palette = ["darkgreen", "purple", "black", "orange", "yellow"]
    gammas = [0.81, 0.42, 0.14, 0.75, 4]
    labels = ["KL ", "TV ", "HE ", fr"$\chi^2$ ", "RB "]
    labels = [labels[i] + fr"$\Gamma^\ast = {gammas[i]}$" for i in range(len(gammas))]
    sns.set(font_scale=1.25)
    sns.set_style("whitegrid")
    # plt.figure(figsize=(20, 6))
    sns.lineplot(x=x.squeeze(), y=cate_oracle.squeeze(), label=fr"Oracle $\tau(x)$", color="darkred", linewidth=3)
    # CMSM bounds
    sns.lineplot(x=x.squeeze(), y=upper_mean_msm, label=fr"MSM $\Gamma^\ast = {2}$", color="darkblue",
                 linewidth=2)
    sns.lineplot(x=x.squeeze(), y=lower_mean_msm, color="darkblue", linewidth=2)
    plt.fill_between(x.squeeze(), upper_mean_msm + upper_std_msm,
                     upper_mean_msm - upper_std_msm, alpha=0.2, color="darkblue")
    plt.fill_between(x.squeeze(), lower_mean_msm + lower_std_msm,
                     lower_mean_msm - lower_std_msm, alpha=0.2, color="darkblue")
    for i in range(lower_mean.shape[0]):
        line_color = palette[i]
        sns.lineplot(x=x.squeeze(), y=upper_mean[i, :].squeeze(), label=labels[i],
                     color=line_color, linewidth=1, linestyle="dashed")
        sns.lineplot(x=x.squeeze(), y=lower_mean[i, :].squeeze().squeeze(), color=line_color, linewidth=1, linestyle="dashed")
        #sns.lineplot(x=x.squeeze(), y=upper_mean_gmsm[i, :].squeeze(), label=fr"CF $\Gamma = {gammas[i]}$",
        #             color=line_color)
        #sns.lineplot(x=x.squeeze(), y=lower_mean_gmsm[i, :].squeeze(), color=line_color)

        # Add shaded areas for standard deviation
        plt.fill_between(x.squeeze(), upper_mean[i, :].squeeze() + upper_std[i, :].squeeze(),
                         upper_mean[i, :].squeeze() - upper_std[i, :].squeeze(), alpha=0.2, color=line_color)
        plt.fill_between(x.squeeze(), lower_mean[i, :].squeeze() + lower_std[i, :].squeeze(),
                            lower_mean[i, :].squeeze() - lower_std[i, :].squeeze(), alpha=0.2, color=line_color)

    # Customize the legend
    plt.legend(loc="lower left").set_visible(True)


    # Add labels and title
    plt.xlabel(r"$X$")
    plt.ylabel(r"$Y$")
    path = "/experiments/sim_binary_1D_f/results/plot_binary_1D_f.pdf"
    plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/sim_binary_1D_f/config")
    run_experiment(config_run, exp_function=exp_function, end_function=end_function)