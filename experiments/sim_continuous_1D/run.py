import utils.utils as utils  # Import module
from experiments.main import run_experiment
import numpy as np
from data.data_structures import CausalDataset
import seaborn as sns
import matplotlib.pyplot as plt

def exp_function(config_run, datasets, joint_model, run):

    n_samples = config_run["monte_carlo_samples"]
    x = np.expand_dims(np.arange(-1, 1, 0.01), 1)
    data_test_a1 = CausalDataset(x=x, a=np.full((x.shape[0], 1), 0.5), y=None)
    # Get upper bounds for all stage 2 models
    all_queries = joint_model.get_all_bounds_diff_avg(data_test_a1, None, n_samples=n_samples,
                                                      bound_type="both")
    cate_upper = all_queries["upper"]
    cate_lower = all_queries["lower"]

    # GMSM bounds
    all_gmsm_bounds = joint_model.get_all_bounds_diff_avg_gmsm(data_test_a1, None, n_samples=n_samples)
    gmsm_bounds_upper = all_gmsm_bounds["upper"]
    gmsm_bounds_lower = all_gmsm_bounds["lower"]

    return {"cate_upper": cate_upper, "cate_lower": cate_lower, "gmsm_bounds_upper": gmsm_bounds_upper,
            "gmsm_bounds_lower": gmsm_bounds_lower}


def end_function(config_run, results, scm=None):
    n_samples = config_run["monte_carlo_samples"]
    results_lower = [result["cate_lower"] for result in results]
    results_upper = [result["cate_upper"] for result in results]
    lower_mean = np.mean(results_lower, axis=0).squeeze()
    upper_mean = np.mean(results_upper, axis=0).squeeze()
    lower_std = np.std(results_lower, axis=0).squeeze()
    upper_std = np.std(results_upper, axis=0).squeeze()
    results_lower_gmsm = [result["gmsm_bounds_lower"] for result in results]
    results_upper_gmsm = [result["gmsm_bounds_upper"] for result in results]
    lower_mean_gmsm = np.mean(results_lower_gmsm, axis=0).squeeze()
    upper_mean_gmsm = np.mean(results_upper_gmsm, axis=0).squeeze()
    lower_std_gmsm = np.std(results_lower_gmsm, axis=0).squeeze()
    upper_std_gmsm = np.std(results_upper_gmsm, axis=0).squeeze()



    # Oracle CATE
    x = np.expand_dims(np.arange(-1, 1, 0.01), 1)
    cate_oracle = scm.get_true_query(x, n_samples=n_samples, int_a=0.5)

    # Plot cate, upper and lower bounds over x
    palette = ["darkblue", "darkgreen", "purple", "orange", "yellow"]
    gammas = [2, 4, 10]
    sns.set(font_scale=1.25)
    sns.set_style("whitegrid")
    # plt.figure(figsize=(20, 6))
    sns.lineplot(x=x.squeeze(), y=cate_oracle.squeeze(), label=fr"Oracle $\mu(x,a)$", color="darkred", linewidth=3)
    for i in range(lower_mean.shape[0]):
        line_color = palette[i]
        sns.lineplot(x=x.squeeze(), y=upper_mean[i, :].squeeze(), label=fr"Ours $\Gamma = {gammas[i]}$",
                     color=line_color, linestyle='dashed', linewidth=3)
        sns.lineplot(x=x.squeeze(), y=lower_mean[i, :].squeeze().squeeze(), color=line_color, linestyle='dashed', linewidth=3)
        sns.lineplot(x=x.squeeze(), y=upper_mean_gmsm[i, :].squeeze(), label=fr"CF $\Gamma = {gammas[i]}$",
                     color=line_color)
        sns.lineplot(x=x.squeeze(), y=lower_mean_gmsm[i, :].squeeze(), color=line_color)

        # Add shaded areas for standard deviation
        plt.fill_between(x.squeeze(), upper_mean[i, :].squeeze() + upper_std[i, :].squeeze(),
                         upper_mean[i, :].squeeze() - upper_std[i, :].squeeze(), alpha=0.2, color=line_color)
        plt.fill_between(x.squeeze(), lower_mean[i, :].squeeze() + lower_std[i, :].squeeze(),
                            lower_mean[i, :].squeeze() - lower_std[i, :].squeeze(), alpha=0.2, color=line_color)
        plt.fill_between(x.squeeze(), upper_mean_gmsm[i, :].squeeze() + upper_std_gmsm[i, :].squeeze(),
                         upper_mean_gmsm[i, :].squeeze() - upper_std_gmsm[i, :].squeeze(), alpha=0.2, color=line_color)
        plt.fill_between(x.squeeze(), lower_mean_gmsm[i, :].squeeze() + lower_std_gmsm[i, :].squeeze(),
                         lower_mean_gmsm[i, :].squeeze() - lower_std_gmsm[i, :].squeeze(), alpha=0.2, color=line_color)

    # Customize the legend
    plt.legend(loc="best").set_visible(True)

    # Add labels and title
    plt.xlabel(r"$X$")
    plt.ylabel(r"$Y$")
    path = "/experiments/sim_continuous_1D/results/plot_continuous_1D_a05.pdf"
    plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    config_run = utils.load_yaml("/experiments/sim_continuous_1D/config")
    run_experiment(config_run, exp_function=exp_function, end_function=end_function)