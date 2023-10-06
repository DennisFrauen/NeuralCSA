import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import utils.utils as utils
from data.data_structures import CausalDataset


def plot_2d_samples(data, path=None):
    # set seaborn style
    sns.set_style("white")
    # Basic 2D density plot
    plt.plot(data[:, 0], data[:, 1], "", linestyle='', marker='o', markersize=3, alpha=0.05, color="purple")
    if path is not None:
        plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()


def plot_2d_density_over_data(data, density, path=None):
    sns.set_style("white")
    f, ax = plt.subplots()
    points = ax.scatter(data[:, 0], data[:, 1], cmap="plasma", s=1,
                        c=density.detach().numpy().squeeze())
    f.colorbar(points)
    if path is not None:
        plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()

def plot_2d_contour(data_test, test_function, xl=-3, xu=3, yl=-3, yu=3, stepsize=0.1, path=None, scaling=None,
                    l_min=None, l_max=None):
    x = np.arange(xl, xu, stepsize)
    y = np.arange(yl, yu, stepsize)
    x_mesh, y_mesh = meshgrid(x, y)
    grid = np.concatenate((np.expand_dims(x_mesh.flatten(), 1), np.expand_dims(y_mesh.flatten(), 1)), axis=1)
    # Plot learned observed distribution by CNF stage 1
    density = test_function(data_test=data_test, y=torch.tensor(grid, dtype=torch.float32), scaling=scaling)
    # transform density to shape of x_mesh
    density = density.reshape(x_mesh.shape).detach().numpy()
    # Create contour plot
    plt.rcParams.update({'font.size': 14})
    if l_min is not None:
        levels = np.linspace(l_min, l_max,50)
        plt.contourf(x_mesh, y_mesh, density, levels, cmap="jet")
    else:
        plt.contourf(x_mesh, y_mesh, density, 50, cmap="jet")

    plt.colorbar()
    if path is not None:
        plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()




def plot_2d_density_stage1_over_data(data_y, joint_model, data_test):
    # Plot learned observed distribution by CNF stage 1
    density = joint_model.stage1.test_likelihood(data_test=data_test.data, y=torch.tensor(data_y))
    plot_2d_density_over_data(data_y, density)

def plot_2d_density_stage1(joint_model, data_test, xl=-3, xu=3, yl=-3, yu=3, stepsize=0.1, path=None, scaling=None, l_min=None, l_max=None):
    plot_2d_contour(data_test, joint_model.stage1.test_likelihood, xl, xu, yl, yu, stepsize, path, scaling, l_min=l_min, l_max=l_max)


def plot_2d_density_stage2(test_likelihood, data_test, xl=-3, xu=3, yl=-3, yu=3, stepsize=0.1, path=None, scaling=None, l_min=None, l_max=None):
    plot_2d_contour(data_test, test_likelihood, xl, xu, yl, yu, stepsize, path, scaling, l_min=l_min, l_max=l_max)

def plot_2d_density_ratio(test_density_ratio, data_test, xl=-3, xu=3, yl=-3, yu=3, stepsize=0.1, path=None):
    plot_2d_contour(data_test, test_density_ratio, xl, xu, yl, yu, stepsize, path)

def plot_2d_density_ratio_y(test_density_ratio_y, data_test, xl=-3, xu=3, yl=-3, yu=3, stepsize=0.1, path=None):
    plot_2d_contour(data_test, test_density_ratio_y, xl, xu, yl, yu, stepsize, path)

def plot_2d_samples_stage2(joint_model, data_test, n_samples=1000, path=None):
    samples = joint_model.stage2.test_samples(data_test.data, n_samples).detach().numpy()
    plot_2d_samples(samples, path)


def plot_propensity_fit(scm, sensitivity_model, path=None):
    # Create test data for validation of propensity fit
    x_test = np.expand_dims(np.linspace(-1, 1, 1000), 1)
    d_test = scm.generate_dataset(1000, int_x=x_test)
    propensity_test = scm.propensity_obs(x_test)
    propensity_hat = sensitivity_model.propensity.predict(d_test.data["x"])
    plt.plot(x_test, propensity_hat.detach().numpy(), label="Predicted propensity scores")
    plt.plot(x_test, propensity_test, label="True propensity scores")
    plt.title("Propensity fit")
    plt.legend()
    if path is not None:
        plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()

def plot_1d_histogram(data, path=None, bins=50):
    sns.set_style("white")
    plt.hist(data, bins=bins, density=True)
    if path is not None:
        plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()

def plot_1d_density(test_likelihood, data_test, l=-3, r=3, stepsize=0.1, path=None):
    x = np.expand_dims(np.arange(l, r, stepsize), 1)
    y = test_likelihood(data_test=data_test, y=torch.tensor(x, dtype=torch.float32))
    plt.plot(x, y.detach().numpy())
    if path is not None:
        plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()

def plot_1d_density_stage1(joint_model, data_test, l=-3, r=3, stepsize=0.1, path=None):
    plot_1d_density(joint_model.stage1.test_likelihood, data_test, l, r, stepsize, path)

def plot_1d_density_stage2(test_likelihood, data_test, l=-3, r=3, stepsize=0.1, path=None):
    plot_1d_density(test_likelihood, data_test, l, r, stepsize, path)


def plot_1d_density_ratio_y(test_density_ratio_y, data_test, l=-3, r=3, stepsize=0.1, path=None):
    plot_1d_density(test_density_ratio_y, data_test, l, r, stepsize, path)


def plot_1d_CATE_GMSM(joint_model, scm, a1=1, a2=0, l=-3, r=3, stepsize=0.05, gmsm_bounds=False, bound_type="both", path=None, legend=True, gmsm_type=None):
    n_samples= 30000
    x = np.expand_dims(np.arange(l, r, stepsize), 1)
    data_test_a1 = CausalDataset(x=x, a=np.full((x.shape[0], 1), a1), y=None)
    #data_test_a1 = CausalDataset(x=np.full((1, 1), 0.5), a=np.zeros((1, 1)), y=None)
    if a2 is not None:
        data_test_a2 = CausalDataset(x=x, a=np.full((x.shape[0], 1), a2), y=None)
    else:
        data_test_a2 = None
    # Get upper bounds for all stage 2 models
    all_queries = joint_model.get_all_bounds_diff_avg(data_test_a1, data_test_a2, n_samples=n_samples, bound_type=bound_type)
    cate_upper = all_queries["upper"]
    cate_lower = all_queries["lower"]

    # Get labels and gammas for legend
    stage2_labels = []
    gammas = []
    for i in range(max(len(joint_model.stage2_upper), len(joint_model.stage2_lower))):
        model_upper = joint_model.stage2_upper[i]
        stage2_labels.append("Stage 2 " + model_upper.config["causal_query"]["name"] + "_" + model_upper.config["sensitivity_model"]["name"])
        gammas.append(model_upper.config["sensitivity_model"]["gamma"])


    # Oracle CATE
    cate_oracle = scm.get_true_query(x, n_samples=n_samples, int_a=a1)
    if a2 is not None:
        cate_a2 = scm.get_true_query(x, n_samples=n_samples, int_a=a2)
        cate_oracle = cate_oracle - cate_a2
    # GMSM bounds
    all_gmsm_bounds = joint_model.get_all_bounds_diff_avg_gmsm(data_test_a1, data_test_a2, n_samples=n_samples)
    gmsm_bounds_upper = all_gmsm_bounds["upper"]
    gmsm_bounds_lower = all_gmsm_bounds["lower"]


    # Plot cate, upper and lower bounds over x
    palette = ["darkblue", "darkgreen", "brown", "orange", "yellow"]
    sns.set_style("whitegrid")
    #plt.figure(figsize=(20, 6))
    sns.lineplot(x=x.squeeze(), y=cate_oracle.squeeze(), label="Oracle Query", color="darkred", linewidth=3)
    for i in range(max(len(cate_upper), len(cate_lower))):
        line_color = palette[i]
        if bound_type == "both":
            sns.lineplot(x=x.squeeze(), y=cate_upper[i].squeeze(), label=fr"Ours $\Gamma = {gammas[i]}$", color=line_color, linestyle='dashed', linewidth=3)
            sns.lineplot(x=x.squeeze(), y=cate_lower[i].squeeze(), color=line_color, linestyle='dashed', linewidth=3)
            if gmsm_bounds:
                sns.lineplot(x=x.squeeze(), y=gmsm_bounds_upper[i].squeeze(), label=fr"CF $\Gamma = {gammas[i]}$", color=line_color)
                sns.lineplot(x=x.squeeze(), y=gmsm_bounds_lower[i].squeeze(), color=line_color)
        elif bound_type == "upper":
            sns.lineplot(x=x.squeeze(), y=cate_upper[i].squeeze(), label=fr"Ours $\Gamma = {gammas[i]}$", color=line_color, linestyle='dashed', linewidth=3)
            if gmsm_bounds:
                sns.lineplot(x=x.squeeze(), y=gmsm_bounds_upper[i].squeeze(), label=fr"CF $\Gamma = {gammas[i]}$", color=line_color,)
        elif bound_type == "lower":
            sns.lineplot(x=x.squeeze(), y=cate_lower[i].squeeze(), label=fr"Ours $\Gamma = {gammas[i]}$", color=line_color, linestyle='dashed', linewidth=3)
            if gmsm_bounds:
                sns.lineplot(x=x.squeeze(), y=gmsm_bounds_lower[i].squeeze(), label=fr"CF $\Gamma = {gammas[i]}$", color=line_color)

    # Customize the legend
    plt.legend(loc="best").set_visible(legend)

    # Add labels and title
    plt.xlabel(r"$X$")
    plt.ylabel(r"$Y$")

    if path is not None:
        plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()



def plot_1d_CATE_f(joint_model, scm, a1=1, a2=None, l=-3, r=3, stepsize=0.05, gmsm_bounds=False, bound_type="both", path=None, legend=True, gmsm_type="msm",
                   gmsm_gamma=2):
    n_samples= 30000
    x = np.expand_dims(np.arange(l, r, stepsize), 1)
    data_test_a1 = CausalDataset(x=x, a=np.full((x.shape[0], 1), a1), y=None)
    if a2 is not None:
        data_test_a2 = CausalDataset(x=x, a=np.full((x.shape[0], 1), a2), y=None)
    else:
        data_test_a2 = None
    # Get upper bounds for all stage 2 models
    all_queries = joint_model.get_all_bounds_diff_avg(data_test_a1, data_test_a2, n_samples=n_samples, bound_type=bound_type)
    cate_upper = all_queries["upper"]
    cate_lower = all_queries["lower"]

    # Get labels and gammas for legend
    stage2_labels = []
    gammas = []
    for i in range(max(len(joint_model.stage2_upper), len(joint_model.stage2_lower))):
        model_upper = joint_model.stage2_upper[i]
        stage2_labels.append("Stage 2 " + model_upper.config["causal_query"]["name"] + "_" + model_upper.config["sensitivity_model"]["name"])
        gammas.append(model_upper.config["sensitivity_model"]["gamma"])


    # Oracle CATE
    cate_oracle = scm.get_true_query(x, n_samples=n_samples, int_a=a1)
    if a2 is not None:
        cate_a2 = scm.get_true_query(x, n_samples=n_samples, int_a=a2)
        cate_oracle = cate_oracle - cate_a2
    # MSM bounds
    Q_gmsm_a1 = joint_model.GMSM_bounds(data_test_a1, n_samples=n_samples, gammas=torch.tensor(gmsm_gamma),
                              sensitivity_model=gmsm_type)
    msm_bounds_upper = Q_gmsm_a1["Q_plus"].detach().numpy()
    msm_bounds_lower = Q_gmsm_a1["Q_minus"].detach().numpy()
    if a2 is not None:
        Q_gmsm_a2 = joint_model.GMSM_bounds(data_test_a2, n_samples=n_samples, gammas=torch.tensor(gmsm_gamma),
                                  sensitivity_model=gmsm_type)
        msm_bounds_upper = msm_bounds_upper - Q_gmsm_a2["Q_minus"].detach().numpy()
        msm_bounds_lower = msm_bounds_lower - Q_gmsm_a2["Q_plus"].detach().numpy()


    # Plot cate, upper and lower bounds over x
    palette = ["darkblue", "darkgreen", "purple", "brown", "orange", "yellow"]
    sns.set_style("whitegrid")
    #plt.figure(figsize=(20, 6))
    sns.lineplot(x=x.squeeze(), y=cate_oracle.squeeze(), label="Oracle Query", color="darkred", linewidth=3)
    # CMSM bounds
    sns.lineplot(x=x.squeeze(), y=msm_bounds_upper.squeeze(), label=fr"MSM $\Gamma = {gmsm_gamma}$", color="darkblue", linewidth=3)
    sns.lineplot(x=x.squeeze(), y=msm_bounds_lower.squeeze(), color="darkblue", linewidth=3)
    for i in range(max(len(cate_upper), len(cate_lower))):
        line_color = palette[i+1]
        if bound_type == "both":
            sns.lineplot(x=x.squeeze(), y=cate_upper[i].squeeze(), label=fr"Ours $\Gamma = {gammas[i]}$", color=line_color, linestyle='dashed', linewidth=3)
            sns.lineplot(x=x.squeeze(), y=cate_lower[i].squeeze(), color=line_color, linestyle='dashed', linewidth=3)
        elif bound_type == "upper":
            sns.lineplot(x=x.squeeze(), y=cate_upper[i].squeeze(), label=fr"Ours $\Gamma = {gammas[i]}$", color=line_color, linestyle='dashed', linewidth=3)
        elif bound_type == "lower":
            sns.lineplot(x=x.squeeze(), y=cate_lower[i].squeeze(), label=fr"Ours $\Gamma = {gammas[i]}$", color=line_color, linestyle='dashed', linewidth=3)

    # Customize the legend
    plt.legend(loc="best").set_visible(legend)

    # Add labels and title
    plt.xlabel(r"$X$")
    plt.ylabel(r"$Y$")

    if path is not None:
        plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()


def plot_1d_stage2_densities(joint_model, scm, x=0, a=1, l=-3, r=3, stepsize=0.05, gmsm_bounds=False, bound_type="upper", path=None, legend=True):
    data_test = CausalDataset(x=np.full((1, 1), x), a=np.full((1, 1), a), y=None)
    y_grid = np.expand_dims(np.arange(l, r, stepsize), 1)
    stage1 = joint_model.stage1.test_likelihood(data_test=data_test, y=torch.tensor(y_grid, dtype=torch.float32))
    sns.lineplot(x=y_grid.squeeze(), y=stage1.detach().numpy(), label="Stage 1", color="darkred", linewidth=3)

    palette = ["darkblue", "darkgreen", "purple", "brown", "orange", "yellow"]
    for i in range(max(len(joint_model.stage2_upper), len(joint_model.stage2_lower))):
        if bound_type == "upper":
            stage2_i = joint_model.stage2_upper[i].test_likelihood(data_test=data_test, y=torch.tensor(y_grid, dtype=torch.float32))
        elif bound_type == "lower":
            stage2_i = joint_model.stage2_lower[i].test_likelihood(data_test=data_test, y=torch.tensor(y_grid, dtype=torch.float32))
        else:
            raise ValueError("bound_type must be either upper or lower")
        name = joint_model.stage2_upper[i].config["sensitivity_model"]["name"]
        sns.lineplot(x=y_grid.squeeze(), y=stage2_i.detach().numpy(), label="Stage 2" + name, color=palette[i], linewidth=3)
    if path is not None:
        plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    # Legend to lower left
    plt.legend(loc="lower left").set_visible(legend)
    plt.show()


def plot_sensitivity_value(scm, sensitivity_model, a=1, l=-0.7, r=0.7, stepsize=0.01, path=None):
    x = np.expand_dims(np.arange(l, r, stepsize), 1)
    if sensitivity_model in ["rosenbaum_binary", "rosenbaum_continuous"]:
        sensitivity_values = scm.get_rosenbaum_sensitivity_value(x=x, a=a)
    elif sensitivity_model in ["msm", "cmsm"]:
        sensitivity_values = scm.get_msm_sensitivity_value(x=x, a=a)
    else:
        sensitivity_values = scm.get_f_sensitivity_value(x=x, f=utils.get_f_by_name(sensitivity_model), a=a)
    # Create test data for validation of propensity fit
    plt.plot(x, sensitivity_values, label=sensitivity_model)
    ax = plt.gca()

    #ax.set_ylim([0, 3])
    plt.legend()
    if path is not None:
        plt.savefig(utils.get_project_path() + path, bbox_inches='tight')
    plt.show()

