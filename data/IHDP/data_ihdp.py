import torch
import pyreadr
import numpy as np
import utils.utils as utils
import random
from sklearn import preprocessing
from sklearn import model_selection
from data.data_structures import CausalDataset
from models.propensity import PropensityNet


# Code adjusted from: Jesson et al. 2022, "Scalable Sensitivity and Uncertainty Analyses for
# Causal-Effect Estimates of Continuous-Valued Interventions"

def load_data_ihdp(config_data):
    cov_continuous = config_data["cov_continuous"]
    cov_binary = config_data["cov_binary"]
    cov_hidden = config_data["cov_hidden"]
    # Load data
    data_path = utils.get_project_path() + "/data/IHDP/ihdp.RData"
    df = pyreadr.read_r(data_path)["ihdp"]

    # Make observational as per Hill 2011
    # df = df[~((df["treat"] == 1) & (df["momwhite"] == 0))]
    # df = df[
    #    cov_continuous + cov_binary + cov_hidden + ["treat"]
    #    ]
    # Standardize continuous covariates
    df[cov_continuous] = preprocessing.StandardScaler().fit_transform(
        df[cov_continuous]
    )

    # Create observational policy (propensity score)
    covs = cov_continuous + cov_binary
    x = df[covs].to_numpy(dtype="float32")

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    score = 3 * np.mean(x, axis=1, keepdims=True)
    p_a1_x = sigmoid(score)
    # Prohibit overlap violations
    # p_a1_x = np.clip(p_a1_x, 0.05, 0.95)
    prop_obs = 0.25 + 0.5 * p_a1_x

    # Generate confounded treatments
    # Uniform hidden confounding
    U = np.random.uniform(0, 1, size=(x.shape[0], 1))

    # Weight function for hidden confounding
    gamma = config_data["gamma"]
    w = np.where(gamma >= 2 - (1 / prop_obs), gamma + U * 2 * (1 - gamma),
                 2 - (1 / prop_obs) + U * 2 * (1 / prop_obs - 1))
    # Sample synthetic treatments (follows observationa distribution
    A_syn = np.random.binomial(1, w * prop_obs, size=(x.shape[0], 1))
    # Generate response surfaces
    seed = random.randint(0, 1000000)
    rng = np.random.default_rng(seed)
    beta_x = rng.choice(
        [0.0, 0.1, 0.2, 0.3, 0.4], size=(24,), p=[0.6, 0.1, 0.1, 0.1, 0.1]
    )
    beta_u = (
        rng.choice(
            [0.1, 0.2, 0.3, 0.4, 0.5], size=(1,), p=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        if config_data["beta_u"] is None
        else np.asarray([config_data["beta_u"]])
    )
    mu0 = np.exp((x + 0.5).dot(beta_x) + (U + 0.5).dot(beta_u))
    # mu0 = 0
    df["mu0"] = mu0
    mu1 = (x + 0.5).dot(beta_x) + (U + 0.5).dot(beta_u)
    omega = (mu1[np.squeeze(A_syn) == 1] - mu0[np.squeeze(A_syn) == 1]).mean(0) - 4
    mu1 -= omega
    df["mu1"] = mu1
    eps = rng.normal(size=A_syn.shape, scale=1, loc=0.0)
    y0 = mu0 + np.squeeze(eps)
    y1 = mu1 + np.squeeze(eps)
    y = np.squeeze(A_syn) * y1 + (1 - np.squeeze(A_syn)) * y0
    # Standardize outcomes
    y_mean = np.mean(y)
    y_std = np.std(y)
    y1 = (y1 - y_mean) / y_std
    y0 = (y0 - y_mean) / y_std
    y = (y - y_mean) / y_std

    # Compute oracle sensitivity parameters for different sensitivity models
    # Compute ground-truth sensitivity parameters -----------------------------------
    # Sample confounders
    U = np.random.uniform(0, 1, size=(x.shape[0], 1000))
    w = np.where(gamma >= 2 - (1 / prop_obs), gamma + U * 2 * (1 - gamma),
                 2 - (1 / prop_obs) + U * 2 * (1 / prop_obs - 1))
    prop_full = w * prop_obs
    odds_ratio = (prop_obs / (1 - prop_obs)) / (prop_full / (1 - prop_full))
    odds_ratio_inv = 1 / odds_ratio
    # MSM
    gamma_msm = np.maximum(np.max(odds_ratio, axis=1, keepdims=True), np.max(odds_ratio_inv, axis=1, keepdims=True))
    # KL
    f = utils.get_f_by_name("kl_binary")
    gamma_kl_1 = np.mean(f(odds_ratio) * prop_full / prop_obs, axis=1, keepdims=True)
    gamma_kl_2 = np.mean(f(odds_ratio_inv) * prop_obs / prop_full, axis=1, keepdims=True)
    gamma_kl = np.maximum(gamma_kl_1, gamma_kl_2)
    # TV
    f = utils.get_f_by_name("tv_binary")
    gamma_tv_1 = np.mean(f(odds_ratio) * prop_full / prop_obs, axis=1, keepdims=True)
    gamma_tv_2 = np.mean(f(odds_ratio_inv) * prop_obs / prop_full, axis=1, keepdims=True)
    gamma_tv = np.maximum(gamma_tv_1, gamma_tv_2)
    # hellinger
    f = utils.get_f_by_name("hellinger_binary")
    gamma_hellinger_1 = np.mean(f(odds_ratio) * prop_full / prop_obs, axis=1, keepdims=True)
    gamma_hellinger_2 = np.mean(f(odds_ratio_inv) * prop_obs / prop_full, axis=1, keepdims=True)
    gamma_hellinger = np.maximum(gamma_hellinger_1, gamma_hellinger_2)
    # chisquared
    f = utils.get_f_by_name("chisquared_binary")
    gamma_chisquared_1 = np.mean(f(odds_ratio) * prop_full / prop_obs, axis=1, keepdims=True)
    gamma_chisquared_2 = np.mean(f(odds_ratio_inv) * prop_obs / prop_full, axis=1, keepdims=True)
    gamma_chisquared = np.maximum(gamma_chisquared_1, gamma_chisquared_2)
    # Rosenbaum
    U = np.random.uniform(0, 1, size=(x.shape[0], 100))
    w = np.where(gamma >= 2 - (1 / prop_obs), gamma + U * 2 * (1 - gamma),
                 2 - (1 / prop_obs) + U * 2 * (1 / prop_obs - 1))
    prop_full = w * prop_obs
    prop_full = np.expand_dims(prop_full, axis=2)
    prop_full2 = np.transpose(prop_full, axes=(0, 2, 1))
    odds_ratio2 = (prop_full / (1 - prop_full)) / (prop_full2 / (1 - prop_full2))
    gamma_rosenbaum = np.maximum(np.max(np.max(odds_ratio2, axis=2), axis=1, keepdims=True),
                                 np.max(np.max(1 / odds_ratio2, axis=2), axis=1, keepdims=True))

    # Add to dataframe
    df["y0"] = y0
    df["y1"] = y1
    df["prop_obs"] = prop_obs
    df["a"] = np.squeeze(A_syn)
    df["y"] = y
    df["gamma_msm"] = gamma_msm
    df["gamma_kl"] = gamma_kl
    df["gamma_tv"] = gamma_tv
    df["gamma_hellinger"] = gamma_hellinger
    df["gamma_chisquared"] = gamma_chisquared
    df["gamma_rosenbaum"] = gamma_rosenbaum



    # Train test split
    df_train, df_test = model_selection.train_test_split(
        df, test_size=0.2, random_state=seed
    )

    # Test data
    x_test = df_test[covs].to_numpy(dtype="float32")
    a_test = np.expand_dims(df_test["treat"].to_numpy(dtype="float32"), 1)
    y1_test = np.expand_dims(df_test["y1"].to_numpy(dtype="float32"), 1)
    y0_test = np.expand_dims(df_test["y0"].to_numpy(dtype="float32"), 1)
    y_test = np.expand_dims(df_test["y"].to_numpy(dtype="float32"), 1)
    prop_obs_test = np.expand_dims(df_test["prop_obs"].to_numpy(dtype="float32"), 1)
    gamma_msm_test = np.expand_dims(df_test["gamma_msm"].to_numpy(dtype="float32"), 1)
    gamma_kl_test = np.expand_dims(df_test["gamma_kl"].to_numpy(dtype="float32"), 1)
    gamma_tv_test = np.expand_dims(df_test["gamma_tv"].to_numpy(dtype="float32"), 1)
    gamma_hellinger_test = np.expand_dims(df_test["gamma_hellinger"].to_numpy(dtype="float32"), 1)
    gamma_chisquared_test = np.expand_dims(df_test["gamma_chisquared"].to_numpy(dtype="float32"), 1)
    gamma_rosenbaum_test = np.expand_dims(df_test["gamma_rosenbaum"].to_numpy(dtype="float32"), 1)
    gammas_test = {"msm": gamma_msm_test, "kl_binary": gamma_kl_test, "tv_binary": gamma_tv_test,
                    "hellinger_binary": gamma_hellinger_test, "chisquared_binary": gamma_chisquared_test,
                    "rosenbaum": gamma_rosenbaum_test}
    d_test = CausalDataset(x=x_test, a=a_test, y=y_test, propensity=prop_obs_test, gammas=gammas_test)


    # Validation set
    if config_data["validation"] == True:
        df_train, df_val = model_selection.train_test_split(
            df_train, test_size=0.3, random_state=seed
        )

        # Validation data
        x_val = df_val[covs].to_numpy(dtype="float32")
        a_val = np.expand_dims(df_val["treat"].to_numpy(dtype="float32"), 1)
        y_val = np.expand_dims(df_val["y"].to_numpy(dtype="float32"), 1)
        prop_obs_val = np.expand_dims(df_val["prop_obs"].to_numpy(dtype="float32"), 1)
        gamma_msm_val = np.expand_dims(df_val["gamma_msm"].to_numpy(dtype="float32"), 1)
        gamma_kl_val = np.expand_dims(df_val["gamma_kl"].to_numpy(dtype="float32"), 1)
        gamma_tv_val = np.expand_dims(df_val["gamma_tv"].to_numpy(dtype="float32"), 1)
        gamma_hellinger_val = np.expand_dims(df_val["gamma_hellinger"].to_numpy(dtype="float32"), 1)
        gamma_chisquared_val = np.expand_dims(df_val["gamma_chisquared"].to_numpy(dtype="float32"), 1)
        gamma_rosenbaum_val = np.expand_dims(df_val["gamma_rosenbaum"].to_numpy(dtype="float32"), 1)
        gammas_val = {"msm": gamma_msm_val, "kl_binary": gamma_kl_val, "tv_binary": gamma_tv_val,
                        "hellinger_binary": gamma_hellinger_val, "chisquared_binary": gamma_chisquared_val,
                        "rosenbaum": gamma_rosenbaum_val}
        d_val = CausalDataset(x=x_val, a=a_val, y=y_val, propensity=prop_obs_val, gammas=gammas_val)


    else:
        d_val = None

    # Training data
    x_train = df_train[covs].to_numpy(dtype="float32")
    a_train = np.expand_dims(df_train["treat"].to_numpy(dtype="float32"), 1)
    y_train = np.expand_dims(df_train["y"].to_numpy(dtype="float32"), 1)
    prop_obs_train = np.expand_dims(df_train["prop_obs"].to_numpy(dtype="float32"), 1)
    gamma_msm_train = np.expand_dims(df_train["gamma_msm"].to_numpy(dtype="float32"), 1)
    gamma_kl_train = np.expand_dims(df_train["gamma_kl"].to_numpy(dtype="float32"), 1)
    gamma_tv_train = np.expand_dims(df_train["gamma_tv"].to_numpy(dtype="float32"), 1)
    gamma_hellinger_train = np.expand_dims(df_train["gamma_hellinger"].to_numpy(dtype="float32"), 1)
    gamma_chisquared_train = np.expand_dims(df_train["gamma_chisquared"].to_numpy(dtype="float32"), 1)
    gamma_rosenbaum_train = np.expand_dims(df_train["gamma_rosenbaum"].to_numpy(dtype="float32"), 1)
    gammas_train = {"msm": gamma_msm_train, "kl_binary": gamma_kl_train, "tv_binary": gamma_tv_train,
                    "hellinger_binary": gamma_hellinger_train, "chisquared_binary": gamma_chisquared_train,
                    "rosenbaum": gamma_rosenbaum_train}
    d_train = CausalDataset(x=x_train, a=a_train, y=y_train, propensity=prop_obs_train, gammas=gammas_train)

    return [d_train, d_val, d_test, y1_test, y0_test, y_mean, y_std]


