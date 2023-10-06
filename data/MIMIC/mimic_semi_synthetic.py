import pandas as pd
import numpy as np
from scipy.special import expit
import utils.utils as utils
import h5py
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from models.propensity import PropensityNet
from data.data_structures import CausalDataset
import torch
from scipy.stats import norm

def real_world_data(config_data):
    X, A, Y, prop_obs, std_information = extract_mimic_data(config_data)
    scaling_y = [std_information["heart rate"], std_information["mean blood pressure"]]
    # Select outcomes (2d)
    Y = Y[:, [0, 2]]
    # Impute missing values
    Y[np.isnan(Y)] = np.nanmedian(Y)

    # Shuffle
    idx = np.random.permutation(X.shape[0])
    X = X[idx, :]
    A = A[idx, :]
    Y = Y[idx, :]

    # Split into train, val set
    n = X.shape[0]
    n_train = int(n * 0.8)
    if config_data["validation"]:
        n_val = int(n * 0.1)
    else:
        n_val = 0
    X_train = X[:n_train, :]
    X_val = X[n_train:n_train + n_val, :]
    X_test = X[n_train + n_val:, :]
    A_train = A[:n_train, :]
    A_val = A[n_train:n_train + n_val, :]
    A_test = A[n_train + n_val:, :]
    Y_train = Y[:n_train, :]
    Y_val = Y[n_train:n_train + n_val, :]
    Y_test = Y[n_train + n_val:, :]

    # Causal datasets
    d_train = CausalDataset(x=X_train, a=A_train, y=Y_train)
    d_val = CausalDataset(x=X_val, a=A_val, y=Y_val)
    d_test = CausalDataset(x=X_test, a=A_test, y=Y_test)

    return [d_train, d_val, d_test, scaling_y]


def semi_synthetic_data(config_data):
    X, A, _, prop_obs, _ = extract_mimic_data(config_data)
    # Uniform hidden confounding
    U = np.random.uniform(0, 1, size=(X.shape[0], 1))

    # Weight function for hidden confounding
    gamma = config_data["gamma"]
    w = np.where(gamma >= 2 - (1/prop_obs), gamma + U * 2 * (1 - gamma),  2 - (1/prop_obs) + U * 2 * (1/prop_obs - 1))
    # Sample synthetic treatments (follows observationa distribution
    A_syn = np.random.binomial(1, w * prop_obs, size=(X.shape[0], 1))

    XU_mean = np.mean(np.concatenate([X, U - 0.5], axis=1), axis=1, keepdims=True)
    Y_1_mean = XU_mean
    Y_0_mean = - XU_mean




    Y_1 = Y_1_mean + np.random.normal(0, 0.1, size=(X.shape[0], 1)) #+ U_lognorm
    Y_0 = Y_0_mean + np.random.normal(0, 0.1, size=(X.shape[0], 1)) #+ U_lognorm
    Y = A_syn * Y_1 + (1 - A_syn) * Y_0


    # Standardize outcomes
    y_mean = np.mean(Y)
    y_std = np.std(Y)
    Y_1 = (Y_1 - y_mean) / y_std
    Y_0 = (Y_0 - y_mean) / y_std
    Y = (Y - y_mean) / y_std

    # Compute ground-truth sensitivity parameters -----------------------------------
    # Sample confounders
    U = np.random.uniform(0, 1, size=(X.shape[0], 1000))
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
    U = np.random.uniform(0, 1, size=(X.shape[0], 100))
    w = np.where(gamma >= 2 - (1 / prop_obs), gamma + U * 2 * (1 - gamma),
                 2 - (1 / prop_obs) + U * 2 * (1 / prop_obs - 1))
    prop_full = w * prop_obs
    prop_full = np.expand_dims(prop_full, axis=2)
    prop_full2 = np.transpose(prop_full, axes=(0, 2, 1))
    odds_ratio2 = (prop_full / (1 - prop_full)) / (prop_full2 / (1 - prop_full2))
    gamma_rosenbaum = np.maximum(np.max(np.max(odds_ratio2, axis=2), axis=1, keepdims=True), np.max(np.max(1/odds_ratio2, axis=2), axis=1, keepdims=True))

    # Create datasets
    # Shuffle data
    idx = np.random.permutation(X.shape[0])
    X = X[idx, :]
    A_syn = A_syn[idx, :]
    Y = Y[idx, :]
    Y_1 = Y_1[idx, :]
    Y_0 = Y_0[idx, :]
    prop_obs = prop_obs[idx, :]
    gamma_msm = gamma_msm[idx, :]
    gamma_kl = gamma_kl[idx, :]
    gamma_tv = gamma_tv[idx, :]
    gamma_hellinger = gamma_hellinger[idx, :]
    gamma_chisquared = gamma_chisquared[idx, :]
    gamma_rosenbaum = gamma_rosenbaum[idx, :]
    # Split into train, val set
    n = X.shape[0]
    n_train = int(n * 0.8)
    if config_data["validation"]:
        n_val = int(n * 0.1)
    else:
        n_val = 0
    X_train = X[:n_train, :]
    X_val = X[n_train:n_train + n_val, :]
    X_test = X[n_train + n_val:, :]
    A_train = A_syn[:n_train, :]
    A_val = A_syn[n_train:n_train + n_val, :]
    A_test = A_syn[n_train + n_val:, :]
    Y_train = Y[:n_train, :]
    Y_val = Y[n_train:n_train + n_val, :]
    Y_test = Y[n_train + n_val:, :]
    Y_1_test = Y_1[n_train + n_val:, :]
    Y_0_test = Y_0[n_train + n_val:, :]
    prop_obs_train = prop_obs[:n_train, :]
    prop_obs_val = prop_obs[n_train:n_train + n_val, :]
    prop_obs_test = prop_obs[n_train + n_val:, :]
    gamma_msm_train = gamma_msm[:n_train, :]
    gamma_msm_val = gamma_msm[n_train:n_train + n_val, :]
    gamma_msm_test = gamma_msm[n_train + n_val:, :]
    gamma_kl_train = gamma_kl[:n_train, :]
    gamma_kl_val = gamma_kl[n_train:n_train + n_val, :]
    gamma_kl_test = gamma_kl[n_train + n_val:, :]
    gamma_tv_train = gamma_tv[:n_train, :]
    gamma_tv_val = gamma_tv[n_train:n_train + n_val, :]
    gamma_tv_test = gamma_tv[n_train + n_val:, :]
    gamma_hellinger_train = gamma_hellinger[:n_train, :]
    gamma_hellinger_val = gamma_hellinger[n_train:n_train + n_val, :]
    gamma_hellinger_test = gamma_hellinger[n_train + n_val:, :]
    gamma_chisquared_train = gamma_chisquared[:n_train, :]
    gamma_chisquared_val = gamma_chisquared[n_train:n_train + n_val, :]
    gamma_chisquared_test = gamma_chisquared[n_train + n_val:, :]
    gamma_rosenbaum_train = gamma_rosenbaum[:n_train, :]
    gamma_rosenbaum_val = gamma_rosenbaum[n_train:n_train + n_val, :]
    gamma_rosenbaum_test = gamma_rosenbaum[n_train + n_val:, :]
    gammas_train = {"msm": gamma_msm_train, "kl_binary": gamma_kl_train, "tv_binary": gamma_tv_train,
                    "hellinger_binary": gamma_hellinger_train, "chisquared_binary": gamma_chisquared_train,
                    "rosenbaum": gamma_rosenbaum_train}
    gammas_val = {"msm": gamma_msm_val, "kl_binary": gamma_kl_val, "tv_binary": gamma_tv_val,
                    "hellinger_binary": gamma_hellinger_val, "chisquared_binary": gamma_chisquared_val,
                    "rosenbaum": gamma_rosenbaum_val}
    gammas_test = {"msm": gamma_msm_test, "kl_binary": gamma_kl_test, "tv_binary": gamma_tv_test,
                    "hellinger_binary": gamma_hellinger_test, "chisquared_binary": gamma_chisquared_test,
                    "rosenbaum": gamma_rosenbaum_test}
    # Causal datasets
    d_train = CausalDataset(x=X_train, a=A_train, y=Y_train, propensity=prop_obs_train, gammas=gammas_train)
    d_val = CausalDataset(x=X_val, a=A_val, y=Y_val, propensity=prop_obs_val, gammas=gammas_val)
    d_test = CausalDataset(x=X_test, a=A_test, y=Y_test, propensity=prop_obs_test, gammas=gammas_test)

    return [d_train, d_val, d_test, Y_1_test, Y_0_test, y_mean, y_std]



def extract_mimic_data(config_data):
    vital_list = config_data["dynamic_cov"]
    static_list = config_data["static_cov"]

    data_path = utils.get_project_path() + "/data/MIMIC/all_hourly_data.h5"
    h5 = pd.HDFStore(data_path, 'r')

    all_vitals = h5['/vitals_labs_mean'][vital_list]
    all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])
    static_features = h5['/patients'][static_list]
    static_features = static_features.droplevel(['hadm_id', 'icustay_id'])
    #Treatment
    treatment = h5['/interventions'][['vent']]

    column_names_vitals = []
    for column in all_vitals.columns:
        if isinstance(column, str):
            column_names_vitals.append(column)
        else:
            column_names_vitals.append(column[0])
    all_vitals.columns = column_names_vitals

    # Filtering out users with time length < necissary time length
    user_sizes = all_vitals.groupby('subject_id').size()
    filtered_users_len = user_sizes.index[user_sizes > config_data["cov_window"] + config_data["treat_window"] + config_data["out_window"]]

    # Filtering out users with time age > 100
    if static_list is not None:
        if "age" in static_list:
            filtered_users_age = static_features.index[static_features.age < 100]
            filtered_users = filtered_users_len.intersection(filtered_users_age)
        else:
            filtered_users = filtered_users_len
    else:
        filtered_users = filtered_users_len


    #filtered_users = np.random.choice(filtered_users, size=n, replace=False)
    all_vitals = all_vitals.loc[filtered_users]

    # Split time-series into pre-treatment, treatmet, post-treatment part, and take mean -> static dataset
    vitals_pretreat = []
    vitals_out = []
    treatment_grouped = treatment.groupby('subject_id')
    treatment_pretreat = []
    treatment_treat = []
    for i, cov in enumerate(all_vitals.groupby('subject_id')):
        test = cov[1].to_numpy()
        T = test.shape[0]
        # sample random treatment time point
        t = np.random.randint(config_data["cov_window"], T-config_data["treat_window"] - config_data["out_window"])
        t_start = t - config_data["cov_window"]
        t_treat_end = t + config_data["treat_window"]


        # get ith treatment group as a numpy array
        treatment_i = treatment_grouped.get_group(cov[0]).to_numpy()

        treatment_pretreat.append(np.nanmean(treatment_i[t_start:t, :], axis=0, keepdims=True))
        if np.mean(treatment_i[t:t_treat_end, :], axis=0) > 0:
            # Add 1 to treatment_treat
            treatment_treat.append(np.ones((1, 1)))
        else:
            treatment_treat.append(np.zeros((1, 1)))


        t_out_end = t + config_data["treat_window"] + config_data["out_window"]
        vitals_pretreat.append(np.nanmean(test[t_start:t, :], axis=0, keepdims=True))
        vitals_out.append(np.nanmean(test[t_treat_end:t_out_end, :], axis=0, keepdims=True))



    vitals_pretreat = np.concatenate(vitals_pretreat, axis=0)
    vitals_pretreat = pd.DataFrame(vitals_pretreat, columns=column_names_vitals)
    # Set indices to subject_id
    vitals_pretreat.index = filtered_users
    vitals_out = np.concatenate(vitals_out, axis=0)
    vitals_out = pd.DataFrame(vitals_out, columns=column_names_vitals)
    vitals_out.index = filtered_users
    treatment_pretreat = np.concatenate(treatment_pretreat, axis=0)
    treatment_pretreat = pd.DataFrame(treatment_pretreat, columns=['vent'])
    treatment_pretreat.index = filtered_users
    treatment_treat = np.concatenate(treatment_treat, axis=0)
    treatment_treat = pd.DataFrame(treatment_treat, columns=['vent'])
    treatment_treat.index = filtered_users

    # One-hot encoding/ Standardization for static covariates
    static_features = static_features.loc[filtered_users]
    # Standardize age
    mean = np.mean(static_features["age"])
    std = np.std(static_features["age"])
    static_features["age"] = (static_features["age"] - mean) / std
    # set gender to 1 if "M", otherwise 0
    static_features["gender"] = np.where(static_features["gender"] == "M", 1, 0)

    # Get indices of rows with missing values
    idx = vitals_pretreat.index[vitals_pretreat.isnull().any(axis=1)]
    # Drop rows with missing values
    vitals_pretreat = vitals_pretreat.drop(idx)
    vitals_out = vitals_out.drop(idx)
    treatment_treat = treatment_treat.drop(idx)
    treatment_pretreat = treatment_pretreat.drop(idx)
    static_features = static_features.drop(idx)
    # Remove outliers
    for column in vitals_pretreat.columns:
        # Get indices of all rows that are below 0.1 percentile or above 99.9 percentile
        col = vitals_pretreat[column].to_numpy()
        quant_low = np.quantile(col, 0.001)
        quant_high = np.quantile(col, 0.999)
        # Get indices
        idx = vitals_pretreat.index[(col < quant_low) | (col > quant_high)]
        # Remove rows
        vitals_pretreat = vitals_pretreat.drop(idx)
        vitals_out = vitals_out.drop(idx)
        treatment_treat = treatment_treat.drop(idx)
        treatment_pretreat = treatment_pretreat.drop(idx)
        static_features = static_features.drop(idx)

    #Standardize covariates
    std_information = {}
    for column in vitals_pretreat.columns:
        mean = np.mean(vitals_pretreat[column])
        std = np.std(vitals_pretreat[column])
        vitals_pretreat[column] = (vitals_pretreat[column] - mean) / std
        vitals_out[column] = (vitals_out[column] - mean) / std
        std_information[column] = [mean, std]

    # concat static features, vitals and treatments pre-treatment
    X = pd.concat([static_features, vitals_pretreat], axis=1, join="inner")
    A_full = treatment_treat.to_numpy()
    Y_full = vitals_out.to_numpy()
    h5.close()

    # Introduce hidden confounding by dropping columns (specified in config)
    #X_obs = X.copy()
    #X_obs = X_obs.drop(columns=config_data["hidden_conf"]).to_numpy()
    X_full = X.to_numpy()


    treat_rate = np.mean(A_full)
    # Split into train, val set
    n = X.shape[0]
    n_train = int(n * 0.8)

    idx = np.random.permutation(n)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:]
    X_full_train = X_full[idx_train, :]
    X_full_val = X_full[idx_val, :]
    A_train = A_full[idx_train, :]
    A_val = A_full[idx_val, :]
    Y_train = Y_full[idx_train, :]
    Y_val = Y_full[idx_val, :]
    # Causal datasets
    d_full_train = CausalDataset(x=X_full_train, a=A_train, y=Y_train)
    d_full_val = CausalDataset(x=X_full_val, a=A_val, y=Y_val)

    # Configuration for propensity nets
    # Base config for propensity
    data_dims_full = d_full_train.get_dims(["x", "a", "y"])
    config_full = {"dim_input": data_dims_full["x"], "dim_output": data_dims_full["a"],
                   "out_type": "discrete", "neptune": True} | utils.load_yaml("/data/MIMIC/propensity")
    # Train NN for full and observed propensity
    propensity_net_full = PropensityNet(config_full)
    #propensity_net_obs = PropensityNet(config_obs)
    if config_data["train_propensities"]:
        propensity_net_full.fit(d_full_train, d_full_val, name="propensity_full")
        utils.save_pytorch_model("/data/MIMIC/propensity_full", propensity_net_full)
    else:
        propensity_net_full = utils.load_pytorch_model("/data/MIMIC/propensity_full", propensity_net_full)

    # Predictions
    propensity_full_pred = propensity_net_full.predict(torch.tensor(X_full, dtype=torch.float)).detach().numpy()
    return X_full, A_full, Y_full, propensity_full_pred, std_information
