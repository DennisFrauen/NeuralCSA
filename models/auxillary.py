from data.data_structures import CausalDataset
from models.stage1 import CNF_stage1
from models.stage2 import NF_stage2, CNF_stage2
from models.propensity import PropensityNet
import utils.utils as utils
import torch
import numpy as np


class JointModel:

    def __init__(self, config):
        self.config = config
        # Stage 1
        self.propensity = None
        if config["propensity"] is not None:
            self.propensity = PropensityNet(config["propensity"])
        self.stage1 = CNF_stage1(config["stage1"])
        # Stage 2
        self.stage2_upper = []
        self.stage2_lower = []
        for k in range(len(config["stage2_upper"])):
            config_k_upper = config["stage2_upper"][k]
            config_k_lower = config["stage2_lower"][k]
            # Sensitivity model and causal query
            sensitivity_model = SensitivityModel(config_k_upper["sensitivity_model"])
            causal_query = CausalQuery(config_k_upper["causal_query"])
            # Check weather NF or CNF should be trained
            if config_k_upper["dim_context"] == 0:
                self.stage2_upper.append(NF_stage2(config_k_upper | {"bound_type": "upper"}, trained_models={"propensity": self.propensity, "stage1": self.stage1},
                                        sensitivity_model=sensitivity_model, causal_query=causal_query))
                self.stage2_lower.append(NF_stage2(config_k_lower | {"bound_type": "lower"}, trained_models={"propensity": self.propensity, "stage1": self.stage1},
                                        sensitivity_model=sensitivity_model, causal_query=causal_query))
            else:
                self.stage2_upper.append(CNF_stage2(config_k_upper | {"bound_type": "upper"}, trained_models={"propensity": self.propensity, "stage1": self.stage1},
                                        sensitivity_model=sensitivity_model, causal_query=causal_query))
                self.stage2_lower.append(CNF_stage2(config_k_lower | {"bound_type": "lower"}, trained_models={"propensity": self.propensity, "stage1": self.stage1},
                                        sensitivity_model=sensitivity_model, causal_query=causal_query))

    def load_propensity(self, path):
        self.propensity = utils.load_pytorch_model(path, self.propensity)

    def load_stage1(self, path):
        self.stage1 = utils.load_pytorch_model(path, self.stage1)

    def load_stage2(self, path):
        for k in range(len(self.stage2_lower)):
            bounds = self.stage2_lower[k].config["bounds"]
            config_model = self.stage2_lower[k].config
            path_k_lower = (path + "lower/" + config_model["causal_query"]["name"] + "_" +
                            config_model["sensitivity_model"]["name"]
                            + "_" + str(config_model["sensitivity_model"]["gamma"]))
            path_k_upper = (path + "upper/" + config_model["causal_query"]["name"] + "_" +
                            config_model["sensitivity_model"]["name"]
                            + "_" + str(config_model["sensitivity_model"]["gamma"]))
            if bounds == "both":
                self.stage2_lower[k] = utils.load_pytorch_model(path_k_lower, self.stage2_lower[k])
                self.stage2_upper[k] = utils.load_pytorch_model(path_k_upper, self.stage2_upper[k])
            elif bounds == "upper":
                self.stage2_upper[k] = utils.load_pytorch_model(path_k_upper, self.stage2_upper[k])
            elif bounds == "lower":
                self.stage2_lower[k] = utils.load_pytorch_model(path_k_lower, self.stage2_lower[k])

    def fit_propensity(self, d_train, d_val=None):
        return self.propensity.fit(d_train, d_val, name="propensity")

    def fit_stage1(self, d_train, d_val=None):
        return self.stage1.fit(d_train, d_val, name="stage1")

    def fit_stage2(self, dataset, seed=0, savepath=None):
        fit_results = []
        for k in range(len(self.stage2_lower)):
            utils.set_seed(seed)
            print("Fitting stage 2 model " + str(k + 1) + " of " + str(len(self.stage2_lower)))
            config_model = self.stage2_lower[k].config
            bounds = self.stage2_lower[k].config["bounds"]
            if savepath is not None:
                path_k_lower = (savepath + "lower/" + config_model["causal_query"]["name"] + "_" +
                                config_model["sensitivity_model"]["name"]
                                + "_" + str(config_model["sensitivity_model"]["gamma"]))
                path_k_upper = (savepath + "upper/" + config_model["causal_query"]["name"] + "_" +
                                config_model["sensitivity_model"]["name"]
                                + "_" + str(config_model["sensitivity_model"]["gamma"]))
            if bounds == "lower":
                try:
                    fit_results_lower = self.stage2_lower[k].fit(dataset, name="stage2_lower_" + str(k))
                    fit_results.append({"lower": fit_results_lower})
                    if savepath is not None:
                        utils.save_pytorch_model(path_k_lower, self.stage2_lower[k])
                except ValueError as e:
                    print("Runtime error: " + str(e))
                    print("Skipping this model")
            elif bounds == "upper":
                try:
                    fit_results_upper = self.stage2_upper[k].fit(dataset, name="stage2_upper_" + str(k))
                    fit_results.append({"upper": fit_results_upper})
                    if savepath is not None:
                        utils.save_pytorch_model(path_k_upper, self.stage2_upper[k])
                except ValueError as e:
                    print("Runtime error: " + str(e))
                    print("Skipping this model")
            elif bounds == "both":
                try:
                    fit_results_lower = self.stage2_lower[k].fit(dataset, name="stage2_lower_" + str(k))
                    fit_results.append({"lower": fit_results_lower})
                    if savepath is not None:
                        utils.save_pytorch_model(path_k_lower, self.stage2_lower[k])
                except ValueError as e:
                    print("Runtime error: " + str(e))
                    print("Skipping this model")
                utils.set_seed(seed)
                try:
                    fit_results_upper = self.stage2_upper[k].fit(dataset, name="stage2_upper_" + str(k))
                    fit_results.append({"upper": fit_results_upper})
                    if savepath is not None:
                        utils.save_pytorch_model(path_k_upper, self.stage2_upper[k])
                except ValueError as e:
                    print("Runtime error: " + str(e))
                    print("Skipping this model")
            else:
                raise ValueError("Unknown bound type")
        return fit_results

    def save_propensity(self, path):
        utils.save_pytorch_model(path, self.propensity)

    def save_stage1(self, path):
        utils.save_pytorch_model(path, self.stage1)

    def save_stage2(self, path):
        for k in range(len(self.stage2_lower)):
            config_model = self.stage2_lower[k].config
            bounds = config_model["bounds"]
            path_k_lower = (path + "lower/" + config_model["causal_query"]["name"] + "_" +
                            config_model["sensitivity_model"]["name"]
                            + "_" + str(config_model["sensitivity_model"]["gamma"]))
            path_k_upper = (path + "upper/" + config_model["causal_query"]["name"] + "_" +
                            config_model["sensitivity_model"]["name"]
                            + "_" + str(config_model["sensitivity_model"]["gamma"]))
            if bounds == "lower":
                utils.save_pytorch_model(path_k_lower, self.stage2_lower[k])
            elif bounds == "upper":
                utils.save_pytorch_model(path_k_upper, self.stage2_upper[k])
            elif bounds == "both":
                utils.save_pytorch_model(path_k_lower, self.stage2_lower[k])
                utils.save_pytorch_model(path_k_upper, self.stage2_upper[k])
            else:
                raise ValueError("Unknown bound type")

    # Analytical MSM/ CMSM bounds for trained stage 1 model
    def GMSM_bounds(self, data_test, n_samples, gammas, sensitivity_model=None, q=None):
        if sensitivity_model is None:
            if data_test.datatypes["a_type"] == "discrete":
                sensitivity_model = "msm"
            else:
                sensitivity_model = "cmsm"
        #Check for binary or continuous treatment
        if sensitivity_model == "msm":
            a = data_test.data["a"].to(torch.int64)
            prop1 = self.propensity.predict(data_test.data["x"])
            prop = torch.concat((1 - prop1, prop1), dim=1)
            prop_a = prop.gather(1, a)
            if len(gammas.size()) == 0:
                gammas = gammas.repeat(prop_a.size(0), 1)
            s_minus = 1 / ((1 - gammas) * prop_a + gammas)
            s_plus = 1 / ((1 - (1 / gammas)) * prop_a + (1 / gammas))
        elif sensitivity_model == "cmsm":
            a = data_test.data["a"]
            if len(gammas.size()) == 0:
                gammas = gammas.repeat(a.size(0), 1)
            s_minus = 1 / gammas
            s_plus = gammas
        else:
            raise ValueError("Unknown sensitivity model")
        c_plus = ((1 - s_minus) * s_plus) / (s_plus - s_minus)
        c_minus = ((1 - s_plus) * s_minus) / (s_minus - s_plus)
        # fill nan values in c_plus and c_minus with 0 (this is the case when Gamma = 1, i.e., no unobserved confounding)
        c_plus[torch.isnan(c_plus)] = 0
        c_minus[torch.isnan(c_minus)] = 0
        if q is None:
            # Bounds for mean
            q_index_plus = torch.floor(c_plus * n_samples).long()
            q_index_minus = torch.floor(c_minus * n_samples).long()

            # Sample and sort
            y_samples = self.stage1.sample(data_test=data_test, n_samples=n_samples)
            y_samples = torch.sort(y_samples, dim=1)[0]

            # Compute cumulative sum
            y_cum = torch.cumsum(y_samples, dim=1)
            # Select indices corresponding to the gammas (size is n_data x n_gammas)
            y_cum_gamma_plus = y_cum.gather(dim=1, index=q_index_plus)
            y_cum_gamma_minus = y_cum.gather(dim=1, index=q_index_minus)
            # Comulative sum over all samples
            y_cum_all = y_cum[:, -1].unsqueeze(1)

            Q_plus = (y_cum_gamma_plus / (n_samples * s_plus)) + \
                      ((y_cum_all - y_cum_gamma_plus) / (n_samples * s_minus))
            # if right most qunatile reached, set to bound of support (formula above breaks down)
            idx_boundary = q_index_plus == n_samples - 1
            max_support = torch.max(y_samples, dim=1)[0].unsqueeze(1).repeat((1, Q_plus.size(1)))
            Q_plus[idx_boundary] = max_support[idx_boundary]

            Q_minus = (y_cum_gamma_minus / (n_samples * s_minus)) + \
                       ((y_cum_all - y_cum_gamma_minus) / (n_samples * s_plus))
        else:
            # Bounds for quantiles
            raise NotImplementedError
        return {"Q_plus": Q_plus, "Q_minus": Q_minus}


    # Return all test bounds of trained stage 2 models. If data_test_a2 is not None, return differences
    def get_all_bounds_diff_avg(self, data_test_a1, data_test_a2=None, n_samples=10000, average=False, bound_type="both"):
        # Get upper bounds for all stage 2 models
        query_a1_upper = []
        query_a1_lower = []
        query_a2_upper = []
        query_a2_lower = []
        if data_test_a2 is not None and bound_type != "both":
            raise ValueError("bound type must be 'both' if data_test_a2 is not None, ie., if we want to compute differences")

        for i in range(max(len(self.stage2_upper), len(self.stage2_lower))):
            if bound_type == "both":
                model_upper = self.stage2_upper[i]
                model_lower = self.stage2_lower[i]
                if model_upper is not None:
                    query_a1_upper.append(
                        np.expand_dims(model_upper.test_query(data_test=data_test_a1, n_samples=n_samples).detach().numpy(),
                                       1))
                else:
                    query_a1_upper.append(np.full((data_test_a1.data["x"].shape[0], 1, 1), np.nan))
                if model_lower is not None:
                    query_a1_lower.append(
                        np.expand_dims(model_lower.test_query(data_test=data_test_a1, n_samples=n_samples).detach().numpy(),
                                       1))
                else:
                    query_a1_lower.append(np.full((data_test_a1.data["x"].shape[0], 1, 1), np.nan))
                if data_test_a2 is not None:
                    if model_upper is not None:
                        query_a2_upper.append(np.expand_dims(
                            model_upper.test_query(data_test=data_test_a2, n_samples=n_samples).detach().numpy(), 1))
                    else:
                        query_a2_upper.append(np.full((data_test_a1.data["x"].shape[0], 1, 1), np.nan))
                    if model_lower is not None:
                        query_a2_lower.append(np.expand_dims(
                            model_lower.test_query(data_test=data_test_a2, n_samples=n_samples).detach().numpy(), 1))
                    else:
                        query_a2_lower.append(np.full((data_test_a1.data["x"].shape[0], 1, 1), np.nan))
            elif bound_type == "upper":
                model_upper = self.stage2_upper[i]
                if model_upper is not None:
                    query_a1_upper.append(
                        np.expand_dims(model_upper.test_query(data_test=data_test_a1, n_samples=n_samples).detach().numpy(),
                                       1))
                else:
                    query_a1_upper.append(np.nan)
            elif bound_type == "lower":
                model_lower = self.stage2_lower[i]
                if model_lower is not None:
                    query_a1_lower.append(model_lower.test_query(data_test=data_test_a1, n_samples=n_samples).numpy())
                else:
                    query_a1_lower.append(np.nan)
            else:
                raise ValueError("bounds must be one of 'both', 'upper' or 'lower'")

        query_upper = []
        query_lower = []
        if bound_type == "both":
            if data_test_a2 is None:
                query_upper = query_a1_upper
                query_lower = query_a1_lower
            else:
                query_upper = [query_a1_upper[i] - query_a2_lower[i] for i in range(len(query_a1_upper))]
                query_lower = [query_a1_lower[i] - query_a2_upper[i] for i in range(len(query_a1_lower))]

            if average:
                query_upper = np.mean(query_upper, keepdims=True)
                query_lower = np.mean(query_lower, keepdims=True)
        elif bound_type == "upper":
            query_upper = query_a1_upper
            if average:
                query_upper = np.mean(query_upper, keepdims=True)
        elif bound_type == "lower":
            query_lower = query_a1_lower
            if average:
                query_lower = np.mean(query_lower, keepdims=True)

        return {"upper": query_upper, "lower": query_lower}

    # Return all test bounds of GMSM corresponding to trained stage 2 models. If data_test_a2 is not None, return differences
    def get_all_bounds_diff_avg_gmsm(self, data_test_a1, data_test_a2=None, n_samples=10000, average=False, gmsm_type=None):
        # GMSM bounds
        gmsm_bounds_a1_upper = []
        gmsm_bounds_a1_lower = []
        gmsm_bounds_a2_upper = []
        gmsm_bounds_a2_lower = []
        for i in range(len(self.stage2_upper)):
            gamma = self.stage2_upper[i].config["sensitivity_model"]["gamma"]
            if isinstance(gamma, str):
                # Use oracle gammas
                gamma = data_test_a1.data[self.stage2_upper[i].config["sensitivity_model"]["name"]]
            if gmsm_type is None:
                gmsm_type = self.stage2_upper[i].config["sensitivity_model"]["name"]
            Q_gmsm = self.GMSM_bounds(data_test_a1, n_samples=n_samples, gammas=torch.tensor(gamma), sensitivity_model=gmsm_type)
            gmsm_bounds_a1_upper.append(Q_gmsm["Q_plus"].detach().numpy())
            gmsm_bounds_a1_lower.append(Q_gmsm["Q_minus"].detach().numpy())
            if data_test_a2 is not None:
                Q_gmsm = self.GMSM_bounds(data_test_a2, n_samples=n_samples, gammas=torch.tensor(gamma), sensitivity_model=gmsm_type)
                gmsm_bounds_a2_upper.append(Q_gmsm["Q_plus"].detach().numpy())
                gmsm_bounds_a2_lower.append(Q_gmsm["Q_minus"].detach().numpy())
        if data_test_a2 is not None:
            gmsm_bounds_upper = [gmsm_bounds_a1_upper[i] - gmsm_bounds_a2_lower[i] for i in
                                 range(len(gmsm_bounds_a1_upper))]
            gmsm_bounds_lower = [gmsm_bounds_a1_lower[i] - gmsm_bounds_a2_upper[i] for i in
                                 range(len(gmsm_bounds_a1_lower))]
        else:
            gmsm_bounds_upper = gmsm_bounds_a1_upper
            gmsm_bounds_lower = gmsm_bounds_a1_lower

        if average:
            gmsm_bounds_upper = np.mean(gmsm_bounds_upper, keepdims=True)
            gmsm_bounds_lower = np.mean(gmsm_bounds_lower, keepdims=True)

        return {"upper": gmsm_bounds_upper, "lower": gmsm_bounds_lower}




class SensitivityModel:
    def __init__(self, config):
        self.gamma = config["gamma"]
        self.name = config["name"]
        if config["name"] == "msm":
            self.sensitivity_constraint = None
        elif config["name"] == "cmsm":
            self.sensitivity_constraint = SensitivityModel.sensitivity_constraint_cmsm
        elif config["name"] == "kl_binary":
            self.sensitivity_constraint = SensitivityModel.sensitivity_constraint_kl_binary
        elif config["name"] == "kl_continuous":
            self.sensitivity_constraint = SensitivityModel.sensitivity_constraint_kl_continuous
        elif config["name"] == "tv_binary":
            self.sensitivity_constraint = SensitivityModel.sensitivity_constraint_tv_binary
        elif config["name"] == "tv_continuous":
            self.sensitivity_constraint = SensitivityModel.sensitivity_constraint_tv_continuous
        elif config["name"] == "hellinger_binary":
            self.sensitivity_constraint = SensitivityModel.sensitivity_constraint_hellinger_binary
        elif config["name"] == "hellinger_continuous":
            self.sensitivity_constraint = SensitivityModel.sensitivity_constraint_hellinger_continuous
        elif config["name"] == "chisquared_binary":
            self.sensitivity_constraint = SensitivityModel.sensitivity_constraint_chisquared_binary
        elif config["name"] == "chisquared_continuous":
            self.sensitivity_constraint = SensitivityModel.sensitivity_constraint_chisquared_continuous
        elif config["name"] == "rosenbaum_binary":
            self.sensitivity_constraint = None
        elif config["name"] == "rosenbaum_continuous":
            self.sensitivity_constraint = None
        else:
            raise ValueError("Unknown sensitivity model")

        # Check for unconstrained model (gamma >= 1000)
        if not isinstance(self.gamma, str):
            if self.gamma >= 1000:
                self.sensitivity_constraint = SensitivityModel.unconstrained

    # Input: probabilities p(u|x, a) and p(u|x) of shape (n_u, n_data)
    #     propensity of shape (n_data, dim_a)
    # Output: violation of sensitivity constraint, shape is (n_data, n_gamma)
    def sensitivity_violation(self, u_sampled, xi_sampled, dist_base, dist_shifted, transform2, train_batch):
        propensity = train_batch["propensity"]
        # Check for oracle gamma available, if yes set gamma to oracles
        if self.name in train_batch.keys():
            self.gamma = train_batch[self.name]
        # Constraint is of shape (n_data, )
        if self.name =="msm":
            u_shifted = transform2(u_sampled)
            u_shifted = (1 - xi_sampled) * u_shifted + xi_sampled * u_sampled
            p_u_x = dist_shifted.log_prob(u_shifted).exp()
            p_u_xa = dist_base.log_prob(u_shifted).exp()
            ratio_max = torch.transpose(torch.max(p_u_xa / p_u_x, dim=0, keepdim=True)[0], 0, 1)
            ratio_min = torch.transpose(torch.min(p_u_xa / p_u_x, dim=0, keepdim=True)[0], 0, 1)
            s_minus = 1 / ((1 - self.gamma) * propensity + self.gamma)
            s_plus = 1 / ((1 - (1 / self.gamma)) * propensity + (1 / self.gamma))
            return torch.minimum(s_plus - ratio_max, ratio_min - s_minus)
        elif self.name == "rosenbaum_binary":
            u_shifted = transform2(u_sampled)
            u_shifted = (1 - xi_sampled) * u_shifted + xi_sampled * u_sampled
            p_u_x = dist_shifted.log_prob(u_shifted).exp()
            p_u_xa = dist_base.log_prob(u_shifted).exp()
            ratio = p_u_x / p_u_xa
            prod1 = 1 / (ratio - torch.transpose(propensity, 0, 1))
            prod2 = ratio - torch.transpose(propensity, 0, 1)
            prod1_ex = prod1.unsqueeze(1)  # Shape: (n, 1, p)
            prod2_ex = prod2.unsqueeze(0) # Shape (1, n, p)
            prod = prod1_ex * prod2_ex # Shape (n, n, p), all possible combinations of products
            # maximize over first two dimensions
            prod_max = torch.max(torch.max(prod, dim=0)[0], dim=0, keepdim=True)[0]
            prod_min = torch.max(torch.max(1/prod, dim=0)[0], dim=0, keepdim=True)[0]
            return self.gamma - torch.transpose(torch.maximum(prod_max, prod_min), 0, 1)
        elif self.name == "rosenbaum_continuous":
            raise NotImplementedError("Rosenbaum continuous not implemented")
        else:
            p_u_x = dist_shifted.log_prob(u_sampled).exp()
            p_u_xa = dist_base.log_prob(u_sampled).exp()
            constraint = self.sensitivity_constraint(p_u_x, p_u_xa, propensity)
            return self.gamma - torch.transpose(constraint, 0, 1)

    def sensitivity_value(self, p_u_x, p_u_xa, propensity):
        # Constraint is of shape (n_data, )
        constraint = self.sensitivity_constraint(p_u_x, p_u_xa, propensity)
        return torch.transpose(constraint, 0, 1)

    @staticmethod
    def unconstrained(p_u_x, p_u_xa, propensity):
        return torch.zeros(1, propensity.size(0))

    # transforming density ratio to msm type logg-odds ratio
    # Input: density ratio of shape (n_u, n_data), propensity of shape (n_data, 1)
    # Output: msm type logg-odds ratio of shape (n_u, n_data)
    @staticmethod
    def msm_transformation(density_ratio, propensity):
        # transpose propensity to shape (1, n_data)
        propensity = torch.transpose(propensity, 0, 1)
        return (density_ratio - propensity) / (1 - propensity)

    @staticmethod
    def sensitivity_constraint_msm(p_u_x, p_u_xa, propensity):
        ratio_msm = SensitivityModel.msm_transformation(p_u_x / p_u_xa, propensity)
        max_ratio1 = torch.max(ratio_msm, dim=0, keepdim=True)[0]
        max_ratio2 = torch.max(1 / ratio_msm, dim=0, keepdim=True)[0]
        return torch.maximum(max_ratio1, max_ratio2)

    @staticmethod
    def sensitivity_constraint_cmsm(p_u_x, p_u_xa, propensity):
        max_ratio1 = torch.max(p_u_x / p_u_xa, dim=0, keepdim=True)[0]
        max_ratio2 = torch.max(p_u_xa / p_u_x, dim=0, keepdim=True)[0]
        return torch.maximum(max_ratio1, max_ratio2)

    # f sensitivity models
    @staticmethod
    def sensitivity_constraint_f_binary(p_u_x, p_u_xa, propensity, f):
        ratio_msm = SensitivityModel.msm_transformation(p_u_x / p_u_xa, propensity)
        ratio1 = torch.mean(f(ratio_msm), dim=0, keepdim=True)
        ratio2 = torch.mean(f(1 / ratio_msm), dim=0, keepdim=True)
        return torch.maximum(ratio1, ratio2)

    @staticmethod
    def sensitivity_constraint_f_continuous(p_u_x, p_u_xa, propensity, f):
        ratio1 = torch.mean(f(p_u_x / p_u_xa), dim=0, keepdim=True)
        ratio2 = torch.mean(f(p_u_xa / p_u_x), dim=0, keepdim=True)
        return torch.maximum(ratio1, ratio2)

    @staticmethod
    def f_kl(x):
        return x * torch.log(x)

    @staticmethod
    def f_tv(x):
        return torch.abs(x - 1) * 0.5

    @staticmethod
    def f_hellinger(x):
        return (torch.sqrt(x) - 1) ** 2

    @staticmethod
    def chisquared(x):
        return (x - 1) ** 2

    @staticmethod
    def sensitivity_constraint_kl_binary(p_u_x, p_u_xa, propensity):
        return SensitivityModel.sensitivity_constraint_f_binary(p_u_x, p_u_xa, propensity, SensitivityModel.f_kl)

    @staticmethod
    def sensitivity_constraint_tv_binary(p_u_x, p_u_xa, propensity):
        return SensitivityModel.sensitivity_constraint_f_binary(p_u_x, p_u_xa, propensity, SensitivityModel.f_tv)

    @staticmethod
    def sensitivity_constraint_hellinger_binary(p_u_x, p_u_xa, propensity):
        return SensitivityModel.sensitivity_constraint_f_binary(p_u_x, p_u_xa, propensity, SensitivityModel.f_hellinger)

    @staticmethod
    def sensitivity_constraint_chisquared_binary(p_u_x, p_u_xa, propensity):
        return SensitivityModel.sensitivity_constraint_f_binary(p_u_x, p_u_xa, propensity, SensitivityModel.chisquared)

    @staticmethod
    def sensitivity_constraint_kl_continuous(p_u_x, p_u_xa, propensity):
        return SensitivityModel.sensitivity_constraint_f_continuous(p_u_x, p_u_xa, propensity, SensitivityModel.f_kl)

    @staticmethod
    def sensitivity_constraint_tv_continuous(p_u_x, p_u_xa, propensity):
        return SensitivityModel.sensitivity_constraint_f_continuous(p_u_x, p_u_xa, propensity, SensitivityModel.f_tv)

    @staticmethod
    def sensitivity_constraint_hellinger_continuous(p_u_x, p_u_xa, propensity):
        return SensitivityModel.sensitivity_constraint_f_continuous(p_u_x, p_u_xa, propensity,
                                                                    SensitivityModel.f_hellinger)

    @staticmethod
    def sensitivity_constraint_chisquared_continuous(p_u_x, p_u_xa, propensity):
        return SensitivityModel.sensitivity_constraint_f_continuous(p_u_x, p_u_xa, propensity,
                                                                    SensitivityModel.chisquared)


class CausalQuery:
    def __init__(self, config):
        if config["name"] == "expectation":
            self.query_computation_train = self.expectation_train
            self.query_computation_test = self.expectation_test
        elif config["name"] == "probability":
            self.query_computation_train = self.probability_train
            self.query_computation_test = self.probability_test
            # Boundary upper/ lower are lists of length d_y (componentwise boundaries for probability)
            self.boundary_upper = torch.tensor(config["boundary_upper"], dtype=torch.float32)
            self.boundary_lower = torch.tensor(config["boundary_lower"], dtype=torch.float32)
        elif config["name"] == "probability_distance":
            self.query_computation_train = self.probability_distance_train
            # Boundary upper/ lower are lists of length d_y (componentwise boundaries for probability)
            self.boundary_upper = torch.tensor(config["boundary_upper"], dtype=torch.float32)
            self.boundary_lower = torch.tensor(config["boundary_lower"], dtype=torch.float32)

        else:
            raise ValueError("Unknown causal query")

    def compute_query_train(self, u_sampled, xi_sampled, dist_y_shifted, transform1, transform2):
        return self.query_computation_train(u_sampled, xi_sampled, dist_y_shifted, transform1, transform2)

    def compute_query_test(self, y_samples_shifted):
        return self.query_computation_test(y_samples_shifted)

    def expectation_train(self, u_sampled, xi_sampled, dist_y_shifted, transform1, transform2):
        u_shifted = transform2(u_sampled)
        u_shifted = (1 - xi_sampled) * u_shifted + xi_sampled * u_sampled
        y_sampled = transform1(u_shifted)
        return torch.squeeze(torch.mean(y_sampled))

    def expectation_test(self, y_samples_shifted):
        return torch.squeeze(torch.mean(y_samples_shifted, dim=(0, 2)))

    def probability_train(self, u_sampled, xi_sampled, dist_y_shifted, transform1, transform2):
        with torch.no_grad():
            y_sampled = transform1(u_sampled)
            conditions_met = ((y_sampled >= self.boundary_lower) & (y_sampled <= self.boundary_upper)).all(dim=2)
            y_signs = torch.where(conditions_met, torch.ones((y_sampled.size(0), y_sampled.size(1))),
                                  -torch.zeros((y_sampled.size(0), y_sampled.size(1)))).detach()
            weight_plus = torch.sum(torch.where(conditions_met, torch.ones(y_signs.size()), torch.zeros(y_signs.size())), dim=0)
            weight_minus = y_sampled.size(0) - weight_plus
            weights = torch.where(conditions_met, 1 / weight_plus, 1 / weight_minus)

            # Exclude the case where no observational samples are in area of interest -> weight with zero
            #if not torch.isfinite(weights).all():
            #    print("test")
            weights = torch.where(torch.isinf(weights), torch.zeros(weights.size()), weights)
            weights = torch.where(weights == 1 / u_sampled.size(0), torch.zeros(weights.size()), weights).detach()

        log_prob = dist_y_shifted.log_prob(y_sampled)
        obj = log_prob * y_signs * weights
        return torch.squeeze(torch.mean(torch.sum(obj, dim=0)))

    def probability_test(self, y_samples_shifted):
        conditions_met = (y_samples_shifted >= self.boundary_lower) & (y_samples_shifted <= self.boundary_upper)
        y_idx = torch.where(conditions_met.all(dim=2), torch.ones((y_samples_shifted.size(0), y_samples_shifted.size(1))),
                            torch.zeros((y_samples_shifted.size(0), y_samples_shifted.size(1)))).detach()
        return torch.squeeze(torch.mean(y_idx, dim=0))



