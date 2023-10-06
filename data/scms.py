from abc import ABC, abstractmethod
import numpy as np
from data.data_structures import CausalDataset
import torch
import pyro.distributions as dist
from scipy.stats import beta


class SCM(ABC):
    def __init__(self, config):
        self.config = config
        self.a_type = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    @abstractmethod
    def f_u(self, x):
        pass

    @abstractmethod
    def f_a(self, x, u):
        pass

    @abstractmethod
    def f_y(self, x, u, a):
        pass

    def generate_dataset(self, n_samples, int_x=None, int_a=None):
        x = np.random.uniform(-1, 1, (n_samples, self.config["d_x"]))
        if int_x is not None:
            x = np.full_like(x, int_x)
        u = self.f_u(x)
        a = self.f_a(x, u)
        if int_a is not None:
            a = np.full_like(a, int_a)
        y = self.f_y(x, u, a)
        return CausalDataset(x=x, a=a, y=y, a_type=self.a_type)

    # Get outcome distribution samples (interventional/ conditional)
    def sample_y(self, n_samples, int_x=None, int_a=None, cond_a=None, tol=0):
        if int_a is not None and cond_a is not None:
            raise ValueError("Cannot specify both int_a and cond_a")
        if cond_a is None:
            data = self.generate_dataset(n_samples, int_x, int_a)
            y = data.get_data_np("y")
        else:
            data = self.generate_dataset(n_samples * 6, int_x, int_a)
            idx = np.squeeze(np.abs(data.get_data_np("a") - cond_a) <= tol)
            y = data.get_data_np("y")[idx]
            n_cond = y.shape[0]
            if n_cond < n_samples:
                y = y
            else:
                y = y[:n_samples]
        return y

    # Computes the ground truth causal or conditional query for (multiple) given x
    # Works for mean and quantiles
    def get_true_query(self, x, n_samples, int_a=None, q=None):
        # create x of size (x.shape[0] * n_samples, 1) by repeating each entry n_sample times
        x = np.repeat(x, n_samples, axis=0)
        y_sampled = self.sample_y(n_samples=x.shape[0], int_x=x, int_a=int_a, cond_a=None)
        y_sampled = np.reshape(y_sampled, (int(x.shape[0] / n_samples), n_samples))
        if q is None:
            return np.mean(y_sampled, axis=1, keepdims=True)
        else:
            return np.quantile(y_sampled, q, axis=1, keepdims=True)


class SCM_binary(SCM):
    def __init__(self, config):
        super().__init__(config)
        self.a_type = "discrete"

    # Observed propensity score
    def propensity_obs(self, x, a=1):
        score = 3 * np.mean(x, axis=1, keepdims=True)
        p_a1_x = self.sigmoid(score)
        # Prohibit overlap violations
        p_a1_x = np.clip(p_a1_x, 0.05, 0.95)
        if a == 0:
            return 1 - p_a1_x
        else:
            return p_a1_x

    # Upper bounds GMSM
    def get_splus(self, x, a):
        return 1 / (((1 - (1 / self.config["gamma"])) * self.propensity_obs(x, a)) + (1 / self.config["gamma"]))

    # Upper bounds GMSM
    def get_sminus(self, x, a):
        return 1 / (((1 - self.config["gamma"]) * self.propensity_obs(x, a)) + self.config["gamma"])

    def propensity_full(self, x, u, a=1):
        p_a1_xu = u * self.get_splus(x, a) * self.propensity_obs(x, a) + (1 - u) * self.get_sminus(x,
                                                                                                   a) * self.propensity_obs(
            x, a)
        if a == 0:
            return 1 - p_a1_xu
        else:
            return p_a1_xu

    def get_pu_given_x(self, x, u):
        p_u1_x = ((self.config["gamma"] - 1) * self.propensity_obs(x, 1) + 1) / (self.config["gamma"] + 1)
        if u == 0:
            return 1 - p_u1_x
        else:
            return p_u1_x

    # Functional assignments for sampling
    def f_u(self, x):
        p_u1_x = self.get_pu_given_x(x, 1)
        return np.random.binomial(1, p_u1_x)

    def f_a(self, x, u):
        p_a1_xu = self.propensity_full(x, u, 1)
        return np.random.binomial(1, p_a1_xu)

    def f_y(self, x, u, a):
        # Conditional mean
        mu_1 = a * (np.sin(1 * x)) + (1 - a) * (np.sin(4 * x))
        mu_2 = a * (np.cos(4 * x)) + (1 - a) * (np.cos(1 * x))
        mu = np.concatenate((mu_1, mu_2), axis=1)
        # Sample noise from multivariate normals
        # Define the mean and covariance matrix
        mean = torch.tensor([0.0, 0.0])
        cov1 = torch.tensor([[1.0, 0.7], [0.7, 1.0]])
        cov2 = torch.tensor([[1.0, -0.7], [-0.7, 1.0]])
        noise1 = dist.MultivariateNormal(mean, cov1).sample((x.shape[0],))
        noise2 = dist.MultivariateNormal(mean, cov2).sample((x.shape[0],))

        # Sample from the distribution
        # noise1 = np.random.multivariate_normal([0, 0], cov1)
        # noise2 = np.random.multivariate_normal([0, 0], cov2)
        # Sample y
        y = mu + noise1.numpy() * u + noise2.numpy() * (1 - u)
        return y


# 1D Simulation setting under MSM, similar to Kallus et al. (2019)
class SCM_binary_1D(SCM):
    def __init__(self, config):
        super().__init__(config)
        self.a_type = "discrete"

    # Observed propensity score
    def propensity_obs(self, x, a=1):
        score = 3 * np.mean(x, axis=1, keepdims=True)
        p_a1_x = self.sigmoid(score)
        # Prohibit overlap violations
        #p_a1_x = np.clip(p_a1_x, 0.05, 0.95)
        p_a1_x = 0.25 + 0.5 * p_a1_x
        if a == 0:
            return 1 - p_a1_x
        else:
            return p_a1_x

    # Upper bounds GMSM
    def get_splus(self, x, a):
        return 1 / (((1 - (1 / self.config["gamma"])) * self.propensity_obs(x, a)) + (1 / self.config["gamma"]))

    # Upper bounds GMSM
    def get_sminus(self, x, a):
        return 1 / (((1 - self.config["gamma"]) * self.propensity_obs(x, a)) + self.config["gamma"])

    def propensity_full(self, x, u, a=1):
        p_a1_xu = u * self.get_splus(x, a) * self.propensity_obs(x, a) + (1 - u) * self.get_sminus(x,
                                                                                                   a) * self.propensity_obs(
            x, a)
        if a == 0:
            return 1 - p_a1_xu
        else:
            return p_a1_xu

    def get_pu_given_x(self, x, u):
        p_u1_x = ((self.config["gamma"] - 1) * self.propensity_obs(x, 1) + 1) / (self.config["gamma"] + 1)
        if u == 0:
            return 1 - p_u1_x
        else:
            return p_u1_x

    # Functional assignments for sampling
    def f_u(self, x):
        p_u1_x = self.get_pu_given_x(x, 1)
        return np.random.binomial(1, p_u1_x)

    def f_a(self, x, u):
        p_a1_xu = self.propensity_full(x, u, 1)
        return np.random.binomial(1, p_a1_xu)

    def f_y(self, x, u, a):
        # Conditional mean
        y_mean = (2 * a - 1) * x + (2 * a - 1) - 2 * np.sin(2 * (2 * a - 1) * x) - 2 * (2 * u - 1) * (1 + 0.5 * x)
        y = y_mean + np.random.normal(0, 1, size=y_mean.shape)
        # Standardize outcomes
        y_mean = np.mean(y)
        y_std = np.std(y)
        y = (y - y_mean) / y_std
        return y


    def get_f_sensitivity_value(self, x, f, a):
        # Compute f sensitivity values
        s_plus = self.get_splus(x, a=1)
        s_minus = self.get_sminus(x, a=1)
        p_u1_x = self.get_pu_given_x(x, u=1)
        gamma = self.config["gamma"]
        f1 = f(gamma) * s_plus * p_u1_x + f(1/gamma) * s_minus * (1 - p_u1_x)
        f2 = f(1/gamma) * s_plus * p_u1_x + f(gamma) * s_minus * (1 - p_u1_x)
        return np.maximum(f1, f2)

    def get_rosenbaum_sensitivity_value(self, x, a):
        pi_u0 = self.propensity_full(x, u=0, a=1)
        pi_u1 = self.propensity_full(x, u=1, a=1)
        odds_ratio = (pi_u0 / (1 - pi_u0)) / (pi_u1 / (1 - pi_u1))
        return np.maximum(odds_ratio, 1/ odds_ratio)

    def get_msm_sensitivity_value(self, x, a):
        pi_ratio = self.propensity_obs(x) / (1 - self.propensity_obs(x))
        pi_u0 = self.propensity_full(x, u=0, a=1)
        pi_u1 = self.propensity_full(x, u=1, a=1)
        odds_ratio_0 = pi_ratio * (1 - pi_u0) / pi_u0
        odds_ratio_1 = pi_ratio * (1 - pi_u1) / pi_u1
        odds_ratio_max = np.maximum(odds_ratio_0, odds_ratio_1)
        odds_ratio_min = np.minimum(odds_ratio_0, odds_ratio_1)
        return np.maximum(odds_ratio_max, 1 / odds_ratio_min)




# 1D Simulation setting under CMSM, similar to Jesson et al. (2022)
class SCM_continuous_1D(SCM):
    def __init__(self, config):
        super().__init__(config)
        self.a_type = "continuous"

    def propensity_obs(self, a, x):
        u0 = np.zeros((x.shape[0], 1))
        u1 = np.ones((x.shape[0], 1))
        p0 = self.propensity_full(a, x, u0)
        p1 = self.propensity_full(a, x, u1)
        return (p0 + p1) / 2

    def get_par_beta(self, x, u):
        par1 = np.mean(x, axis=1, keepdims=True) + 2 + self.config["coef_u"] * (u - 0.5)
        par2 = np.mean(x, axis=1, keepdims=True) + 2 + self.config["coef_u"] * (u - 0.5)
        par1 = np.clip(par1, 0.1, 10)
        par2 = np.clip(par2, 0.1, 10)
        return par1, par2

    def propensity_full(self, a, x, u):
        # Parameter of beta distribution (par1=a, par2=b)
        par1, par2 = self.get_par_beta(x, u)
        prob = beta.pdf(a, par1, par2)
        return prob

    def get_density_ratios(self, a, x):
        # Create propensity ratios both for u=0 and u=1
        prop_u0 = self.propensity_full(a, x, u=np.zeros((x.shape[0], 1)))
        prop_u1 = self.propensity_full(a, x, u=np.ones((x.shape[0], 1)))
        prop_obs = self.propensity_obs(a, x)
        # Ratios are vectors of same length as x/ a
        ratio_u0 = prop_u0 / prop_obs
        ratio_u1 = prop_u1 / prop_obs
        return ratio_u0, ratio_u1

    def get_oracle_gamma(self, a, x):
        # Ratios are vectors of same length as x/ a
        ratio_u0, ratio_u1 = self.get_density_ratios(a, x)
        # Maximum/ minimum ratios
        max_ratio = np.maximum(ratio_u0, ratio_u1)
        min_ratio = np.minimum(ratio_u0, ratio_u1)
        # Worst case gammas for maximum/ minimum ratios
        max_gamma = max_ratio
        min_gamma = 1 / min_ratio
        # Worst case gammas
        gamma = np.maximum(max_gamma, min_gamma)
        return gamma

    def f_u(self, x):
        return np.random.binomial(1, 0.5, size=x.shape)

    def f_a(self, x, u):
        par1, par2 = self.get_par_beta(x, u)
        return beta.rvs(par1, par2)

    def f_y(self, x, u, a):
        # Conditional mean
        y_mean = a + x * np.exp(-x * a) - 0.5 * (u - 0.5) * x + (0.5 * x + 1)
        y = y_mean + np.random.normal(0, 1, size=y_mean.shape)
        # Standardize outcomes
        y_mean = np.mean(y)
        y_std = np.std(y)
        y = (y - y_mean) / y_std
        return y


    def get_f_sensitivity_value(self, x, f, a):
        density_ratio_0 = self.propensity_full(x=x, u=0, a=a) / self.propensity_obs(x=x, a=a)
        density_ratio_1 = self.propensity_full(x=x, u=1, a=a) / self.propensity_obs(x=x, a=a)
        f1 = f(density_ratio_0) * 0.5 * density_ratio_0 + f(density_ratio_1) * 0.5 * density_ratio_1
        f2 = f(1/density_ratio_0) * 0.5 * density_ratio_0 + f(1/density_ratio_1)* 0.5 * density_ratio_1
        return np.maximum(f1, f2)

    def get_rosenbaum_sensitivity_value(self, x, a):
        pi_u0 = self.propensity_full(x=x, u=0, a=a)
        pi_u1 = self.propensity_full(x=x, u=1, a=a)
        density_ratio = pi_u0 / pi_u1
        return np.maximum(density_ratio, 1/ density_ratio)

    def get_msm_sensitivity_value(self, x, a):
        density_ratio_0 = self.propensity_full(x=x, u=0, a=a) / self.propensity_obs(x=x, a=a)
        density_ratio_1 = self.propensity_full(x=x, u=1, a=a) / self.propensity_obs(x=x, a=a)
        density_ratio_max = np.maximum(density_ratio_0, density_ratio_1)
        density_ratio_min = np.minimum(density_ratio_0, density_ratio_1)
        return np.maximum(density_ratio_max, 1 / density_ratio_min)
