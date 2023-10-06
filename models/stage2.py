import pytorch_lightning as pl
import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import ConditionalAutoRegressiveNN, AutoRegressiveNN
from abc import ABC, abstractmethod
import utils.utils as utils
from typing import List

from torch.distributions import Distribution, Normal, OneHotCategorical, MixtureSameFamily, Independent, MultivariateNormal, TransformedDistribution, Categorical


class Stage2(pl.LightningModule, ABC):
    def __init__(self, config, trained_models, sensitivity_model, causal_query):
        super().__init__()
        self.config = config
        self.trained_models = trained_models
        self.sensitivity_model = sensitivity_model
        self.causal_query = causal_query
        # Variables for training
        self.n_train = None
        self.n_val = None
        self.lambdas = None
        self.mu = None
        # Latent base distribution, multivariate normal
        self.dist_base = dist.MultivariateNormal(torch.zeros(self.config["dim_y"]),
                                                 torch.eye(self.config["dim_y"], self.config["dim_y"]))
        if self.config["bound_type"] == "upper":
            self.query_sign = -1
        elif self.config["bound_type"] == "lower":
            self.query_sign = 1
        else:
            raise ValueError("Invalid bound type")
        self.automatic_optimization = False
        # Logging
        self.neptune = config["neptune"]
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    # Returns the causal query (from latent samples)
    @abstractmethod
    def get_query_train(self, train_batch, u_sampled, xi_sampled):
        pass

    # Returns the estimated sensitivity violation for samples from U
    @abstractmethod
    def get_sensitivity_violation(self, train_batch, u_sampled, xi_sampled):
        pass
        # return self.sensitivity_model.get_sensitivity_violation(p_u_x, p_u_xa, propensity)

    def get_context(self, batch):
        x = batch["x"]
        a = batch["a"]
        return torch.cat([x, a], dim=1)

    def training_step(self, train_batch, batch_idx):
        n_batch = train_batch["x"].size(0)
        # Select lambdas by batch indices from train_batch["idx"]
        idx_lambdas = train_batch["idx"][:, 0]
        if self.current_epoch % self.config["nested_epochs"] != 0 or self.current_epoch == 0:
            # Sample from latent distribution, of size (n_train, n_batch, dim_y)
            u_sampled = self.dist_base.sample(torch.Size([self.n_train, n_batch]))
            # If treatment is discrete, sample bernoulli from propensity scores (account for data constraints in left supremum)
            if self.config["a_type"] == "discrete":
                p_a_x = train_batch["propensity"].transpose(0, 1).expand(self.n_train, n_batch)
                xi_sampled = torch.bernoulli(p_a_x).unsqueeze(dim=2).expand(self.n_train, n_batch, u_sampled.size(2))
            else:
                xi_sampled = torch.zeros((self.n_train, n_batch, u_sampled.size(2)))
            # Get sensitivity violations
            sensitivity = self.get_sensitivity_violation(train_batch, u_sampled, xi_sampled)
            # Get query
            query = self.get_query_train(train_batch, u_sampled, xi_sampled)
            # Update model parameters
            lagrangian = self.query_sign * query + torch.sum(- sensitivity * self.lambdas[idx_lambdas, :]
                                                             + (self.mu * sensitivity ** 2 / 2))
            # Optimization step
            self.optimizer.zero_grad()
            self.manual_backward(lagrangian)
            self.optimizer.step()
            # Logging
            if self.neptune:
                logging_dict = {"sensitivity": sensitivity.mean().item(), "query": query.mean().item(),
                                "mu": self.mu, "lambda": self.lambdas.mean().item()}
                self.log_dict(logging_dict, logger=True, on_epoch=True, on_step=False)
        else:
            with torch.no_grad():
                # Sample from latent distribution
                u_sampled = self.dist_base.sample(torch.Size([self.n_train, n_batch]))
                # If treatment is discrete, sample bernoulli from propensity scores (account for data constraints in left supremum)
                if self.config["a_type"] == "discrete":
                    p_a_x = train_batch["propensity"].transpose(0, 1).expand(self.n_train, n_batch)
                    xi_sampled = torch.bernoulli(p_a_x).unsqueeze(dim=2).expand(self.n_train, n_batch,
                                                                                u_sampled.size(2))
                else:
                    xi_sampled = torch.zeros((self.n_train, n_batch, u_sampled.size(2)))
                # Get sensitivity violations
                sensitivity = self.get_sensitivity_violation(train_batch, u_sampled, xi_sampled)
                # Update lambdas and mu
                self.lambdas[idx_lambdas, :] = torch.maximum(torch.zeros_like(self.lambdas[idx_lambdas, :]),
                                                             self.lambdas[idx_lambdas, :] - sensitivity * self.mu)
                if self.trainer.num_training_batches == batch_idx + 1:
                    self.mu = self.config["alpha"] * self.mu

    # Stage 2 model training
    # Input: causal dataset -> data points for which model should learn the causal query. if only 1 row is contained,
    # an (unconditional) NF is trained
    # n_train -> number of training points sampled from latent distribution
    # n_val -> number of validation points sampled from latent distribution
    def fit(self, dataset, name=""):
        # Initialize lagranga multipliers
        self.lambdas = torch.full((dataset.data["x"].size(0), 1), self.config["lambda_init"], dtype=torch.float32)
        self.mu = self.config["mu_init"]
        self.n_train = self.config["n_samples_train"]
        self.n_val = 0
        # Add propensity scores to dataset
        if self.trained_models["propensity"] is not None:
            dataset.add_propensity_scores(self.trained_models["propensity"])
        else:
            if "propensity" not in dataset.data.keys():
                dataset.data["propensity"] = torch.full_like(dataset.data["a"], 0)
        if self.n_val > 0:
            d_val = dataset
        else:
            d_val = None
        return utils.fit_model(self, dataset, d_val, name=name)

    def test_query(self, data_test, n_samples):
        self.eval()
        n_batch = data_test.data["x"].size(0)
        context = self.get_context(data_test.data)
        # Get query
        if self.config["a_type"] == "discrete":
            y_samples_shifted = self.dist_shifted_y.condition(context).sample(torch.Size([n_samples]))
        else:
            y_samples_shifted = self.dist_shifted_y.condition(context).sample(torch.Size([n_samples, n_batch]))
        query = self.causal_query.compute_query_test(y_samples_shifted)
        if data_test.data["x"].size(0) == 1:
            query = torch.unsqueeze(query, dim=0)
        return query.detach().unsqueeze(dim=1)

    def test_sensitivity_violation(self, data_test, n_samples):
        self.eval()
        n_batch = data_test.data["x"].size(0)
        # Sample from latent distribution
        u_sampled = self.dist_base.sample(torch.Size([n_samples, n_batch]))
        if self.trained_models["propensity"] is not None:
            data_test.add_propensity_scores(self.trained_models["propensity"])
        sensitivity_violation = self.get_sensitivity_violation(data_test.data, u_sampled)
        return sensitivity_violation

    def test_likelihood(self, data_test, y, scaling=None):
        self.eval()
        context = self.get_context(data_test.data)
        dist_y_shifted_cond = self.dist_shifted_y.condition(context)
        if scaling is not None:
            y = (y - scaling["mean"]) / scaling["std"]
        pred = dist_y_shifted_cond.log_prob(y).exp()
        if scaling is not None:
            pred = pred / torch.prod(scaling["std"])
        return pred

    def test_samples(self, data_test, n_samples):
        self.eval()
        n_batch = data_test.data["x"].size(0)
        context = self.get_context(data_test.data)
        dist_y_shifted_cond = self.dist_shifted_y.condition(context)
        samples = dist_y_shifted_cond.sample(torch.Size([n_samples, n_batch]))
        return samples

    def test_density_ratio_y(self, data_test, y):
        self.eval()
        context = self.get_context(data_test.data)
        shifted_density = self.dist_shifted_y.condition(context).log_prob(y).exp()
        base_density_dist = dist.ConditionalTransformedDistribution(self.dist_base,
                                                                    [self.trained_models["stage1"].transform])
        base_density = base_density_dist.condition(context).log_prob(y).exp()
        return shifted_density / base_density


# Stage 2 model for a fixed datapoint using an unconditional NF
class NF_stage2(Stage2):
    def __init__(self, config, trained_models, sensitivity_model, causal_query):
        super().__init__(config, trained_models, sensitivity_model, causal_query)
        # Architecture
        dim_y = config["dim_y"]
        dim_hidden = config["dim_hidden"]
        count_bins = config["count_bins"]
        # Invertible transformation
        hypernet = AutoRegressiveNN(dim_y, hidden_dims=[dim_hidden, dim_hidden],
                                    param_dims=[count_bins, count_bins, count_bins - 1, count_bins])
        self.transform = T.SplineAutoregressive(dim_y, hypernet, count_bins=count_bins)
        # Shifted distribution on latent induced by the transformation
        self.dist_shifted = dist.TransformedDistribution(self.dist_base, [self.transform])
        # Induced distribution on Y by concatinating Stage 1 and 2 transformations
        self.dist_shifted_y = dist.ConditionalTransformedDistribution(self.dist_shifted,
                                                              [self.trained_models["stage1"].transform])
        # Optimizer
        self.optimizer = torch.optim.Adam(self.transform.parameters(), lr=config["lr"])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.dist_shifted.clear_cache()

    # Returns the causal query (from latent samples)
    def get_query_train(self, train_batch, u_sampled, xi_sampled):
        context = self.get_context(train_batch)
        transform_stage2 = self.transform
        transform_stage1 = self.trained_models["stage1"].transform.condition(context)
        return self.causal_query.compute_query_train(u_sampled, xi_sampled, self.dist_shifted_y.condition(context), transform_stage1,
                                                     transform_stage2)

    # Returns the estimated sensitivity violation for samples from U
    def get_sensitivity_violation(self, train_batch, u_sampled, xi_sampled):
        return self.sensitivity_model.sensitivity_violation(u_sampled, xi_sampled, self.dist_base, self.dist_shifted, self.transform, train_batch)

    def test_density_ratio(self, data_test, y):
        self.eval()
        shifted_density = self.dist_shifted.log_prob(y).exp()
        base_density = self.dist_base.log_prob(y).exp()
        return shifted_density / base_density


#Helper classes to define conditional mixture distributions
class StackedDistribution(Distribution):
    def __init__(self, components: List[Distribution], batch_shape, condition_shape):
        super().__init__(batch_shape=(batch_shape, 2), event_shape=components[0].event_shape)
        self.components = components
        self.condition_shape = condition_shape

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_probs = [comp.log_prob(value.squeeze(-2)) for comp in self.components]
        return torch.stack(log_probs, dim=-1)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        sample_shape = sample_shape + (self.condition_shape, )

        sample_0 = self.components[0].sample(sample_shape)
        sample_1 = self.components[1].sample(sample_shape)
        return torch.stack([sample_0, sample_1], dim=-2)

# Class for conditional mixture distribution between base and shifted distribution, weights are outputs of propensity net
class ConditionalMixtureSameFamily(dist.ConditionalDistribution):
    def __init__(self, dist_base, dist_shifted, propensity_net):
        self.dist_base = dist_base
        self.dist_shifted = dist_shifted
        self.propensity_net = propensity_net

    def condition(self, context):
        context_x = context[:, :-1]
        context_a = context[:, -1:]
        #Predict propensities
        propensities = self.propensity_net.predict(context_x).detach()
        propensities_a = context_a * propensities + (1 - context_a) * (1 - propensities)
        cat_dist = Categorical(probs=torch.concat([propensities_a, 1 - propensities_a], dim=1))
        comps = StackedDistribution([self.dist_base, self.dist_shifted.condition(context)], batch_shape=context.shape[0], condition_shape=context.shape[0])
        return MixtureSameFamily(cat_dist, Independent(comps, 0))
    def clear_cache(self):
        pass


# Stage 2 model for a multiple datapoints using an conditional NF
class CNF_stage2(Stage2):
    def __init__(self, config, trained_models, sensitivity_model, causal_query):
        super().__init__(config, trained_models, sensitivity_model, causal_query)
        # Architecture
        dim_y = config["dim_y"]
        dim_hidden = config["dim_hidden"]
        dim_context = config["dim_context"]
        count_bins = config["count_bins"]
        spline_bound = config["spline_bound"]
        # Invertible transformation
        hypernet = ConditionalAutoRegressiveNN(dim_y, dim_context, hidden_dims=[dim_hidden, dim_hidden],
                                               param_dims=[count_bins, count_bins, count_bins - 1, count_bins])
        self.transform = T.ConditionalSplineAutoregressive(dim_y, hypernet, count_bins=count_bins, bound=spline_bound)
        # Shifted distribution on latent induced by the transformation
        self.dist_shifted_obj = dist.ConditionalTransformedDistribution(self.dist_base, [self.transform])

        if self.config["a_type"] == "discrete":
            self.dist_shifted = ConditionalMixtureSameFamily(self.dist_base, self.dist_shifted_obj, self.trained_models["propensity"])
        else:
            self.dist_shifted = self.dist_shifted_obj

        # Induced shifted distribution on Y by passing the shifted latent distribution into the stage 1 transformation
        self.dist_shifted_y = dist.ConditionalTransformedDistribution(self.dist_shifted,
                                                              [self.trained_models["stage1"].transform])
        # Optimizer
        self.optimizer = torch.optim.Adam(self.transform.parameters(), lr=config["lr"])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.dist_shifted.clear_cache()

    # Returns the causal query
    def get_query_train(self, train_batch, u_sampled, xi_sampled):
        context = self.get_context(train_batch)
        transform_stage2 = self.transform.condition(context)
        transform_stage1 = self.trained_models["stage1"].transform.condition(context)
        return self.causal_query.compute_query_train(u_sampled, xi_sampled, self.dist_shifted_y.condition(context), transform_stage1,
                                                     transform_stage2)

    # Returns the estimated sensitivity violation for samples from U
    def get_sensitivity_violation(self, train_batch, u_sampled, xi_sampled):
        context = self.get_context(train_batch)
        return self.sensitivity_model.sensitivity_violation(u_sampled, xi_sampled, self.dist_base, self.dist_shifted.condition(context),
                                                            self.transform.condition(context), train_batch)

    def test_density_ratio(self, data_test, y):
        self.eval()
        context = self.get_context(data_test.data)
        shifted_density = self.dist_shifted.condition(context).log_prob(y).exp()
        base_density = self.dist_base.log_prob(y).exp()
        return shifted_density / base_density
