import pytorch_lightning as pl
import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import ConditionalAutoRegressiveNN, AutoRegressiveNN
from abc import ABC, abstractmethod
import utils.utils as utils


# Conditional normalizing flow base class
class CNF(pl.LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim_context = config["dim_context"]
        dim_y = config["dim_y"]
        dim_hidden = config["dim_hidden"]
        count_bins = config["count_bins"]
        hypernet = ConditionalAutoRegressiveNN(dim_y, dim_context, hidden_dims=[dim_hidden, dim_hidden],
                                               param_dims=[count_bins, count_bins, count_bins - 1, count_bins])
        self.transform = T.ConditionalSplineAutoregressive(dim_y, hypernet, count_bins=count_bins)
        # Standard normal distribution (on latent space) used for training
        dist_base_training = dist.MultivariateNormal(torch.zeros(dim_y), torch.eye(dim_y, dim_y))
        # Corresponding output distribution induced by the transformation
        self.dist_y_given_x_training = dist.ConditionalTransformedDistribution(dist_base_training, [self.transform])
        # Logging
        self.neptune = config["neptune"]
        # Optimizer
        self.optimizer = torch.optim.Adam(self.transform.parameters(), lr=config["lr"])
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.dist_y_given_x_training.clear_cache()

    @abstractmethod
    def training_objective(self, batch):
        pass

    @abstractmethod
    def training_step(self, train_batch, batch_idx):
        pass

    def get_context(self, batch):
        x = batch["x"]
        a = batch["a"]
        return torch.cat([x, a], dim=1)


# Conditional normalizing flow stage 1
class CNF_stage1(CNF):
    def __init__(self, config):
        super().__init__(config)

    def training_objective(self, batch):
        x = batch["x"]
        a = batch["a"]
        xa = torch.cat([x, a], dim=1)
        y = batch["y"]
        # Forward pass
        log_p_y_given_xa = self.dist_y_given_x_training.condition(xa).log_prob(y)
        return {"obj": - log_p_y_given_xa.mean()}

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Loss
        obj_dict = self.training_objective(train_batch)
        # Logging
        if self.neptune:
            obj_dict_train = dict([("train_" + key, value) for key, value in obj_dict.items()])
            self.log_dict(obj_dict_train, logger=True, on_epoch=True, on_step=False)
        return obj_dict["obj"]

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        # Loss
        obj_val = self.training_objective(val_batch)
        # Logging
        obj_dict_val = dict([("val_" + key, value) for key, value in obj_val.items()])
        self.log_dict(obj_dict_val, logger=True, on_epoch=True, on_step=False)
        return obj_val["obj"]

    # Evaluates density of y given x on a grid of y values
    # x is of shape (batch_size, d_in), y is of shape (n_grid, d_y)
    def test_likelihood(self, data_test, y, scaling=None):
        self.eval()
        context = self.get_context(data_test.data)
        context_full = context.repeat_interleave(y.shape[0], dim=0)
        y_full = y.repeat(context.shape[0], 1)
        if scaling is not None:
            y_full = (y_full - scaling["mean"]) / scaling["std"]
        pred = self.dist_y_given_x_training.condition(context_full).log_prob(y_full).exp()
        if scaling is not None:
            pred = pred / torch.prod(scaling["std"])
        return pred

    def sample(self, data_test, n_samples, scaling_params=None):
        self.eval()
        context = self.get_context(data_test.data)
        # context is of shape (batch_size, d_context)
        samples = torch.squeeze(self.dist_y_given_x_training.condition(context).sample(torch.Size([n_samples, context.shape[0]])))
        if samples.dim() == 1:
            samples = samples.unsqueeze(1)
        samples = torch.transpose(samples, 0, 1)
        if scaling_params is not None:
            samples = samples * scaling_params["sd"] + scaling_params["mean"]
        return samples

    def fit(self, d_train, d_val=None, name=""):
        return utils.fit_model(self, d_train, d_val, name=name)

