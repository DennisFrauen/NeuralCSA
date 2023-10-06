import pytorch_lightning as pl
import torch
import torch.nn as nn
import utils.utils as utils
from torch.utils.data import DataLoader


# MLP for multi-class classification (discrete conditional distributions) or continuous regression
class PropensityNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_size = config["dim_input"]
        hidden_size = config["dim_hidden"]
        self.output_size = config["dim_output"]
        self.output_type = config["out_type"]
        if self.output_type == "discrete":
            if self.output_size == 1:
                self.loss = torch.nn.BCELoss(reduction='mean')
            else:
                self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        elif self.output_type == "continuous":
            self.loss = torch.nn.MSELoss(reduction='mean')
        else:
            raise ValueError("out_type must be either 'discrete' or 'continuous'")
        dropout = config["dropout"]

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.output_size),
            # nn.Softmax()
        )
        self.neptune = config["neptune"]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def objective(self, batch, out):
        loss = self.loss(out, batch["a"])
        return {"obj": loss}

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Forward pass
        out = self.network(train_batch["x"])
        if self.output_size == 1 and self.output_type == "discrete":
            out = out.sigmoid()
        # Loss
        obj_dict = self.objective(train_batch, out)
        # Logging
        if self.neptune:
            obj_dict_train = dict([("train_" + key, value) for key, value in obj_dict.items()])
            self.log_dict(obj_dict_train, logger=True, on_epoch=True, on_step=False)
        return obj_dict["obj"]

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        # Forward pass
        out = self.network(val_batch["x"])
        if self.output_size == 1 and self.output_type == "discrete":
            out = out.sigmoid()
        # Loss
        obj_val = self.objective(val_batch, out)
        # Logging
        obj_dict_val = dict([("val_" + key, value) for key, value in obj_val.items()])
        self.log_dict(obj_dict_val, logger=True, on_epoch=True, on_step=False)
        return obj_val["obj"]

    def predict(self, x, scaling_params=None):
        self.eval()
        if self.output_type == "discrete":
            if self.output_size > 1:
                out = self.network(x).softmax(dim=-1)
            else:
                out = self.network(x).sigmoid()
        else:
            out = self.network(x)
            if scaling_params is not None:
                out = out * scaling_params["sd"] + scaling_params["mean"]
        return out.detach()

    def fit(self, d_train, d_val=None, name=""):
        return utils.fit_model(self, d_train, d_val, name=name)
