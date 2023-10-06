import torch
from torch.utils.data import Dataset
import numpy as np


# Customized dataset for multiple outcome causal inference setting
class CausalDataset(Dataset):
    # covariates x, treatment a, outcomes y
    # x is a numpy array of shape (n, d_x), a is a numpy array of shape (n, d_a), a is a numpy array of shape (n, d_y)
    # x_type is a list of strings of len d_x, a_type is a string, values "continuous" or "discrete"
    
    def __init__(self, x, a, y, propensity=None, gammas=None, x_type=None, a_type="discrete"):
        self.scaling_params = {}
        self.datatypes = {"x_type": x_type, "a_type": a_type}
        #Convert to pytorch tensors
        self.data = {}
        self.data["x"] = torch.from_numpy(x.astype(np.float32))
        if a is not None:
            self.data["a"] = torch.from_numpy(a.astype(np.float32))
        if y is not None:
            self.data["y"] = torch.from_numpy(y.astype(np.float32))
        else:
            self.data["y"] = None
        if propensity is not None:
            self.data["propensity"] = torch.from_numpy(propensity.astype(np.float32))
        else:
            self.data["propensity"] = None
        if gammas is not None:
            self.oracle_gammas = gammas.keys()
            for sensitivity_model in gammas.keys():
                self.data[sensitivity_model] = torch.from_numpy(gammas[sensitivity_model].astype(np.float32))
        else:
            self.oracle_gammas = None
        self.data["idx"] = torch.arange(0, self.data["x"].size(0)).view(-1, 1)

    def __len__(self):
        return self.data["x"].size(0)

    def get_data_np(self, key):
        return self.data[key].detach().numpy()
    
    def get_type_by_key(self, key):
        if key == "x":
            return self.datatypes["x_type"]
        if key == "a":
            return self.datatypes["a_type"]

    def get_dims(self, keys):
        dims = {}
        for key in keys:
            if key == "x":
                dims["x"] = self.data["x"].size(1)
            if key == "a":
                dims["a"] = self.data["a"].size(1)
            if key == "y":
                dims["y"] = self.data["y"].size(1)
        return dims

    def __getitem__(self, index) -> dict:
        x = {"x" : self.data["x"][index]}
        a = {"a" : self.data["a"][index]}
        idx = {"idx" : self.data["idx"][index]}
        if self.data["y"] is not None:
            y = {"y" : self.data["y"][index]}
            if self.data["propensity"] is not None:
                propensity = {"propensity": self.data["propensity"][index]}
                if self.oracle_gammas is not None:
                    gammas = {}
                    for sensitivity_model in self.oracle_gammas:
                        gammas[sensitivity_model] = self.data[sensitivity_model][index]
                    return {**x, **a, **y, **propensity, **gammas, **idx}
                else:
                    return {**x, **a, **y, **propensity, **idx}
            else:
                if self.oracle_gammas is not None:
                    gammas = {}
                    for sensitivity_model in self.oracle_gammas:
                        gammas[sensitivity_model] = self.data[sensitivity_model][index]
                    return {**x, **a, **y, **idx, **gammas}
                else:
                    return {**x, **a, **y, **idx}
        else:
            if self.data["propensity"] is not None:
                propensity = {"propensity": self.data["propensity"][index]}
                if self.oracle_gammas is not None:
                    gammas = {}
                    for sensitivity_model in self.oracle_gammas:
                        gammas[sensitivity_model] = self.data[sensitivity_model][index]
                    return {**x, **a, **propensity, **gammas, **idx}
                else:
                    return {**x, **a, **propensity, **idx}
            else:
                if self.oracle_gammas is not None:
                    gammas = {}
                    for sensitivity_model in self.oracle_gammas:
                        gammas[sensitivity_model] = self.data[sensitivity_model][index]
                    return {**x, **a, **gammas, **idx}
                else:
                    return {**x, **a, **idx}

    def add_propensity_scores(self, propensity_model):
        self.data["propensity"] = propensity_model.predict(self.data["x"]).detach()
        if self.get_dims(["a"])["a"] == 1:
            self.data["propensity"] = self.data["a"] * self.data["propensity"] + \
                                      (1 - self.data["a"]) * (1 - self.data["propensity"])
        elif self.get_dims(["a"])["a"] > 1:
            raise NotImplementedError("Propensity scores for multi-dimensional treatments not implemented yet.")

    # Standardizing ------------------
    def transform(self, key="y", type="normal"):
        means = []
        stds = []
        if type == "normal":
            for i in range(self.data[key].size(1)):
                mean = torch.mean(self.data[key][:, i])
                std = torch.std(self.data[key][:, i])
                self.data[key][:, i] = self.__scale_vector(self.data[key][:, i], mean, std)
                means.append(mean)
                stds.append(std)
            self.scaling_params[key]={"means": means, "sds": stds, "type": "normal"}

    def retransform(self, key="y", type="normal"):
        d = self.data[key].size(1)
        if type == "normal":
            for i in range(d):
                mean = self.scaling_params[key]["means"][i]
                std = self.scaling_params[key]["sds"][i]
                self.data[key][:, i] = self.__unscale_vector(self.data[key][:, i], mean, std)
                self.scaling_params["y"]["mean"] = 0
                self.scaling_params["y"]["sd"] = 1
            self.scaling_params[key]={"type": "none"}

    @staticmethod
    def __scale_vector(data, m, sd):
        return (data - m) / sd

    @staticmethod
    def __unscale_vector(data, m, sd):
        return (data * sd) + m



#Dataset used to train pytorch saved_models
#Contains (concatinated) input and output tensors
class Train_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, index) -> dict:
        return {"x": self.x[index], "y": self.y[index]}

    def get_sizes(self):
        return {"n": self.y.size(0), "d_in": self.x.size(1), "d_out": self.y.size(1)}
