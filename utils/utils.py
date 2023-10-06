import os
import yaml
import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as fctnl
from torch.utils.data import DataLoader
import random
import numpy as np

def get_device_string():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_project_path():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(path.parent.absolute())


def load_yaml(path_relative):
    return yaml.safe_load(open(get_project_path() + path_relative + ".yaml", 'r'))

def load_all_yaml_from_directory(path_relative):
    path = get_project_path() + path_relative
    files = os.listdir(path)
    configs = []
    for file in files:
        if file.endswith(".yaml"):
            configs.append(load_yaml(path_relative + file[:-5]))
    return configs


def save_yaml(path_relative, file):
    with open(get_project_path() + path_relative + ".yaml", 'w') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)


def save_pytorch_model(path_relative, model):
    torch.save(model.state_dict(), get_project_path() + path_relative + ".pt")


def load_pytorch_model(path_relative, model):
    path = get_project_path() + path_relative + ".pt"
    # Check if file exists
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        return model
    else:
        print("Model file not found, returning no model")
        return None


def get_logger(use_logger=True, name=""):
    if use_logger:
        logger = WandbLogger(project='neural-sensitivity', name=name)
    else:
        logger = True
    return logger


def get_config_names(model_configs):
    config_names = []
    for model_config in model_configs:
        config_names.append(model_config["name"])
    return config_names


def mse_bce(y, y_hat, y_type="continuous"):
    if y_type == "continuous":
        return torch.mean((y - y_hat) ** 2)
    if y_type == "binary":
        return fctnl.binary_cross_entropy(y_hat, y, reduction='mean')


def fit_model(model, d_train, d_val=None, name=""):
    epochs = model.config["epochs"]
    batch_size = model.config["batch_size"]
    logger = get_logger(model.config["neptune"], name=name)
    trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False,
                         accelerator="auto", logger=logger, enable_checkpointing=False)

    train_loader = DataLoader(dataset=d_train, batch_size=batch_size, shuffle=True)
    if d_val is not None:
        val_loader = DataLoader(dataset=d_val, batch_size=batch_size, shuffle=False)
        trainer.fit(model, train_loader, val_loader)
        try:
            trainer.fit(model, train_loader, val_loader)
            val_results = trainer.validate(model=model, dataloaders=val_loader, verbose=False)
        except:
            print("Model training + validation failed, returning large validation error")
            val_results = [{"val_obj": 1000000}]
    else:
        trainer.fit(model, train_loader)
        val_results = None
    if model.config["neptune"]:
        logger.experiment.finish()
    return val_results


def get_f_by_name(sensitivity_model_name):
    if sensitivity_model_name in ["kl_binary", "kl_continuous"]:
        def f_kl(x):
            return x * np.log(x)
        return f_kl

    elif sensitivity_model_name in ["tv_binary", "tv_continuous"]:
        def f_tv(x):
            return np.abs(x - 1) * 0.5
        return f_tv
    elif sensitivity_model_name in ["hellinger_binary", "hellinger_continuous"]:
        def f_hellinger(x):
            return (np.sqrt(x) - 1) ** 2
        return f_hellinger
    elif sensitivity_model_name in ["chisquared_binary", "chisquared_continuous"]:
        def f_chisquared(x):
            return (x - 1) ** 2
        return f_chisquared
    else:
        raise(NotImplementedError("Sensitivity model not implemneted"))