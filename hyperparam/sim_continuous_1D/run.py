import utils.utils as utils
from hyperparam.main import run_hyper_tuning

if __name__ == "__main__":
    config_hyper = utils.load_yaml("/hyperparam/sim_continuous_1D/config")
    config_data = utils.load_yaml("/experiments/sim_continuous_1D/config")["data"]
    run_hyper_tuning(config_hyper, config_data)