# Neural causal sensitivity analysis

This repository contains the code for our paper "A Neural Framework for Generalized Causal Sensitivity Analysis".

![Plot Intuition](media/architecture.png)


#### Project structure 
- *data* contains the data generation files/ real-world data preprocessing
- *experiments* contains the code to run the experiments and the results, trained models and hyperparameter
- *hyperparam* contains the tuning ranges and code for hyperparameter tuning
- *models* contains code for the sensitivity models, bound computation and estimation, and neural networks


#### Requirements
The project is build with python 3.10 and uses the packages listed in the file `requirements.txt`. 

#### Reproducing the experiments
The scripts running the experiments are contained in the `/experiments` folder. Each experimental setting is contained within its own subfolder. These are:
- `/exp_real`: real-world data
- `/exp_sim_binary`: binary treatment, no mediators
- `/exp_sim_binary_m1`: binary treatment, one mediator
- `/exp_sim_binary_2`: binary treatment, two mediators
- `/exp_sim_continuous`: continuous treatment, no mediators
- `/exp_sim_continuous_m1`: continuous treatment, one mediator
- `/exp_sim_continuous_m2`: continuous treatment, two mediators
- `/exp_sim_continuous_weight`: continuous treatment, no mediators, weighed GMSM experiment (Table 1).

The files *run.py* and *run_quatile.py* execute the experiments. Configuration settings are in *config.yaml*. By default, the pretrained models in the `/saved_models` will be used to estimate bounds. For model training, the parameter *train_models* in *config.yaml* can be changed to *True*.


#### Reproducing hyperparameter tuning
The code for hyperparameter tuning is contained in the `/hyperparams` folder. As for the `/experiments` folder, each experimental setting is contained within its own subfolder. The optimal hyperparameters are stored as `.yaml` files in the `/params` subfolder. Hyperparameter tuning may be executed via *run.py*.
