import os
from typing import List
from typing import Optional
import hydra
import omegaconf
import optuna
import torch.utils.data as data_utils
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
# optuna imports
from optuna.trial import TrialState
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

SEED = 42
# Torch imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.utils.data as data_utilis
from torchvision import datasets
from torchvision import transforms
from multiprocesing_test import CIFAR10ModelSAM, CIFAR10Model
from main import get_simple_masking
from rigl_repo_utils.models import registry as model_registry
from multiprocesing_test.main import single
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 3
DIR = os.getcwd()


def objective_SAM(trial: optuna.trial.Trial) -> float:
    # We are optimizing rho, learning rate, if using adaptive or not is better
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1)
    rho = trial.suggest_float("rho", 0.0005, 2.5)
    momentum = trial.suggest_float("momentum", 0.1, 0.99)
    adaptative = trial.suggest_categorical("adaptative",[False,True])
    nesterov = trial.suggest_categorical("nesterov",[False,True])
    type_of_model, arguments = model_registry["wrn-22-2"]
    # Create an instace of said model
    dummy_model = type_of_model(*arguments)
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(),lr=0.1)
    mask = get_simple_masking(dummy_optimizer, density=0.05)

    real_model = type_of_model(*arguments)
    model = CIFAR10ModelSAM(model=real_model, learning_rate=learning_rate,adaptative=adaptative, rho=rho,momentum=momentum)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        checkpoint_callback=False,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")]
    )
    hyperparameters = dict(lr=learning_rate, rho=rho)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model)
    return trainer.callback_metrics["val_acc"].item()


def main():
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
    )

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective_SAM, n_trials=500)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig1 = plot_optimization_history(study)
    fig2 = plot_intermediate_values(study)
    fig3 = plot_param_importances(study)
    fig4 = plot_slice(study,params=["learning_rate","rho"])

    fig1.savefig("images/opt_history.png")
    fig2.savefig("images/intermediate_values.png")
    fig3.savefig("images/para_importances.png")
    fig4.savefig("images/slice.png")


if __name__ == '__main__':
    main()
