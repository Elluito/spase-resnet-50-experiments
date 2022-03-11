import glob
import re
import time
import json
import os
from pathlib import Path
from itertools import cycle
import logging
# GGplot IMPORTS

# Seaborsn IMPORTS
import seaborn as sn
# Matplotlib IMPORTS
import matplotlib.pyplot as plt
# Scypy IMPORTS
from scipy.interpolate import make_interp_spline, BSpline
# Pytorch IMPORTS
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import torch.nn as nn
from torch.nn import Flatten
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer, LightningModule, Callback, LightningDataModule
from ignite.metrics import Accuracy
import torch
from torch.nn import Parameter
# Hessian eigenthings imports
from hessian_eigenthings import compute_hessian_eigenthings
# Optimizers imports
from KFAC_Pytorch.optimizers import KFACOptimizer
from sam import SAM
# RigL repository imports
from rigl_repro_utils.models.resnet import resnet50, ResNet
from rigl_repro_utils.models.wide_resnet import WideResNet
from rigl_repro_utils.sparselearning.core import Masking
from rigl_repro_utils.loss import LabelSmoothingCrossEntropy
from copy import deepcopy

"============================DEFINITION OF " \
"CLASSES----------------*********************************#----================================="


def loss_one_epoch(model, data_loader, loss,batch_size):
    assert loss.reduction is "sum", "The reduction of the loss object must be \"sum\" "
    # loss = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    model.cuda()
    cumulative_sum = 0
    total_items = 0

    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            label = label.cuda()
            prediction = model(data)
            value = loss(prediction, label)
            cumulative_sum += value.item()
            total_items += len(data)
            break
    return cumulative_sum / total_items


class DenseResnet50(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet50(num_classes)
        # init a pretrained resnet

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self, epochs):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        # In theory this  scheduler will decrease the learning rate 3 times during training
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3)
        return [optimizer], [lr_scheduler]


class DenseWideRes(LightningModule):
    def __init__(self, num_classes, loss_object):
        super().__init__()
        self.model = WideResNet(22, 2, num_classes=num_classes)
        self.loss = loss_object
        # init a pretrained resnet

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self, epochs):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        # In theory this  scheduler will decrease the learning rate 3 times during training
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3)
        return [optimizer], [lr_scheduler]


class LitModel(pl.LightningModule):
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


"==================================================================================================================="
"============================== DEFINITION OF FUNCTIONS=============================================================="
"Disables batch normalization update for the model"


def disable_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm):
            module.eval()


"Enables batch normalization update for the model"


def enable_bn(model):
    model.train()


def load_CIFAR10(datapath, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=datapath, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=datapath, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


def get_masked_model_from_checkpoint(path_to_checkpoint: "Path", model: nn.Module) -> 'nn.Module':
    # if isinstance(model, WideResNet):
    #     mask = Masking()
    #     mask.add_module(model, path_to_checkpoint)
    #
    assert path_to_checkpoint.is_file(), f"No .pth file at {path_to_checkpoint}"
    full_state_dict = torch.load(path_to_checkpoint, map_location="cpu")
    state_dict = full_state_dict["model"]
    assert "mask" in full_state_dict, f"No mask found at {path_to_checkpoint}"
    setattr(model, "masks", full_state_dict["mask"]["masks"])
    non_zero = 0
    total_params = 0
    for name, weight in model.masks.items():
        # Skip modules we arent masking from the model path
        if name not in state_dict.keys():
            continue
        non_zero += model.masks[name].sum().int().item()
        total_params += weight.numel()

    logging.info(
        f"Loaded mask from {path_to_checkpoint} with density: {non_zero / total_params}"
    )


" This is done in order to avoid batch notmalization parameters"


def get_vector_of_subset_weights(full_state_dict: dict, subset_state_dict: dict):
    dictionary_to_convert = {}
    for key in subset_state_dict.keys():
        dictionary_to_convert[key] = full_state_dict[key].detach().clone()
    vector = parameters_to_vector(dictionary_to_convert.values())
    return vector


def get_weights_of_vector(full_state_dict: dict, vector: torch.Tensor, subset_state_dict: dict):
    # This is because the "vector" only see the subset_state_dict
    # I need to fill the small_dict with the vector incoming and
    # vector_to_parameters replaces the values inside the iterable he gets.
    # also small_dict has the variable names and the shapes I need for the analysis
    pre_tensors = list(subset_state_dict.values())

    vector_to_parameters(vector, pre_tensors)
    # TODO: In theory, when you replace the data of the tensor inside the state dictionary it is a pointer and as such
    # it points to the original state dictionary, I do not know if I have to call model.load_state_dict() with
    # full_stat_dict. I need to check this
    for index, key in enumerate(subset_state_dict.keys()):
        full_state_dict[key].data = pre_tensors[index].data


def get_linear_path_between_sparse_points(dataset, model, batch_size, path_to_solution, number_of_points=10):
    "We are doing a linear interpolation between a pruned initialization and a pruning solution"
    trainloader, testLoader = None, None
    step = 1 / number_of_points
    solution = None
    num_classes = None
    if dataset == "cifar10":
        trainloader, testloader = load_CIFAR10("../data", batch_size)
        num_classes = 10
    if isinstance(model, WideResNet):
        # get the last point which is the solution
        solution = WideResNet(depth=22, widen_factor=2, small_dense_density=1.0)
        # The initial declaration of the class do not have "masks" on it so we add them
        full_dict = torch.load(path_to_solution)
        solution.load_state_dict(full_dict["model"])
        setattr(solution, "masks", full_dict["mask"]["masks"])
        # get the mask for the initial point
        get_masked_model_from_checkpoint(Path(path_to_solution), model)
    if isinstance(model, ResNet):
        solution = resnet50(num_classes=num_classes)
        # The initial declaration of the class do not have "masks" on it so we add them
        full_dict = torch.load(path_to_solution)
        solution.load_state_dict(full_dict["model"])
        setattr(solution, "masks", full_dict["mask"]["masks"])
        # get the mask for the initial point
        get_masked_model_from_checkpoint(path_to_solution, model)
    loss_object = nn.CrossEntropyLoss(reduction="sum")

    print(f"Loss of final point:{loss_one_epoch(solution,trainloader,loss_object,batch_size=batch_size)}")
    # TODO: See what happens if I use named_parameters() instead of state_dict()
    initial_point = get_vector_of_subset_weights(model.state_dict(), model.masks)
    final_point = get_vector_of_subset_weights(solution.state_dict(), solution.masks)
    get_weights_of_vector(solution.state_dict(),final_point,solution.masks)
    print(f"Loss of final point after the reshape"
          f":{loss_one_epoch(solution,trainloader,loss_object,batch_size=batch_size)}")
    final_point = final_point.cuda()
    initial_point = initial_point.cuda()
    models_in_line = []
    line = torch.linspace(0, 1, number_of_points)
    line.cuda()
    loss_ = []
    for weight in line:
        current_model = deepcopy(model)
        # Everything must be in cuda or CPU. Right now is in cuda
        vector = torch.lerp(initial_point, final_point, weight.cuda())

        current_state_dict = current_model.state_dict()
        get_weights_of_vector(current_state_dict, vector, model.masks)
        models_in_line.append(current_model)
        # TODO: Verify that the variable current_state_dict indeed modifies the current_model weigths
        point_of_loss = loss_one_epoch(current_model, trainloader,loss_object,batch_size=batch_size)
        print(f"Loss: {point_of_loss}, alpha:{weight}")
        loss_.append(point_of_loss)

    return models_in_line, line, loss_


def KFAC_sparse_training_step():
    ''' In this I should ENFORCE THE MASK AFTER THE TRAINING STEP HAS BEEN DONE. The mask should be an attribute of the
    network model
    '''

    pass


def SAM_sparse_training_step():
    ''' In this I should ENFORCE THE MASK AFTER THE TRAINING STEP HAS BEEN DONE. The mask should be an attribute of the
    network model.
    Because I'm using ResNet I should be careful with batch normalization acording to the SAM github
   @hjq133: The suggested usage can potentially cause problems if you use batch normalization.
   The running statistics are computed in both forward passes, but they should be computed only for
   the first one. A possible solution is to set BN momentum to zero (kindly suggested by @ahmdtaha) to
   bypass the running statistics during the second pass. An example usage is on lines 51 and 58 in example/train.py:
        for batch in dataset.train:
          inputs, targets = (b.to(device) for b in batch)

          # first forward-backward step
          enable_running_stats(model)  # <- this is the important line
          predictions = model(inputs)
          loss = smooth_crossentropy(predictions, targets)
          loss.mean().backward()
          optimizer.first_step(zero_grad=True)

          # second forward-backward step
          disable_running_stats(model)  # <- this is the important line
          smooth_crossentropy(model(inputs), targets).mean().backward()
          optimizer.second_step(zero_grad=True)
    '''
    pass


def sparse_training_step():
    ''' In this I should ENFORCE THE MASK AFTER THE TRAINING STEP HAS BEEN DONE. The mask should be an attribute of the
    network model
    '''
    pass


def dense_training_step():
    pass


"=========================================== EXECUTES THE MAIN========================================================="


def main():
    # model = ()
    # trainer = Trainer(gpus=hparams.gpus)
    # trainer.fit(model)
    checkpointpath = 'C:/Users/Luis Alfredo/OneDrive - University of Leeds/PhD/Code/Local Experiments/spase-resnet-50-experiments/checkpoints/pruning/+specific=cifar10_wrn_22_2_pruning,masking.final_density=0.05,seed=2/ckpts/best_model.pth'

    train, test = load_CIFAR10("../data", 64)

    model = WideResNet(depth=22, widen_factor=2, small_dense_density=1.0)
    models_line, line, loss = get_linear_path_between_sparse_points("cifar10", model, batch_size=64,
                                                                    path_to_solution=checkpointpath,
                                                                    number_of_points=3)
    plt.plot(line, loss)
    plt.ylabel('Train CrossEntropy', fontsize=20)
    plt.xlabel('$\\alpha$', fontsize=20)
    plt.show()


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--gpus", default=None)
    # args = parser.parse_args()
    main()
