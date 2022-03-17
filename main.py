import glob
from collections.abc import Iterator, Generator
import re
import time
import json
import os
from pathlib import Path
from itertools import cycle
import logging
import numpy as np
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
'''
Creates the random seed for the dataloaders, I want them to use the same seed for debbuging porpuses
'''


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)
'''
Disables batch normalization update for the model
'''


def loss_one_epoch(model, data_loader, loss, train=True):
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
            value = compute_loss_batch(batch=(data, label), model=model, loss_object=loss, train=train)
            cumulative_sum += value.item()
            total_items += len(data)
    return cumulative_sum / total_items


def disable_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm):
            module.eval()


'''
Enables batch normalization update for the model, I migth need this for optimizing with SAM
'''


def enable_bn(model):
    model.train()


def load_CIFAR10(datapath, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=datapath, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

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


def get_vector_of_subset_weights(full_named_parameters: Generator, subset_state_dict: dict):
    dictionary_to_convert = []
    for name, param in full_named_parameters:
        if name in subset_state_dict.keys():
            # Detach and clone are used to avoid any issues with the original model.
            dictionary_to_convert.append(param.data.detach().clone())
    vector = parameters_to_vector(dictionary_to_convert)
    return vector


def get_weights_of_vector(full_named_parameters: Iterator, vector: torch.Tensor, subset_state_dict: dict):
    # This is because the "vector" only see the subset_state_dict
    # I need to fill the small_dict with the vector incoming and
    # vector_to_parameters replaces the values inside the iterable he gets.
    # also small_dict has the variable names and the shapes I need for the analysis
    pre_tensors = deepcopy(list(subset_state_dict.values()))

    vector_to_parameters(vector, pre_tensors)

    # The index is outside because pre_tensors do not have the same amount of elements that full_named_parameters
    index = 0
    for key, param in full_named_parameters:
        if key in subset_state_dict.keys():
            param.data.copy_(pre_tensors[index].clone())
            index += 1
    # return full_named_parameters, pre_tensors


def compute_loss_batch(batch, model, loss_object, train=True):
    value = 0
    if not train:
        model.eval()
        with torch.no_grad():
            data, label = batch
            prediction = model(data)
            value = loss_object(prediction, label)
    else:
        data, label = batch
        prediction = model(data)
        value = loss_object(prediction, label)
    return value


def set_biases_to_zero(model:nn.Module):
    for name,weight in model.named_parameters():
        if "bias" in name and not "bn"in name:
           weight.data.copy_(torch.zeros_like(weight))
def compare_biases(model1: nn.Module, model2: nn.Module):
    parameters_2 = list(model2.named_parameters())

    for i, (name, weigth) in enumerate(model1.named_parameters()):
        if "bias" in name and not "bn"in name:
            name2,weigth2 = parameters_2[i]
            temp = 0

def upload_only_batch_normalization_and_bias(named_parameters,state_dict,upload_bias = False):

    for name, weigth in named_parameters:
        if "bn" in name:
            named_parameters[name].data.copy_(state_dict[name].data)
        if "bias" in name and not "bn" in name and upload_bias:
            named_parameters[name].data.copy_(state_dict[name].data)

def get_linear_path_between_sparse_points(dataset, initial_model, batch_size, path_to_solution, number_of_points=10):
    "We are doing a linear interpolation between a pruned initialization and a pruning solution"
    trainloader, testloader = None, None
    step = 1 / number_of_points
    solution = None
    num_classes = None
    if dataset == "cifar10":
        trainloader, testloader = load_CIFAR10("../data", batch_size)
        num_classes = 10
    if isinstance(initial_model, WideResNet):
        # get the last point which is the solution
        solution = WideResNet(depth=22, widen_factor=2, small_dense_density=1.0)
        # The initial declaration of the class do not have "masks" on it so we add them
        full_dict = torch.load(path_to_solution)
        solution.load_state_dict(full_dict["model"])
        setattr(solution, "masks", full_dict["mask"]["masks"])
        # get the mask for the initial point
        get_masked_model_from_checkpoint(Path(path_to_solution), initial_model)
        # Register the buffers which are related to the running mean and variance of the batch normalization layers
        vector_to_parameters(parameters_to_vector(solution.buffers()),initial_model.buffers())
    if isinstance(initial_model, ResNet):
        solution = resnet50(num_classes=num_classes)
        # The initial declaration of the class do not have "masks" on it so we add them
        full_dict = torch.load(path_to_solution)
        solution.load_state_dict(full_dict["model"])
        setattr(solution, "masks", full_dict["mask"]["masks"])
        # get the mask for the initial point
        get_masked_model_from_checkpoint(path_to_solution, initial_model)
    #set_biases_to_zero(solution)
    loss_object = nn.CrossEntropyLoss(reduction="sum")
    iter_dataloader = iter(testloader)
    test_input, test_output = next(iter_dataloader)
    test_input = test_input.cuda()
    test_output = test_output.cuda()
    initial_model.cuda()
    solution.cuda()
    del iter_dataloader
    solution_loss_value = compute_loss_batch(batch=(test_input, test_output), model=solution,
                                             loss_object=loss_object, train=False)
    initial_point_loss_value = compute_loss_batch(batch=(test_input, test_output), model=initial_model,
                                                  loss_object=loss_object, train=False)
    initial_point = parameters_to_vector(initial_model.parameters())
    final_point = parameters_to_vector(solution.parameters())

    #
    #
    # # Here I test that the reshaping does not affect the output
    # # For the final point
    # get_weights_of_vector(solution.named_parameters(), final_point, solution.masks)
    # solution_reshape_loss_value = compute_loss_batch(batch=(test_input, test_output), model=solution,
    #                                                  loss_object=loss_object, train=False)
    # # For the initial point
    # get_weights_of_vector(initial_model.named_parameters(), initial_point, initial_model.masks)
    # initial_point_reshape_loss_value = compute_loss_batch(batch=(test_input, test_output), model=initial_model,
    #                                                       loss_object=loss_object, train=False)
    # # Both of these loss values must be unaffected by the reshape
    # assert solution_loss_value == solution_reshape_loss_value, f"Loss of the solution is not the same after the " \
    #                                                            f"reshape. Lpre =" \
    #                                                            f" {solution_loss_value},Lpost =" \
    #                                                            f" {solution_reshape_loss_value}"
    # assert initial_point_loss_value == initial_point_reshape_loss_value, f"Loss of the initial model is not the same " \
    #                                                                      f"after " \
    #                                                                      f"the reshape. Lpre = {initial_point_loss_value}," \
    #                                                                      f"Lpost = {initial_point_reshape_loss_value}"
    #



    # Now I want to know if I replace these values in one copy of the initial model, the value changes to the correct
    # value
    temp_model = deepcopy(initial_model)
    # buffers = parameters_to_vector(solution.buffers())
    #get_weights_of_vector(temp_model.named_parameters(), final_point, initial_model.masks)
    vector_to_parameters(final_point,temp_model.parameters())
    # vector_to_parameters(buffers,temp_model.buffers())
    #thing2 = parameters_to_vector(temp_model.parameters())
    temp_model_with_final_weigths__loss = compute_loss_batch(batch=(test_input, test_output), model=temp_model,
                                                             loss_object=loss_object, train=False)
    assert temp_model_with_final_weigths__loss == solution_loss_value, f"Loss of the temp model after ´giving him the " \
                                                                       f"solution's weights´ is not the same as the solution " \
                                                                       f"reshape. Ltemp =" \
                                                                       f" {temp_model_with_final_weigths__loss},Lsolution =" \
                                                                       f" {solution_loss_value}"
    assert compute_loss_batch(batch=(test_input,test_output),model=initial_model,loss_object=loss_object,
                              train=False)!= temp_model_with_final_weigths__loss , "Loss of the temp model is equal to the initial model when " \
                                                         "it should´nt be"
    
    # I delete all of these since I don't need it from this point onwards
    del temp_model_with_final_weigths__loss, temp_model,initial_point_loss_value,solution_loss_value

    final_point = final_point.cuda()
    initial_point = initial_point.cuda()
    models_in_line = []
    line = torch.linspace(0, 1, number_of_points)
    line.cuda()
    loss_ = []
    for weight in line:
        current_model = deepcopy(initial_model)

        # WARNING: Everything must be in cuda or CPU. Right now is in cuda

        vector = torch.lerp(initial_point, final_point, weight.to(initial_point.device))
        vector_to_parameters(vector,current_model.parameters())




        # point_of_loss = compute_loss_batch(batch=(test_input,test_output),model=current_model,loss_object=loss_object, train=False)
        point_of_loss = loss_one_epoch(model=current_model,data_loader=trainloader,loss=loss_object,train=False)
        print(f"Loss: {point_of_loss}, alpha:{weight}")
        # Put the model on CPU because I dont want to saturate the GPU memory.
        current_model.cpu()
        models_in_line.append(current_model)
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

    # models_line, line, loss = get_linear_path_between_sparse_points("cifar10", model, batch_size=64,
    #                                                                 path_to_solution=checkpointpath,
    #                                                                 number_of_points=100)
    # plt.plot(line, loss)
    # plt.ylabel('Train CrossEntropy', fontsize=20)
    # plt.xlabel('$\\alpha$', fontsize=20)
    # plt.show()
def experiment_with_population():
    


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--gpus", default=None)
    # args = parser.parse_args()
    main()
