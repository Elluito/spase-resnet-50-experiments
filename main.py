import glob
import platform
import multiprocessing as mp
import typing
from collections.abc import Iterator, Generator
import re
import time
import json
import os
import threading as thread
from itertools import cycle
import logging
import numpy as np
from copy import deepcopy
import hydra
import pathlib
import GPUtil
import omegaconf
from omegaconf import OmegaConf
from waiting import wait
# GGplot IMPORTS
# Seaborsn IMPORTS
# Matplotlib IMPORTS
import matplotlib.pyplot as plt
# Scypy IMPORTS
import sparselearning.core
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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ignite.metrics import Accuracy
import torch

# Hessian eigenthings imports
from hessian_eigenthings import compute_hessian_eigenthings
# Optimizers imports
# from KFAC_Pytorch.optimizers import KFACOptimizer
from sam import SAM
# RigL repository imports
from rigl_repo_utils.models.resnet import resnet50, ResNet
from rigl_repo_utils.models.wide_resnet import WideResNet
from rigl_repo_utils.sparselearning.core import Masking

from rigl_repo_utils.data import get_dataloaders
from rigl_repo_utils.loss import LabelSmoothingCrossEntropy
from rigl_repo_utils.models import registry as model_registry
from rigl_repo_utils.sparselearning.counting.ops import get_inference_FLOPs
from rigl_repo_utils.sparselearning.funcs.decay import registry as decay_registry
from rigl_repo_utils.sparselearning.utils.accuracy_helper import get_topk_accuracy
from rigl_repo_utils.sparselearning.utils.smoothen_value import SmoothenValue
from rigl_repo_utils.sparselearning.utils.train_helper import (
    get_optimizer,
    load_weights,
    save_weights,
)
from rigl_repo_utils.main import train, single_seed_run
# Imports from sparselearning (is rigl_repo_utils but installed as a package)
from sparselearning.utils import layer_wise_density

"============================DEFINITION OF " \
"CLASSES----------------*********************************#----================================="


class GPUMonitor(thread.Thread):
    def __init__(self, delay):
        super(GPUMonitor, self).__init__()
        self.stopped = False
        self.gpu_avail = threading.Event()
        self.delay = delay  # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUs = GPUtil.getFirstAvailable(maxLoad=0.66, maxMemory=0.66)
            if len(GPUs) != 0:
                self.gpu_avail.set()
            else:
                self.gpu_avail.clear()
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


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


# ==================================================================================================================="
# ============================== DEFINITION OF FUNCTIONS=============================================================="

def RigL_train_FLOPs(
        sparse_FLOPs: int, dense_FLOPs: int, mask_interval: int = 100
) -> float:
    """
    Train FLOPs for Rigging the Lottery (RigL), Evci et al. 2020.

    :param sparse_FLOPs: FLOPs consumed for sparse model's forward pass
    :type sparse_FLOPs: int
    :param dense_FLOPs: FLOPs consumed for dense model's forward pass
    :type dense_FLOPs: int
    :param mask_interval: Mask update interval
    :type mask_interval: int
    :return: RigL train FLOPs
    :rtype: float
    """
    return (2 * sparse_FLOPs + dense_FLOPs + 3 * sparse_FLOPs * mask_interval) / (
            mask_interval + 1
    )


# Creates the random seed for the dataloaders, I want them to use the same seed for debbuging porpuses
def get_simple_masking(optimizer: torch.optim.Optimizer, density: float = 0.1):
    """
    This function returns a simplified version of Masking given the optimizer.The masking itself will not
    optimize the module is simply used for applying the masking to the gradients and to the model itself.
    :param optimizer: Optimizer for the masking
    :return: Masking with defaut values and given optimizer
    """
    kwargs = {
    #    "final_sparsity": 1 - 0.95,
        "t_max": 75000,
        "t_start":0 ,
        "interval": 100,
    }

    m = Masking(optimizer=optimizer,
                prune_rate_decay=decay_registry["cosine"](),
                density=density,
                sparse_init="erdos-renyi"
                )

    return m


def is_something_ready(something):
    if something.ready():
        return True
    return False


###
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_arguments(cfg: omegaconf.DictConfig):
    raise NotImplemented("Get_train__arguments is not implemented")


def get_a_model(cfg: omegaconf.DictConfig) -> nn.Module:
    # Select model
    assert (
            cfg.model in model_registry.keys()
    ), f"Select from {','.join(model_registry.keys())}"
    model_class, model_args = model_registry[cfg.model]
    _small_density = cfg.masking.density if cfg.masking.name == "Small_Dense" else 1.0
    model = model_class(*model_args, _small_density).to(device)
    return model


def get_mask(cfg: omegaconf.DictConfig) -> Masking:
    mask = None
    if not cfg.masking.dense:
        max_iter = (
            cfg.masking.end_when
            if cfg.masking.apply_when == "step_end"
            else cfg.masking.end_when * len(train_loader)
        )

        kwargs = {"prune_rate": cfg.masking.prune_rate, "t_max": max_iter}

        if cfg.masking.decay_schedule == "magnitude-prune":
            kwargs = {
                "final_sparsity": 1 - cfg.masking.final_density,
                "t_max": max_iter,
                "t_start": cfg.masking.start_when,
                "interval": cfg.masking.interval,
            }

        decay = decay_registry[cfg.masking.decay_schedule](**kwargs)

        optimizer, (lr_scheduler, warmup_scheduler) = get_optimizer(model, **cfg.optimizer)
        mask = masking(
            optimizer,
            decay,
            density=cfg.masking.density,
            dense_gradients=cfg.masking.dense_gradients,
            sparse_init=cfg.masking.sparse_init,
            prune_mode=cfg.masking.prune_mode,
            growth_mode=cfg.masking.growth_mode,
            redistribution_mode=cfg.masking.redistribution_mode,
        )

        # support for lottery mask
        lottery_mask_path = path(cfg.masking.get("lottery_mask_path", ""))
        mask.add_module(model, lottery_mask_path)
        return mask, lr_scheduler, warmup_scheduler


def experiment_with_population(cfg: omegaconf.DictConfig) -> typing.List[nn.Module]:
    N_population = cfg.population_experiment.population_size
    number_of_generations = cfg.population_experiment.generations
    monitor = GPUMonitor(delay=1)

    #####################################################################################################################

    # Training Step
    train_step = 0
    # Set device
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(**cfg.dataset)

    # Select model
    assert (
            cfg.model in model_registry.keys()
    ), f"Select from {','.join(model_registry.keys())}"
    model_class, model_args = model_registry[cfg.model]
    _small_density = cfg.masking.density if cfg.masking.name == "Small_Dense" else 1.0

    # Training multiplier
    cfg.optimizer.decay_frequency *= cfg.optimizer.training_multiplier
    cfg.optimizer.decay_frequency = int(cfg.optimizer.decay_frequency)

    cfg.optimizer.epochs *= cfg.optimizer.training_multiplier
    cfg.optimizer.epochs = int(cfg.optimizer.epochs)

    if cfg.masking.get("end_when", None):
        cfg.masking.end_when *= cfg.optimizer.training_multiplier
        cfg.masking.end_when = int(cfg.masking.end_when)
    # THings that need to be duplicated for every process##########################################
    # Setup optimizers, lr schedulers
    optimizer, (lr_scheduler, warmup_scheduler) = get_optimizer(model, **cfg.optimizer)

    model = model_class(*model_args, _small_density).to(device)
    # Setup mask
    mask = None
    if not cfg.masking.dense:
        max_iter = (
            cfg.masking.end_when
            if cfg.masking.apply_when == "step_end"
            else cfg.masking.end_when * len(train_loader)
        )

        kwargs = {"prune_rate": cfg.masking.prune_rate, "t_max": max_iter}

        if cfg.masking.decay_schedule == "magnitude-prune":
            kwargs = {
                "final_sparsity": 1 - cfg.masking.final_density,
                "t_max": max_iter,
                "t_start": cfg.masking.start_when,
                "interval": cfg.masking.interval,
            }

        decay = decay_registry[cfg.masking.decay_schedule](**kwargs)

        mask = masking(
            optimizer,
            decay,
            density=cfg.masking.density,
            dense_gradients=cfg.masking.dense_gradients,
            sparse_init=cfg.masking.sparse_init,
            prune_mode=cfg.masking.prune_mode,
            growth_mode=cfg.masking.growth_mode,
            redistribution_mode=cfg.masking.redistribution_mode,
        )

        # support for lottery mask
        lottery_mask_path = path(cfg.masking.get("lottery_mask_path", ""))
        mask.add_module(model, lottery_mask_path)

    # Load from checkpoint
    # model, optimizer, mask, step, start_epoch, best_val_loss = load_weights(
    #     model, optimizer, mask, ckpt_dir=cfg.ckpt_dir, resume=cfg.resume
    # )

    # Train model
    epoch = 0
    warmup_steps = cfg.optimizer.get("warmup_steps", 0)
    warmup_epochs = warmup_steps / len(train_loader)
    # TODO: put here the training loop with all the parameters it needs, I need to initialize the parameters

    # Start the training procedure

    _masking_args = {}
    ######################################################################################################################

    pool = [Process(target=single_generation_step, args=())
            for range_ in ranges]
    for g in range(number_of_generations):
        for p in pool:
            wait(lambda: monitor.gpu_avail.is_set(), waiting_for="GPUs to be availables")
            p.start()

    pass


'''
Disables batch normalization update for the model
'''


def single_generation_training_step(model: nn.Module,
                                    mask: sparselearning.core.Masking,
                                    cfg: omegaconf.DictConfig,
                                    optimizer: torch.optim.Optimizer,
                                    lr_scheduler=None,
                                    warmup_scheduler=None,
                                    device: torch.device = None,
                                    train_loader: torch.utils.data.DataLoader = None,
                                    val_loader: torch.utils.data.DataLoader = None,
                                    test_loader: torch.utils.data.DataLoader = None,
                                    found_best_event: mp.Event = None,
                                    pop_index: int = None

                                    ) -> typing.Tuple[float, float]:
    # FIXME: This  method is not complete Needs to be completed
    epoch = 0
    warmup_steps = cfg.optimizer.get("warmup_steps", 0)
    warmup_epochs = warmup_steps / len(train_loader)
    best_val_loss = float("Inf")
    step = 0

    # Start the training procedure

    _masking_args = {}
    if mask:
        _masking_args = {
            "masking_apply_when": cfg.masking.apply_when,
            "masking_interval": cfg.masking.interval,
            "masking_end_when": cfg.masking.end_when,
            "masking_print_FLOPs": cfg.masking.get("print_FLOPs", False),
        }
    for epoch in range(cfg.optimizer.epochs):
        # step here is training iters not global steps
        scheduler = lr_scheduler if (epoch >= warmup_epochs) else warmup_scheduler

        _, step = train(
            model,
            mask,
            train_loader,
            optimizer,
            scheduler,
            step,
            epoch + 1,
            device,
            label_smoothing=cfg.optimizer.label_smoothing,
            log_interval=cfg.log_interval,
            use_wandb=cfg.wandb.use,
            **_masking_args,
        )
        # Run validation
        if epoch % cfg.val_interval == 0:
            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                step,
                epoch + 1,
                device,
                use_wandb=cfg.wandb.use,
            )

            # Save weights
            if (epoch + 1 == cfg.optimizer.epochs) or ((epoch + 1) % cfg.ckpt_interval == 0):
                if val_loss < best_val_loss:
                    is_min = True
                    best_val_loss = val_loss
                else:
                    is_min = False

                save_weights(
                    model,
                    optimizer,
                    mask,
                    val_loss,
                    step,
                    epoch + 1,
                    ckpt_dir=cfg.ckpt_dir,

                    is_min=is_min,
                )

        # Apply mask
        if (
                mask
                and cfg.masking.apply_when == "epoch_end"
                and epoch < cfg.masking.end_when
        ):
            if epoch % cfg.masking.interval == 0:
                mask.update_connections()
    if not epoch:
        # Run val anyway
        epoch = cfg.optimizer.epochs - 1
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            step,
            epoch + 1,
            device,
            use_wandb=cfg.wandb.use,
        )

    val_loss, val_accuracy = evaluate(
        model,
        test_loader,
        step,
        epoch + 1,
        device,
        is_test_set=True,
        use_wandb=cfg.wandb.use,
    )

    if cfg.wandb.use:
        # Close wandb context
        wandb.join()

    training_flops = 0
    sparse_FLOPS = get_inference_FLOPs(mask, input_tensor=torch.rand(*(1, 3, 32, 32)))
    dense_FLOPS = mask.dense_FLOPs
    if cfg.masking.name is "RigL":
        training_flops = RigL_train_FLOPs(sparse_FLOPS * step * cfg.dataset.batch_size,
                                          dense_FLOPS * step * cfg.dataset.batch_size, cfg.masking.interval)
    if cfg.masking.name is "Static":
        training_flops = sparse_FLOPS * 2 * step * cfg.dataset.batch_size

    return val_accuracy, training_flops


def loss_one_epoch(model, data_loader, loss, train=True):
    assert loss.reduction is "sum", "The reduction of the loss object must be \"sum\" "
    # loss = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    model.cuda()
    cumulative_sum = 0
    total_items = 0

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


def get_masked_model_from_checkpoint(path_to_checkpoint: pathlib.Path, model: nn.Module) -> 'nn.Module':
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


def set_biases_to_zero(model: nn.Module):
    for name, weight in model.named_parameters():
        if "bias" in name and not "bn" in name:
            weight.data.copy_(torch.zeros_like(weight))


def compare_biases(model1: nn.Module, model2: nn.Module):
    parameters_2 = list(model2.named_parameters())

    for i, (name, weigth) in enumerate(model1.named_parameters()):
        if "bias" in name and not "bn" in name:
            name2, weigth2 = parameters_2[i]
            temp = 0


def upload_only_batch_normalization_and_bias(named_parameters, state_dict, upload_bias=False):
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
        get_masked_model_from_checkpoint(pathlib.Path(path_to_solution), initial_model)
        # Register the buffers which are related to the running mean and variance of the batch normalization layers
        vector_to_parameters(parameters_to_vector(solution.buffers()), initial_model.buffers())
    if isinstance(initial_model, ResNet):
        solution = resnet50(num_classes=num_classes)
        # The initial declaration of the class do not have "masks" on it so we add them
        full_dict = torch.load(path_to_solution)
        solution.load_state_dict(full_dict["model"])
        setattr(solution, "masks", full_dict["mask"]["masks"])
        # get the mask for the initial point
        get_masked_model_from_checkpoint(path_to_solution, initial_model)
    # set_biases_to_zero(solution)
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
    # get_weights_of_vector(temp_model.named_parameters(), final_point, initial_model.masks)
    vector_to_parameters(final_point, temp_model.parameters())
    # vector_to_parameters(buffers,temp_model.buffers())
    # thing2 = parameters_to_vector(temp_model.parameters())
    temp_model_with_final_weigths__loss = compute_loss_batch(batch=(test_input, test_output), model=temp_model,
                                                             loss_object=loss_object, train=False)
    assert temp_model_with_final_weigths__loss == solution_loss_value, f"Loss of the temp model after ´giving him the " \
                                                                       f"solution's weights´ is not the same as the solution " \
                                                                       f"reshape. Ltemp =" \
                                                                       f" {temp_model_with_final_weigths__loss},Lsolution =" \
                                                                       f" {solution_loss_value}"
    assert compute_loss_batch(batch=(test_input, test_output), model=initial_model, loss_object=loss_object,
                              train=False) != temp_model_with_final_weigths__loss, "Loss of the temp model is equal to the initial model when " \
                                                                                   "it should´nt be"

    # I delete all of these since I don't need it from this point onwards
    del temp_model_with_final_weigths__loss, temp_model, initial_point_loss_value, solution_loss_value

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
        vector_to_parameters(vector, current_model.parameters())

        # point_of_loss = compute_loss_batch(batch=(test_input,test_output),model=current_model,loss_object=loss_object, train=False)
        point_of_loss = loss_one_epoch(model=current_model, data_loader=trainloader, loss=loss_object, train=False)
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


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    # I do this to access the path to the dataset, which is a shared location
    single_seed_run(cfg)

    # model = ()
    # trainer = Trainer(gpus=hparams.gpus)
    # trainer.fit(model)
    # checkpointpath = 'C:/Users/Luis Alfredo/OneDrive - University of Leeds/PhD/Code/LocalExperiments/spase-resnet-50-experiments/checkpoints/pruning/+specific=cifar10_wrn_22_2_pruning,masking.final_density=0.05,seed=2/ckpts/best_model.pth'
    # train, test = load_CIFAR10("../data", 64)

    # models_line, line, loss = get_linear_path_between_sparse_points("cifar10", model, batch_size=64,
    #                                                                 path_to_solution=checkpointpath,
    #                                                                 number_of_points=100)
    # plt.plot(line, loss)
    # plt.ylabel('Train CrossEntropy', fontsize=20)
    # plt.xlabel('$\\alpha$', fontsize=20)
    # plt.show()



if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--gpus", default=None)
    # args = parser.parse_args()

    if "Windows" in platform.system():
        dataset_path = str(pathlib.Path(__file__).parent.parent.absolute())
    else:
        dataset_path = "/nobackup/sclaam"
    OmegaConf.register_new_resolver("data_path", lambda: hydra.utils.to_absolute_path(dataset_path))
    main()
