import os
import multiprocessing as mp
import threading as thread
import logging
import time
import pathlib as path
import datetime as date
import omegaconf
import wandb
from waiting import wait
import hydra
from omegaconf import OmegaConf, DictConfig
# rig and torch imports
from rigl_repo_utils.models import registry as model_registry
from rigl_repo_utils.loss import LabelSmoothingCrossEntropy
from sparselearning.funcs.decay import registry as decay_registry
from sparselearning.funcs.decay import CosineDecay
from sparselearning.core import Masking
from sparselearning.counting.ops import get_inference_FLOPs
from sparselearning.utils import train_helper, smoothen_value

# torch imports
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
# other inputs
import typing
import GPUtil
from sam import SAM
from main import get_simple_masking
import platform
from pytorch_lightning.loggers import WandbLogger
from sparselearning.utils.accuracy_helper import get_topk_accuracy
import tqdm

PATH_DATASETS = ""
if "Linux" in platform.system():
    PATH_DATASETS = os.environ.get("PATH_DATASETS", "/nobackup/sclaam")
else:
    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())

BATCH_SIZE = 128 if AVAIL_GPUS else 64
# USABLE_CORES = os.cpu_count() // 3 if "Linux" in platform.system() else 2
PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()


################################################# CLASSSES #############################################################

class SAMLoop(pl.loops.FitLoop):
    def advance(self):
        """Advance from one iteration to the next."""

        loss = lightning_module.training_step(batch, i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def on_advance_end(self):
        """Do something at the end of an iteration."""

    def on_run_end(self):
        """Do something when the loop ends."""


class MNISTModel(pl.LightningModule):

    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)

    def on_train_batch_start(self, batch: typing.Any, batch_idx: int, unused: int = 0) -> typing.Optional[int]:

        # TODO: Verify that I do not need to train on this function
        global found_best_event
        if found_best_event.value:
            return -1
        else:
            return 1


class CIFAR10ModelSAM(pl.LightningModule):

    def __init__(self, data_dir=PATH_DATASETS, model: nn.Module = None, mask: Masking = None,
                 learning_rate: float = 2e-4, rho: float = 0.05, adaptative: bool = False, decay_frequency: int = 100,
                 decay_factor: float = 0.1, label_smoothing: int = 0, weight_decay: float = 5e-4,
                 momentum: float = 0.9):

        super().__init__()
        assert model is not None, "The model needs to be Specified"

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.rho = rho
        self.adaptive = adaptative
        self.weight_decay = weight_decay
        self.momentum = momentum
        # self.lr_scheduler = None
        # self.decay_frequency = decay_frequency
        # self.decay_factor = decay_factor

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (3, 32, 32)
        channels, width, height = self.dims

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        self.train_transform = transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        # Define PyTorch model
        # type_of_model, arguments = model_registry[type_model]
        # Create an instace of said model
        self.model = model
        self.first_time = 1
        self.mask = mask
        if mask:
            self.mask.add_module(self.model, lottery_mask_path=path.Path(""))
            self.mask.to_module_device_()
            self.mask.apply_mask()
            self.sparse_inference_flops = self.mask.inference_FLOPs
            self.dense_inference_flops = self.mask.dense_FLOPs
        self.loss_object = nn.CrossEntropyLoss()
        self.training_flops: float = 0

    def forward(self, x):
        y = self.model(x)
        return y

    def compute_loss(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_object(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        if self.first_time and self.mask:
            self.mask.to_module_device_()
            self.first_time = 0
        optimizer = self.optimizers()
        optimizer.zero_grad()
        # first forward-backward pass
        # enable_bn(self)
        # loss_1 = self.compute_loss(batch)
        # self.manual_backward(loss_1)
        loss_1 = self.loss_object(self(batch[0]), batch[1])
        loss_1.backward()
        optimizer.first_step(zero_grad=True)
        # second forward-backward pass
        # disable_bn(self)
        # loss_2 = self.compute_loss(batch)
        # self.manual_backward(loss_2)
        self.loss_object(self(batch[0]), batch[1]).backward()
        optimizer.second_step(zero_grad=True)
        if self.mask:
            self.mask.apply_mask()
            one_forward_backward_pass = self.sparse_inference_flops * 2
            self.training_flops += one_forward_backward_pass * 2
            self.log("epoch_flops", self.training_flops, on_step=false, on_epoch=true)

        # flops for one forward pass
        # assuming that the backward pass uses approximately the same number of flops as the
        # forward.
        # acording to  this website shorturl.at/hpap7 they said that is between 2 and 3 times the
        # number of flops

        self.log("train_loss", loss_1, prog_bar=True)

        return loss_1

    def validation_step(self, batch, batch_idx):

        x, y = batch
        with torch.no_grad():
            logits = self(x)
        loss = self.loss_object(logits, y)

        top_1_accuracy, top_5_accuracy = get_topk_accuracy(
            logits, y, topk=(1, 5)
        )
        # Calling self.log will surface up scalars for you in TensorBoard
        # self.log("val_loss", loss, prog_bar=True)
        self.log("top1_acc", top_1_accuracy, prog_bar=True)
        self.log("top5_acc", top_5_accuracy, prog_bar=True)
        return {"val_loss": loss, "top1_acc": top_1_accuracy, "top5_acc": top_5_accuracy}

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = {k: [dic[k] for dic in validation_step_outputs] for k in validation_step_outputs[0]}
        self.log("val_loss",
                 torch.tensor(all_preds["val_loss"], dtype=torch.float32).mean())
        self.log("val_accuracy",
                 torch.tensor(all_preds["top1_acc"], dtype=torch.float32).mean())
        self.log("val_top_5_accuracy",
                 torch.tensor(all_preds["top5_acc"], dtype=torch.float32).mean())
        # self.log("Epoch_FLOPS", self.training_FLOPS)
        if self.mask:
            log_dict = {
                "Inference FLOPs": self.mask.inference_FLOPs / self.mask.dense_FLOPs,
                "Avg Inference FLOPs": self.mask.avg_inference_FLOPs / self.mask.dense_FLOPs,
            }
            for key, value in log_dict.items():
                self.log(name=key, value=value)
        # if isinstance(self.logger, WandbLogger):
        #     temp_dict = {"layer_wise_density": layer_wise_density.wandb_bar(self.mask)}
        #     wandb.log(temp_dict)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        weight_decay = self.weight_decay
        if self.weight_decay:
            parameters = train_helper._add_weight_decay(self, weight_decay)
            weight_decay = 0
        else:
            parameters = self.parameters()
        optimizer = SAM(params=parameters, base_optimizer=torch.optim.SGD, lr=self.learning_rate,
                        rho=self.rho,
                        adaptive=self.adaptive, momentum=self.momentum, weight_decay=weight_decay)

        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True,
                                                      transform=self.train_transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)

    @property
    def automatic_optimization(self) -> bool:
        return False
    #
    # def on_train_batch_start(self, batch: typing.Any, batch_idx: int, unused: int = 0) -> typing.Optional[int]:
    #
    #     global found_best_event
    #     if found_best_event.value:
    #         return -1
    #     else:
    #         return 1


class CIFAR10Model(pl.LightningModule):

    def __init__(self, data_dir=PATH_DATASETS, type_model: str = "wrn-22-2", mask: Masking = None,
                 learning_rate: float = 2e-4, momentum: float = 0.99):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (3, 32, 32)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Define PyTorch model
        type_of_model, arguments = model_registry[type_model]
        # Create an instace of said model
        self.model = type_of_model(*arguments)
        self.mask = mask
        if mask:
            self.mask.add_module(self.model)
            self.mask.apply_mask()
        self.loss_object = F.nll_loss
        self.accuracy = Accuracy()
        self.sparse_inference_FLOPS = get_inference_FLOPs(self.mask, input_tensor=torch.rand(*(1, 3, 32, 32)))
        self.training_FLOPS: float = 0

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def compute_loss(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_object(logits, y)
        return loss

    def training_step(self, batch, batch_idx):

        # first forward-backward pass
        enable_bn(self.model)
        loss_1 = self.compute_loss(batch)
        self.mask.apply_mask_gradients()
        # Assuming that the backward pass uses approximately the same number of Flops as the
        # forward.
        # Acording to  this website shorturl.at/hpAP7 they said that is between 2 and 3 times the
        # number of FLOPS
        one_forward_backward_pass = self.sparse_inference_FLOPS

        self.training_FLOPS += one_forward_backward_pass
        self.log("FLOPS", self.training_FLOPS, prog_bar=True)

        return loss_1

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True,
                                                      transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=BATCH_SIZE)

    def on_train_batch_start(self, batch: typing.Any, batch_idx: int, unused: int = 0) -> typing.Optional[int]:

        # TODO: Verify that I do not need to train on this function
        global found_best_event
        if found_best_event.value:
            return -1
        else:
            return 1


class SimpleWrapper(pl.LightningModule):
    def __init__(self, model=None, loss: typing.Callable = None, optimizer: torch.optim.Optimizer = None,
                 scheduler=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert model is not None, "The model needs to be Specified"
        self.model = model
        self.loss_object = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x) -> torch.TensorType:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_object(pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits = self.model(x)
        loss = self.loss_object(logits, y)
        top_1_accuracy, top_5_accuracy = get_topk_accuracy(
            F.log_softmax(logits), y, topk=(1, 5)
        )
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", top_1_accuracy, prog_bar=True)
        self.log("top_5_accuracy", top_5_accuracy, prog_bar=True)

    def test_step(self, batch, batch_indx):
        self.validation_step(batch, batch_indx)

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class GPUMonitor(thread.Thread):
    def __init__(self, delay):
        super(GPUMonitor, self).__init__()
        self.stopped = False
        self.gpu_avail = thread.Event()
        self.delay = delay  # Time between calls to GPUtil

    def run(self):
        while not self.stopped:
            GPUs = GPUtil.getFirstAvailable(maxLoad=0.66, maxMemory=0.66)
            if len(GPUs) != 0:
                self.gpu_avail.set()
            else:
                self.gpu_avail.clear()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


########################################## Functions ###################################################################
def disable_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) \
                or isinstance(module, nn.BatchNorm3d):
            module.eval()


def get_cifar10():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose(
        [
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Assign train/val datasets for use in dataloaders
    cifar_full = torchvision.datasets.CIFAR10(PATH_DATASETS, train=True, download=True,
                                              transform=train_transform)
    cifar10_train, cifar10_val = random_split(cifar_full, [45000, 5000])

    # Assign test dataset for use in dataloader(s)
    cifar10_test = torchvision.datasets.CIFAR10(PATH_DATASETS, train=False, transform=test_transform)

    train_loader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)
    val_loader = DataLoader(cifar10_val, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)
    test_loader = DataLoader(cifar10_test, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)
    return train_loader, val_loader, test_loader


def manual_SGD_optimization(cfg: omegaconf.DictConfig):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

        # Get data
    train_loader, val_loader, test_loader = get_cifar10()
    type_of_model, arguments = model_registry[cfg.model]
    model = type_of_model(*arguments).to(device)
    loss_object = torch.nn.CrossEntropyLoss()

    weight_decay = cfg.weight_decay
    if cfg.weight_decay:

        parameters = train_helper._add_weight_decay(model, weight_decay)
        weight_decay = 0
    else:
        parameters = model.parameters()

    optimizer = torch.optim.SGD(parameters, lr=0.1, momentum=0.9, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    ligthning_model = SimpleWrapper(model=model, loss=loss_object, optimizer=optimizer, scheduler=lr_scheduler)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=cfg.percent_valid_examples,
        checkpoint_callback=False,
        max_epochs=cfg.epochs,
        gpus=AVAIL_GPUS if torch.cuda.is_available() else None
    )
    hyperparameters = cfg
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(ligthning_model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.test(ligthning_model, dataloaders=test_loader)

    return 0


def train_SAM(model: nn.Module,mask:Masking ,optimizer: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader,
              loss_object: typing.Callable, epoch: int, global_step: int, train_flops:float,log_interval: int = 100,
              use_wandb: bool = False):
    model.train()
    _mask_update_counter = 0
    _loss_collector = smoothen_value.SmoothenValue()
    pbar = tqdm.tqdm(total=len(train_loader), dynamic_ncols=True)
    smooth_CE = loss_object

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # first forward-backward step
        predictions = model(data)
        # enable_bn(model)
        loss = smooth_CE(predictions, target)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        # disable_bn(model)
        smooth_CE(model(data), target).backward()
        optimizer.second_step(zero_grad=True)
        if mask:
            mask.apply_mask()
            one_forward_backward_pass = self.sparse_inference_flops * 2
            train_flops += one_forward_backward_pass * 2
        # L2 Regularization

        # Exp avg collection
        _loss_collector.add_value(loss.item())

        # Mask the gradient step
        # stepper = mask if mask else optimizer
        # if (
        #         mask
        #         and masking_apply_when == "step_end"
        #         and global_step < masking_end_when
        #         and ((global_step + 1) % masking_interval) == 0
        # ):
        #     mask.update_connections()
        #     _mask_update_counter += 1
        # else:
        #     stepper.step()

        # Lr scheduler
        # lr_scheduler.step()
        pbar.update(1)
        global_step += 1

        if batch_idx % log_interval == 0:
            msg = f"Train Epoch {epoch} Iters {global_step} Train loss {_loss_collector.smooth:.6f}"
            pbar.set_description(msg)

            if use_wandb :
                log_dict = {"train_loss": loss}
                # if mask:
                #     density = mask.stats.total_density
                #     log_dict = {
                #         **log_dict,
                #         "prune_rate": mask.prune_rate,
                #         "density": density,
                #     }
                wandb.log(
                    log_dict,
                    step=global_step,
                )


def evaluate(model: nn.Module, valLoader: DataLoader, device: torch.device, loss_object: typing.Callable,
             epoch: int, is_test_set: bool = False, use_wandb: bool = False):
    model.eval()
    top1_list = []
    top5_list = []
    loss = 0
    pbar = tqdm.tqdm(total=len(valLoader), dynamic_ncols=True)
    with torch.no_grad():
        for inputs, targets in valLoader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            y_pred = model(inputs)
            loss += loss_object(y_pred, targets).item()

            top_1_accuracy, top_5_accuracy = get_topk_accuracy(
                F.log_softmax(y_pred, dim=1), targets, topk=(1, 5)
            )
            top1_list.append(top_1_accuracy)
            top5_list.append(top_5_accuracy)

            pbar.update(1)

    loss /= len(valLoader)
    mean_top_1_accuracy = torch.tensor(top1_list).mean()
    mean_top_5_accuracy = torch.tensor(top5_list).mean()

    val_or_test = "val" if not is_test_set else "test"
    msg = f"{val_or_test.capitalize()} Epoch {epoch} {val_or_test} loss {loss:.6f} top-1 accuracy" \
          f" {mean_top_1_accuracy:.4f} top-5 accuracy {mean_top_5_accuracy:.4f}"
    pbar.set_description(msg)
    logging.info(msg)

    if use_wandb:
        wandb.log({f"{val_or_test}_loss": loss}, epoch=epoch)
        wandb.log({f"{val_or_test}_accuracy": top_1_accuracy}, epoch=epoch)
        wandb.log({f"{val_or_test}_top_5_accuracy": top_5_accuracy}, epoch=epoch)


def enable_bn(model):
    model.train()


def collect_result(val):
    return val


def init_multi(args):
    ''' store the counter for later use '''
    global found_best_event
    found_best_event = args


def manual_SAM_optimization(cfg: omegaconf.DictConfig):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Get data
    train_loader, val_loader, test_loader = get_cifar10()
    type_of_model, arguments = model_registry[cfg.model]
    model = type_of_model(*arguments).to(device)
    loss_object = torch.nn.CrossEntropyLoss()

    type_of_model, arguments = model_registry["wrn-22-2"]
    # Create an instace of said model
    dummy_model = type_of_model(*arguments)
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.1)
    decay = CosineDecay(T_max=45000 * (cfg.epochs))
    # mask = get_simple_masking(dummy_optimizer, density=cfg.density)
    mask = Masking(dummy_optimizer, prune_rate_decay=decay, density=cfg.density)
    mask.add_module(model)
    mask.apply_mask()

    weight_decay = cfg.weight_decay
    training_FLOPS: float = 0


    if cfg.weight_decay:
        parameters = train_helper._add_weight_decay(model, weight_decay)
        weight_decay = 0
    else:
        parameters = model.parameters()

    optimizer = SAM(params=parameters, base_optimizer=torch.optim.SGD, lr=cfg.learning_rate,
                    rho=cfg.rho,
                    adaptive=cfg.adaptive, momentum=cfg.momentum, weight_decay=weight_decay)


    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, cfg.epochs)
    if cfg.wandb:
        # old code
        # with open(cfg.wandb.api_key) as f:
        #       os.environ["WANDB_API_KEY"] = f.read().strip()
        #       os.environ["WANDB_START_METHOD"] = "thread"

        # new code
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(
            entity="luis_alfredo",
            config=OmegaConf.to_container(cfg, resolve=True),
            project="sparse_learning",
            name=f"SAM_manual",
            reinit=True,
            # save_code=True,
        )
        wandb.watch(model)
    global_step = 0

    for epoch in range(cfg.epochs):

        train_SAM(model,mask, optimizer, device, train_loader, loss_object, epoch, global_step,
                  use_wandb=cfg.wandb,train_flops=training_FLOPS)
        wandb.log("train_FLOPS", training_FLOPS, step=global_step)
        lr_scheduler.step()

        if epoch % cfg.val_interval == 0:
            evaluate(
                model,
                val_loader,
                device,
                loss_object,
                epoch,
                use_wandb=cfg.wandb
            )

    evaluate(
        model,
        test_loader,
        device,
        loss_object,
        cfg.epochs,
        use_wandb=cfg.wandb,
        is_test_set=True)


def single_train_SAM_Ligthning(cfg: omegaconf.DictConfig):
    type_of_model, arguments = model_registry["wrn-22-2"]
    # Create an instace of said model
    dummy_model = type_of_model(*arguments)
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.1)
    decay = CosineDecay(T_max=45000 * (cfg.epochs))
    # mask = get_simple_masking(dummy_optimizer, density=cfg.density)
    mask = Masking(dummy_optimizer, prune_rate_decay=decay, density=cfg.density)
    real_model = type_of_model(*arguments)
    wandb_logger = None
    if cfg.wandb:
        now = date.datetime.now().strftime("%D-%H:%M")
        wandb_logger = WandbLogger(project="sparse_training",
                                   notes="Testing dense SAM training",
                                   name=f"SAM_TEST_density_{cfg.density}")

    model = CIFAR10ModelSAM(model=real_model, mask=mask, learning_rate=cfg.learning_rate, adaptative=cfg.adaptive,
                            rho=cfg.rho)
    # log gradients and model topology
    if cfg.wandb:
        wandb_logger.watch(model)
    trainer = pl.Trainer(
        logger=True if not wandb_logger else wandb_logger,
        limit_val_batches=cfg.percent_valid_examples,
        checkpoint_callback=False,
        max_epochs=cfg.epochs,
        gpus=AVAIL_GPUS if torch.cuda.is_available() else None
    )
    hyperparameters = cfg
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model)
    trainer.test(model)
    return 0


def get_individual_arguments(model_type: str = "mnist"):
    # Init our model
    if model_type == "mnist":
        model = MNISTModel()
    if model_type == "cifarSAM":
        # todo: this is not going to work as it is, It needs other arguments
        model = CIFAR10ModelSAM()
    if model_type == "cifarSGD":
        model = CIFAR10Model()

    # Init DataLoader from MNIST Dataset
    # train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    return model


def run_one_training(mnist_model: pl.LightningModule, epochs: int, target_value: float) -> float:
    global found_best_event
    # Here I wait in each process until I have availability in my GPU
    wait(lambda: len(GPUtil.getFirstAvailable(maxLoad=0.66, maxMemory=0.66)) != 0, waiting_for="GPUs to be available")
    trainer = None
    if not found_best_event.value:
        stoper_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val_acc", stopping_threshold=
        target_value)
        # Initialize a trainer
        trainer = pl.Trainer(
            gpus=AVAIL_GPUS,
            max_epochs=epochs,
            progress_bar_refresh_rate=20,
            callbacks=[stoper_callback]
        )
        # Train the model ⚡
        trainer.fit(mnist_model)

    else:
        trainer = pl.Trainer(
            gpus=AVAIL_GPUS,
            max_epochs=epochs,
            progress_bar_refresh_rate=20
        )
    # Test the model ⚡
    output_test = trainer.test(mnist_model)
    # If I reach the accuracy I want in the test set then I set the event found_best so other processes do not
    # continue training
    if output_test[0]["val_acc"] >= target_value:
        with found_best_event.get_lock():
            found_best_event.value = 1

    return output_test[0]["val_acc"]


@hydra.main(config_path="configs", config_name="population_training_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    number_of_generations = cfg.generations
    global found_best_event
    found_best_event = mp.Value('i', 0)
    # The value to which I stop the process
    target_value = cfg.target_value

    pool = mp.Pool(processes=2, initializer=init_multi, initargs=(found_best_event,))
    model_population = [(get_individual_arguments(), cfg.optimizer.epochs, target_value) for _ in
                        range(cfg.population_size)]
    # val_accuracy = run_one_training(*model_population[0])
    # print(f"Validation accuracy: {val_accuracy}")
    #
    for g in range(number_of_generations):
        # pool = [mp.Process(target=run_one_training, args=(arg, cfg.optimizer.epoch)) for arg in model_population]
        results = pool.starmap_async(func=run_one_training, iterable=model_population, callback=collect_result).get()
        print(f"Results for generation {g}:\n\t{results}")
    print("The end")


if __name__ == '__main__':
    # cfg = omegaconf.DictConfig({
    #     "wandb":False,
    #     "learning_rate": 0.095409631107804,
    #     "rho": 1.5392140101476401,
    #     "adaptive": True,
    #     "weigth_decay": 5e-4,
    #     "momentum": 0.9,
    #     "epochs": 10,
    #     "percent_valid_examples": 0.1,
    #     "density": 0.1
    # }
    cfg = omegaconf.DictConfig({
        "wandb": True,
        "model": "wrn-22-2",
        "learning_rate": 0.1,
        "rho": 2,
        "adaptive": True,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "nesterov": True,
        "epochs": 10,
        "percent_valid_examples": 0.1,
        "density": 0.1,
        "val_interval": 1
    })
    global USABLE_CORES
    USABLE_CORES = 2
    manual_SAM_optimization(cfg)
