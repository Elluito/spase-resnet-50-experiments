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
from sparselearning.funcs.decay import registry as decay_registry
from sparselearning.core import Masking
from sparselearning.counting.ops import get_inference_FLOPs
from sparselearning.utils import layer_wise_density
#torch imports
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

PATH_DATASETS = ""
if "Linux" in platform.system():
    PATH_DATASETS = os.environ.get("PATH_DATASETS", "/nobackup/sclaam")
else:
    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())

BATCH_SIZE = 128 if AVAIL_GPUS else 64
# USABLE_CORES = len(os.sched_getaffinity(0)) if "Linux" in platform.system() else 2
USABLE_CORES = os.cpu_count()//3 if "Linux" in platform.system() else 2
print(f"Usable cores {USABLE_CORES}")
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

    def __init__(self, data_dir=PATH_DATASETS, type_model: str = "wrn-22-2", mask: Masking = None,
                 learning_rate: float = 2e-4, rho: float = 0.05, adaptative: bool = False, decay_frequency: int = 100,
                 decay_factor: float = 0.1):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.rho = rho
        self.adaptive = adaptative
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
        type_of_model, arguments = model_registry[type_model]
        # Create an instace of said model
        self.model = type_of_model(*arguments)
        self.mask = mask
        if mask:
            self.mask.add_module(self.model, lottery_mask_path=path.Path(""))
            self.mask.to_module_device_()
            self.mask.apply_mask()
        self.loss_object = F.nll_loss
        self.accuracy = Accuracy()
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

        optimizer = self.optimizers()

        # first forward-backward pass
        enable_bn(self.model)
        loss_1 = self.compute_loss(batch)
        self.manual_backward(loss_1)
        if self.mask:
            self.mask.to_module_device_()
            self.mask.apply_mask_gradients()
        optimizer.first_step(zero_grad=True)

        # second forward-backward pass
        disable_bn(self.model)
        loss_2 = self.compute_loss(batch)
        self.manual_backward(loss_2)
        if self.mask:
            self.mask.apply_mask_gradients()
        optimizer.second_step(zero_grad=True)
        # Flops for one forward pass
        sparse_inference_FLOPS = get_inference_FLOPs(self.mask, input_tensor=torch.rand(*(1, 3, 32, 32)))
        # Assuming that the backward pass uses approximately the same number of Flops as the
        # forward.
        # Acording to  this website shorturl.at/hpAP7 they said that is between 2 and 3 times the
        # number of FLOPS
        one_forward_backward_pass = sparse_inference_FLOPS * 2

        self.training_FLOPS += one_forward_backward_pass * 2
        # self.log("FLOPS", self.training_FLOPS)

        return loss_1

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = self.accuracy(preds, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return {"val_loss": loss, "val_acc": accuracy}

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = {k: [dic[k] for dic in validation_step_outputs] for k in validation_step_outputs[0]}
        self.log("Avg_acc",
                 torch.tensor(all_preds["val_acc"], dtype=torch.float32).mean())
        #self.log("Epoch_FLOPS", self.training_FLOPS)
        log_dict = {
            "Inference FLOPs": self.mask.inference_FLOPs / self.mask.dense_FLOPs,
            "Avg Inference FLOPs": self.mask.avg_inference_FLOPs / self.mask.dense_FLOPs,
        }
        if isinstance(self.logger,WandbLogger):
            log_dict["layer-wise-density"] = layer_wise_density.wandb_bar(self.mask)
        self.log_dict(**log_dict)




    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = SAM(params=self.model.parameters(), base_optimizer=torch.optim.SGD, lr=self.learning_rate,
                        rho=self.rho,
                        adaptive=self.adaptive)
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


def enable_bn(model):
    model.train()


def collect_result(val):
    return val


def init(args):
    ''' store the counter for later use '''
    global found_best_event
    found_best_event = args


def single_train_SAM(cfg: omegaconf.DictConfig):
    type_of_model, arguments = model_registry["wrn-22-2"]
    # Create an instace of said model
    dummy_model = type_of_model(*arguments)
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.1)
    mask = get_simple_masking(dummy_optimizer, density=cfg.density)
    wandb_logger = None
    if cfg.wandb:
        now = date.datetime.now().strftime("%D-%H:%M")
        wandb_logger = WandbLogger(project="sparse_training",
                                   notes="Testing WAND logging capabilities in the cluster",
                                   name=f"SAM_{now}_density_{cfg.density}")

    model = CIFAR10ModelSAM(mask=mask, learning_rate=cfg.learning_rate, adaptative=cfg.adaptive, rho=cfg.rho)
    # log gradients and model topology
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
    wandb_logger.finalize()
    wandb.finish()
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

    pool = mp.Pool(processes=2, initializer=init, initargs=(found_best_event,))
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
    cfg = omegaconf.DictConfig({
        "wandb": True,
        "learning_rate": 0.09540963110780444,
        "rho": 1.5392140101476401,
        "adaptive": True,
        "epochs": 10,
        "percent_valid_examples": 0.1,
        "density": 0.05
    })
    single_train_SAM(cfg)
