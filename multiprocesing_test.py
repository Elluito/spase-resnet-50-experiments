import os
import multiprocessing as mp
import threading as thread
import logging
import time
from waiting import wait
import hydra
from omegaconf import OmegaConf, DictConfig
from rigl_repo_utils.models import registry as model_registry
from sparselearning.core import  Masking
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
import typing
import GPUtil
from sam import SAM

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

################################################# CLASSSES #############################################################
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


class CIFAR10Model(pl.LightningModule):

    def __init__(self, data_dir=PATH_DATASETS,type_model:str="wrn-22-2",mask:Masking= None, learning_rate=2e-4):

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
        type_of_model,arguments = model_registry[type_model]
        #Create an instace of said model
        self.model = type_of_model(*arguments)
        self.mask = mask
        if mask:
            self.mask.add_module(self.model)

        self.accuracy = Accuracy()

    def forward(self, x):
        if self.mask
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
def collect_result(val):
    return val


def init(args):
    ''' store the counter for later use '''
    global found_best_event
    found_best_event = args


def get_individual_arguments():
    # Init our model
    mnist_model = MNISTModel()

    # Init DataLoader from MNIST Dataset
    # train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    return mnist_model


def run_one_training(mnist_model: MNISTModel, epochs: int, target_value: float) -> float:
    global found_best_event
    # Here I wait in each process until I have availability in my GPU
    wait(lambda: len(GPUtil.getFirstAvailable(maxLoad=0.66, maxMemory=0.66)) != 0, waiting_for="GPUs to be available")
    trainer = None
    if not found_best_event.value:
        stoper_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val_acc", stopping_threshold =
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
    output = trainer.test(mnist_model)
    # If I reach the accuracy I want in the test set then I set the event found_best so other processes do not
    # continue training
    if output[0]["val_acc"] >= target_value:
        with found_best_event.get_lock():
            found_best_event.value = 1

    return output[0]["val_acc"]


@hydra.main(config_path="configs", config_name="population_training_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    number_of_generations = cfg.generations
    # TODO:Need to be sure that this particular event is called in the method on_train_batch_start() from the MNISTModel
    global found_best_event
    found_best_event = mp.Value('i', 0)
   # The value to which I stop the process
    target_value = cfg.target_value

    pool = mp.Pool(processes=2, initializer=init, initargs=(found_best_event,))
    model_population = [(get_individual_arguments(), cfg.optimizer.epochs,target_value) for _ in range(cfg.population_size)]
    # val_accuracy = run_one_training(*model_population[0])
    # print(f"Validation accuracy: {val_accuracy}")
    #
    for g in range(number_of_generations):
        # pool = [mp.Process(target=run_one_training, args=(arg, cfg.optimizer.epoch)) for arg in model_population]
        results = pool.starmap_async(func=run_one_training, iterable=model_population, callback=collect_result).get()
        print(f"Results for generation {g}:\n\t{results}")
    print("The end")


if __name__ == '__main__':
    main()
