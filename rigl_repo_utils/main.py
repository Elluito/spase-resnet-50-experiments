from pathlib import Path
import os
from typing import TYPE_CHECKING
import logging
import hydra
import sparselearning.core
import torch
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import typing
import rigl_repo_utils.main
from .data import get_dataloaders
from .loss import LabelSmoothingCrossEntropy
from .models import registry as model_registry
from sparselearning.core import Masking
import sparselearning as sparseL
import sparselearning.counting as sparseCount
import torch.nn.functional as F
# The following is an dictionary defined at the end of decay file in funcs.py file
from sparselearning.funcs.decay import registry as decay_registry
from sparselearning.counting.ops import get_inference_FLOPs
from sparselearning.utils.accuracy_helper import get_topk_accuracy
from sparselearning.utils.smoothen_value import SmoothenValue
from sparselearning.utils.train_helper import (
    get_optimizer,
    load_weights,
    save_weights,
)
from sparselearning.utils import layer_wise_density

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def static_train_sparse_FLOPs(
        sparse_FLOPs: int, dense_FLOPs: int, iterations: int
) -> float:
    return sparse_FLOPs * 2 * iterations


def RigL_train_FLOPs(
        sparse_FLOPs: float, dense_FLOPs: float, mask_interval: int = 100
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


def train(
        model: "nn.Module",
        mask: "Masking",
        train_loader: "DataLoader",
        optimizer: "optim",
        lr_scheduler: "lr_scheduler",
        global_step: int,
        epoch: int,
        device: torch.device,
        masking_name:str,
        train_flops: float,
        label_smoothing: float = 0.0,
        log_interval: int = 100,
        use_wandb: bool = False,
        masking_apply_when: str = "epoch_end",
        masking_interval: int = 1,
        masking_end_when: int = -1,
        masking_print_FLOPs: bool = False,
) -> "Union[float,int]":
    assert masking_apply_when in ["step_end", "epoch_end"]
    model.train()
    _mask_update_counter = 0
    _loss_collector = SmoothenValue()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    # this loss is if the model does output F.log_max
    # smooth_CE = LabelSmoothingCrossEntropy(label_smoothing)
    # This loss is if the model does NOT output F.log_max() and instead is just the linear layer
    smooth_CE = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = smooth_CE(output, target)
        loss.backward()

        # L2 Regularization

        # Exp avg collection
        _loss_collector.add_value(loss.item())

        # Mask the gradient step
        stepper = mask if mask else optimizer
        if (
                mask
                and masking_apply_when == "step_end"
                and global_step < masking_end_when
                and ((global_step + 1) % masking_interval) == 0
        ):
            mask.update_connections()
            _mask_update_counter += 1
        else:
            stepper.step()

        if masking_name == "RigL":
            train_flops += RigL_train_FLOPs(mask.inference_FLOPs,mask.dense_FLOPs, mask.interval)
        if masking_name == "Static":
            # Here  we assume that  the backward pass consumes appoximately the same number of flops that the forward
            # pass
            train_flops += mask.inference_FLOPs*2
        # Lr scheduler
        lr_scheduler.step()
        pbar.update(1)
        global_step += 1

        if batch_idx % log_interval == 0:
            msg = f"Train Epoch {epoch} Iters {global_step} Mask Updates {_mask_update_counter} Train loss {_loss_collector.smooth:.6f}"
            pbar.set_description(msg)

            if use_wandb:
                log_dict = {"train_loss": loss, "lr": lr_scheduler.get_lr()[0]}
                if mask:
                    density = mask.stats.total_density
                    log_dict = {
                        **log_dict,
                        "prune_rate": mask.prune_rate,
                        "density": density,
                    }
                wandb.log(
                    log_dict,
                    step=global_step,
                )

    density = mask.stats.total_density if mask else 1.0
    msg = f"Train Epoch {epoch} Iters {global_step} Mask Updates {_mask_update_counter} Train loss {_loss_collector.smooth:.6f} Prune Rate {mask.prune_rate if mask else 0:.5f} Density {density:.5f}"

    if masking_print_FLOPs:
        log_dict = {
            "Inference FLOPs": mask.inference_FLOPs / mask.dense_FLOPs,
            "Avg Inference FLOPs": mask.avg_inference_FLOPs / mask.dense_FLOPs,
            # "train_FLOPS": RigL_train_FLOPs(mask.inference_FLOPs * global_step, mask.dense_FLOPs * global_step,
            #                                 masking_interval),
            # "EPOCH": epoch
        }

        log_dict_str = " ".join([f"{k}: {v:.4f}" for (k, v) in log_dict.items()])
        msg = f"{msg} {log_dict_str}"
        if use_wandb:
            wandb.log(
                {
                    **log_dict,
                    "layer-wise-density": layer_wise_density.wandb_bar(mask),
                },
                step=global_step,
            )

    logging.info(msg)

    return _loss_collector.smooth, global_step,train_flops


def evaluate(
        model: "nn.Module",
        loader: "DataLoader",
        global_step: int,
        epoch: int,
        training_flops:float,
        device: torch.device,
        is_test_set: bool = False,
        use_wandb: bool = False,
) -> "Union[float, float]":
    model.eval()

    loss = 0
    correct = 0
    n = 0
    pbar = tqdm(total=len(loader), dynamic_ncols=True)
    smooth_CE = LabelSmoothingCrossEntropy(0.0)  # No smoothing for val

    top_1_accuracy_ll = []
    top_5_accuracy_ll = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss += F.log_softmax(smooth_CE(output, target).item())  # sum up batch loss

            top_1_accuracy, top_5_accuracy = get_topk_accuracy(
                F.log_softmax(output,dim=-1), target, topk=(1, 5)
            )
            top_1_accuracy_ll.append(top_1_accuracy)
            top_5_accuracy_ll.append(top_5_accuracy)

            pbar.update(1)

        loss /= len(loader)
        top_1_accuracy = torch.tensor(top_1_accuracy_ll).mean()
        top_5_accuracy = torch.tensor(top_5_accuracy_ll).mean()

    val_or_test = "val" if not is_test_set else "test"
    msg = f"{val_or_test.capitalize()} Epoch {epoch} Iters {global_step} {val_or_test} loss {loss:.6f} top-1 accuracy {top_1_accuracy:.4f} top-5 accuracy {top_5_accuracy:.4f}"
    pbar.set_description(msg)
    logging.info(msg)

    # Log loss, accuracy
    if use_wandb:

        wandb.log({f"{val_or_test}_loss": loss, f"{val_or_test}_accuracy": top_1_accuracy,
                   f"{val_or_test}_top_5_accuracy": top_5_accuracy, "train_FLOPS": training_flops, "EPOCH": epoch})
        # wandb.log({f"{val_or_test}_loss": loss, "EPOCH": epoch})
        # wandb.log({f"{val_or_test}_accuracy": top_1_accuracy, "EPOCH": epoch})
        # wandb.log({f"{val_or_test}_top_5_accuracy": top_5_accuracy, "EPOCH": epoch})


    return loss, top_1_accuracy


def single_seed_run(cfg: DictConfig) -> typing.Union[float, sparselearning.core.Masking]:
    # Manual seeds
    torch.manual_seed(cfg.seed)
    # Training Step
    train_step = 0
    # Set device
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    #Training flops
    train_flops:float = 0


    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(**cfg.dataset)

    # Select model
    assert (
            cfg.model in model_registry.keys()
    ), f"Select from {','.join(model_registry.keys())}"
    model_class, model_args = model_registry[cfg.model]
    _small_density = cfg.masking.density if cfg.masking.name == "Small_Dense" else 1.0
    model = model_class(*model_args, _small_density).to(device)

    # wandb
    if cfg.wandb.use:
        # old code
        # with open(cfg.wandb.api_key) as f:
        #       os.environ["WANDB_API_KEY"] = f.read().strip()
        #       os.environ["WANDB_START_METHOD"] = "thread"

        # new code
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            reinit=True,
            # save_code=True,
        )
        wandb.watch(model)

    # Training multiplier
    cfg.optimizer.decay_frequency *= cfg.optimizer.training_multiplier
    cfg.optimizer.decay_frequency = int(cfg.optimizer.decay_frequency)

    cfg.optimizer.epochs *= cfg.optimizer.training_multiplier
    cfg.optimizer.epochs = int(cfg.optimizer.epochs)

    if cfg.masking.get("end_when", None):
        cfg.masking.end_when *= cfg.optimizer.training_multiplier
        cfg.masking.end_when = int(cfg.masking.end_when)

    # Setup optimizers, lr schedulers
    optimizer, (lr_scheduler, warmup_scheduler) = get_optimizer(model, **cfg.optimizer)

    # Setup mask
    mask = None
    if not cfg.masking.dense:
        max_iter = (
            cfg.masking.end_when
            if cfg.masking.apply_when == "step_end"
            else cfg.masking.end_when * len(train_loader)
        )

        kwargs = {"prune_rate": cfg.masking.prune_rate, "T_max": max_iter}

        if cfg.masking.decay_schedule == "magnitude-prune":
            kwargs = {
                "final_sparsity": 1 - cfg.masking.final_density,
                "T_max": max_iter,
                "T_start": cfg.masking.start_when,
                "interval": cfg.masking.interval,
            }

        decay = decay_registry[cfg.masking.decay_schedule](**kwargs)

        mask = Masking(
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
        lottery_mask_path = Path(cfg.masking.get("lottery_mask_path", ""))
        mask.add_module(model, lottery_mask_path)

    # Load from checkpoint
    model, optimizer, mask, step, start_epoch, best_val_loss = load_weights(
        model, optimizer, mask, ckpt_dir=cfg.ckpt_dir, resume=cfg.resume
    )

    # Train model
    epoch = 0
    warmup_steps = cfg.optimizer.get("warmup_steps", 0)
    warmup_epochs = warmup_steps / len(train_loader)

    if (cfg.masking.print_FLOPs and cfg.wandb.use) and (start_epoch, step == (0, 0)):
        if mask:
            # Log initial inference flops etc
            log_dict = {
                "Inference FLOPs": mask.inference_FLOPs / mask.dense_FLOPs,
                "Avg Inference FLOPs": mask.avg_inference_FLOPs / mask.dense_FLOPs,
                "layer-wise-density": layer_wise_density.wandb_bar(mask),
            }
            wandb.log(log_dict, step=0)
    # Start the training procedure

    _masking_args = {}
    if mask:
        _masking_args = {
            "masking_apply_when": cfg.masking.apply_when,
            "masking_interval": cfg.masking.interval,
            "masking_end_when": cfg.masking.end_when,
            "masking_print_FLOPs": cfg.masking.get("print_FLOPs", False),
        }
    for epoch in range(start_epoch, cfg.optimizer.epochs):
        # step here is training iters not global steps
        scheduler = lr_scheduler if (epoch >= warmup_epochs) else warmup_scheduler

        _, step,train_flops = train(
            model,
            mask,
            train_loader,
            optimizer,
            scheduler,
            step,
            epoch + 1,
            device,
            masking_name=cfg.masking.name,
            train_flops=train_flops,
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
                train_flops,
                device,
                use_wandb=cfg.wandb.use,
            )

            # Save weights
            if (epoch + 1 == cfg.optimizer.epochs) or (
                    (epoch + 1) % cfg.ckpt_interval == 0
            ):
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
            train_flops,
            device,
            use_wandb=cfg.wandb.use,
        )

    val_loss, val_accuracy = evaluate(
        model,
        test_loader,
        step,
        epoch + 1,
        train_flops,
        device,
        is_test_set=True,
        use_wandb=cfg.wandb.use,
    )

    if cfg.wandb.use:
        # Close wandb context
        wandb.join()

    # training_flops = 0
    # sparse_FLOPS = get_inference_FLOPs(mask, input_tensor=torch.rand(*(1, 3, 32, 32)))
    # dense_FLOPS = mask.dense_FLOPs
    # if cfg.masking.name is "RigL":
    #     training_flops = RigL_train_FLOPs(sparse_FLOPS * step * cfg.dataset.batch_size,
    #                                       dense_FLOPS * step * cfg.dataset.batch_size, cfg.masking.interval)
    # if cfg.masking.name is "Static":
    #
    #     training_flops = sparse_FLOPS * 2 * step * cfg.dataset.batch_size


    return val_accuracy, train_flops


# @hydra.main(config_path="conf/specific",config_name="cifar10_wrn_22_2_static_modified")
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    if cfg.multi_seed:
        val_accuracy_ll = []
        for seed in cfg.multi_seed:
            run_cfg = deepcopy(cfg)
            run_cfg.seed = seed
            run_cfg.ckpt_dir = f"{cfg.ckpt_dir}_seed={seed}"
            val_accuracy, _ = single_seed_run(run_cfg)
            val_accuracy_ll.append(val_accuracy)

        return sum(val_accuracy_ll) / len(val_accuracy_ll)
    else:
        val_accuracy, _ = single_seed_run(cfg)
        return val_accuracy


if __name__ == "__main__":
    main()
