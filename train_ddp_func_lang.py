import os
import argparse
import math
import torch
from time import time
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs

from dataset.TROFuncLangDataset import create_dataloader
from model.trol_graph_pdm_acc import RobotGraph

def build_step_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
):
    warmup_steps = int(max(0, warmup_steps))
    total_steps  = int(max(1, total_steps))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    save_dir: str,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    filename: str
):
    ckpt_dir = os.path.join(save_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, filename)
    tmp_path = f"{ckpt_path}.tmp"

    unwrapped = accelerator.unwrap_model(model)
    payload = {
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "global_step": int(global_step),
        "model_state": unwrapped.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }

    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)
    return ckpt_path


def load_checkpoint(accelerator: Accelerator, model, optimizer, scheduler, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    unwrapped = accelerator.unwrap_model(model)

    unwrapped.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state", None) is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    global_step = int(ckpt.get("global_step", 0))
    batch_in_epoch = int(ckpt.get("batch_in_epoch", 0))
    epoch = int(ckpt.get("epoch", 0))
    return epoch, batch_in_epoch, global_step


def train(config):

    # DDP and Accelerator
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    ddp_kwargs = DistributedDataParallelKwargs(
        gradient_as_bucket_view=True,
        bucket_cap_mb=50,
        find_unused_parameters=False
    )
    accelerator = Accelerator(
        mixed_precision=config.train.mixed_precision,
        gradient_accumulation_steps=config.train.accumulator,
        kwargs_handlers=[ddp_kwargs]
    )
    print(f"[rank={accelerator.process_index}] world={accelerator.num_processes} local_rank={accelerator.local_process_index}", flush=True)
    seed = int(getattr(config.train, "seed", 0) or 0)
    set_seed(seed)

    # Output dir
    save_dir = config.train.save_dir
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    tb_writer = None
    if accelerator.is_main_process:
        tb_dir = os.path.join(save_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_dir)

###dataloader
    if accelerator.is_main_process:
        print("Building dataloader...")
    dataloader = create_dataloader(config.dataset, is_train=True)
###end

###modelloader
    if accelerator.is_main_process:
        print("Building model...")
    model = RobotGraph(**config.model)
###end

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.train.lr),
        fused=False,
    )

    print(f"[rank={accelerator.process_index}] Accelerator Preparing...", flush=True)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Scheduler
    warmup_steps = config.train.warmup_steps
    num_epoch = config.train.num_epoch
    scheduler = build_step_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=num_epoch * len(dataloader)
    )

    # Log total params
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        total_params = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
        print(f"Total params: {total_params}", flush=True)

    # Resume
    start_epoch = 0
    resume_batch_in_epoch = 0
    global_step = 0
    resume_from = str(getattr(config.train, "resume_from", "")).strip()
    if resume_from:
        start_epoch, resume_batch_in_epoch, global_step = load_checkpoint(
            accelerator, model, optimizer, scheduler, resume_from
        )
        if global_step > 0:
            scheduler.last_epoch = global_step - 1
            scheduler.step()
        if accelerator.is_main_process:
            print(
                f"Resumed from {resume_from} at "
                f"epoch={start_epoch}, batch_in_epoch={resume_batch_in_epoch}, "
                f"global_step={global_step}"
            )

    save_every = config.train.save_every
    save_steps = int(getattr(config.train, "save_steps", 0) or 0)
    max_steps = int(getattr(config.train, "max_steps", 0) or 0)
    log_every = config.train.log_every
    grad_clip = config.train.grad_clip

    print(f"[rank={accelerator.process_index}] Start Training...", flush=True)

    # Training
    model.train()
    time_start = time()
    last_log_time = time_start

    running_loss_sum = 0.0
    running_loss_count = 0
    stop_training = False

    for epoch in range(start_epoch, num_epoch):
        batch_in_epoch = 0
        for batch in dataloader:
            if epoch == start_epoch and batch_in_epoch < resume_batch_in_epoch:
                batch_in_epoch += 1
                continue

            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)

                # forward
                with accelerator.autocast():
                    loss_dict = model(batch)
                    loss = loss_dict["loss_total"]

                # Backward
                accelerator.backward(loss)
                
                # Update Para
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                
                    scheduler.step()
                    global_step += 1
                    if max_steps > 0 and global_step >= max_steps:
                        stop_training = True

            loss_local = loss.detach()
            loss_mean = loss_local.item()
            running_loss_sum += loss_mean
            running_loss_count += 1
            batch_in_epoch += 1

            do_log = (batch_in_epoch % log_every == 0)
            if do_log:
                with torch.no_grad():
                    loss_global = accelerator.gather_for_metrics(loss_local).mean().item()
                    trans_global = accelerator.gather_for_metrics(loss_dict["loss_trans"].detach()).mean().item()
                    rot_global   = accelerator.gather_for_metrics(loss_dict["loss_rot"].detach()).mean().item()

                if accelerator.is_main_process:
                    tb_writer.add_scalar("train/loss_total", loss_global, global_step)
                    tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    tb_writer.add_scalar("train/loss_trans", trans_global, global_step)
                    tb_writer.add_scalar("train/loss_rot", rot_global, global_step)

                    now = time()
                    avg_run = running_loss_sum / max(1, running_loss_count)
                    it_s = (now - last_log_time) / float(log_every)

                    print(
                        f"[epoch {epoch + 1}] "
                        f"batch={batch_in_epoch:>5d} | "
                        f"loss={loss_global:.6f} | "
                        f"avg_local={avg_run:.6f} | "
                        f"lr={optimizer.param_groups[0]['lr']:.3e} | "
                        f"{it_s:.3f}s/iter",
                        flush=True,
                    )

                    last_log_time = now
                    running_loss_sum = 0.0
                    running_loss_count = 0

            if (
                accelerator.is_main_process
                and accelerator.sync_gradients
                and save_steps > 0
                and global_step > 0
                and (global_step % save_steps == 0)
            ):
                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    save_dir,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    global_step=global_step,
                    filename="latest.pth",
                )

            if stop_training:
                break

        if accelerator.is_main_process:
            save_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                save_dir,
                epoch=epoch + 1,
                batch_in_epoch=0,
                global_step=global_step,
                filename="latest.pth",
            )
            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    save_dir,
                    epoch=epoch + 1,
                    batch_in_epoch=0,
                    global_step=global_step,
                    filename=f"epoch_{epoch + 1}.pth",
                )

        if stop_training:
            break

    # End training
    accelerator.end_training()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    train(config)

if __name__ == "__main__":
    main()
