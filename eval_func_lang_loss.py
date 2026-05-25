import argparse
import json
import math
import time

import torch
from omegaconf import OmegaConf

from dataset.TROFuncLangDataset import create_dataloader
from model.trol_graph_pdm_acc import RobotGraph


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def parse_thresholds(text):
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def summarize_values(values):
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "min": float(tensor.min()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "max": float(tensor.max()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument(
        "--loss-thresholds",
        default="0.1,0.25,0.5,1.0",
        help="Comma-separated thresholds used for batch-level loss accuracy.",
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    config = OmegaConf.load(args.config)
    config.dataset.data_root = args.data_root
    config.dataset.num_workers = min(int(config.dataset.num_workers), 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    dataloader = create_dataloader(config.dataset, is_train=False)

    model = RobotGraph(**config.model).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    totals = {"loss_total": 0.0, "loss_trans": 0.0, "loss_rot": 0.0}
    batch_losses = {key: [] for key in totals}
    threshold_hits = {threshold: 0 for threshold in parse_thresholds(args.loss_thresholds)}
    count = 0
    batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss_dict = model(batch)

        batch_size = int(batch["object_pc"].shape[0])
        for key in totals:
            value = float(loss_dict[key].detach().cpu())
            totals[key] += value * batch_size
            batch_losses[key].append(value)
        for threshold in threshold_hits:
            if batch_losses["loss_total"][-1] <= threshold:
                threshold_hits[threshold] += 1
        count += batch_size
        batches += 1
        if args.max_batches > 0 and batches >= args.max_batches:
            break

    elapsed = time.time() - start_time
    dataset_size = len(dataloader.dataset)
    result = {
        "ckpt": args.ckpt,
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "ckpt_global_step": int(ckpt.get("global_step", -1)),
        "ckpt_batch_in_epoch": int(ckpt.get("batch_in_epoch", -1)),
        "data_root": args.data_root,
        "device": str(device),
        "dataset_size": dataset_size,
        "max_batches": args.max_batches,
        "batches": batches,
        "samples": count,
        "eval_fraction": count / max(1, dataset_size),
        "elapsed_seconds": elapsed,
        "samples_per_second": count / max(1e-8, elapsed),
        "batches_per_second": batches / max(1e-8, elapsed),
    }
    for key, value in totals.items():
        avg = value / max(1, count)
        result[key] = avg
        result[f"{key}_rmse"] = math.sqrt(max(0.0, avg))
        result[f"{key}_batch_stats"] = summarize_values(batch_losses[key])

    result["loss_total_threshold_accuracy"] = {
        f"<= {threshold:g}": threshold_hits[threshold] / max(1, batches)
        for threshold in threshold_hits
    }
    result["accuracy_note"] = (
        "This is batch-level threshold accuracy over diffusion loss_total, "
        "not physical grasp success rate. Physical success needs inference + IK + Isaac validation."
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
