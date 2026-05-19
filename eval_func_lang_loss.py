import argparse
import json

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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--max-batches", type=int, default=50)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    config = OmegaConf.load(args.config)
    config.dataset.data_root = args.data_root
    config.dataset.num_workers = min(int(config.dataset.num_workers), 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = create_dataloader(config.dataset, is_train=False)

    model = RobotGraph(**config.model).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    totals = {"loss_total": 0.0, "loss_trans": 0.0, "loss_rot": 0.0}
    count = 0
    batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss_dict = model(batch)

        batch_size = int(batch["object_pc"].shape[0])
        for key in totals:
            totals[key] += float(loss_dict[key].detach().cpu()) * batch_size
        count += batch_size
        batches += 1
        if args.max_batches > 0 and batches >= args.max_batches:
            break

    result = {
        "ckpt": args.ckpt,
        "data_root": args.data_root,
        "device": str(device),
        "batches": batches,
        "samples": count,
    }
    for key, value in totals.items():
        result[key] = value / max(1, count)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
