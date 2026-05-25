#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import torch
import yourdfpy
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R


ROOT = Path(__file__).resolve().parent
DATA_FILTER_ROOT = Path("/home/tap/Data-Filter")

sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(DATA_FILTER_ROOT))

from dataset.TROFuncLangDataset import TROFuncDataset, collate_fn  # noqa: E402
from model.trol_graph_pdm_acc import RobotGraph  # noqa: E402
from utils.tro_hand_model import create_hand_model  # noqa: E402

import new_filter as nf  # noqa: E402
from validation.validate_utils import validate_isaac  # noqa: E402


DEFAULT_TEST_JSONL = DATA_FILTER_ROOT / "new_log" / "filtered_current_test05.jsonl"
DEFAULT_TRAIN_JSONL = DATA_FILTER_ROOT / "new_log" / "filtered_current_train95.jsonl"
DEFAULT_CKPT = ROOT / "results" / "shadow_hand_filtered_motion_0519" / "ckpt" / "latest.pth"
DEFAULT_CONFIG = ROOT / "config" / "train_func_float_lang.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "isaac_eval_latest"


class CompatiblePyrokiRetarget:
    def __init__(self, urdf_path, target_link_names):
        urdf = yourdfpy.URDF.load(urdf_path)
        self.robot = pk.Robot.from_urdf(urdf)
        self.target_link_indices = [
            self.robot.links.names.index(name) for name in target_link_names
        ]

    def solve_retarget(self, initial_q, target_link_mats):
        joint_var = self.robot.joint_var_cls(0)

        def solve_single(init_q, target_mats):
            factors = []
            for link_index, target_mat in zip(self.target_link_indices, target_mats):
                target_pose = jaxlie.SE3.from_matrix(target_mat)
                factors.append(
                    pk.costs.pose_cost_analytic_jac(
                        self.robot,
                        joint_var,
                        target_pose,
                        jnp.array(link_index, dtype=jnp.int32),
                        pos_weight=10.0,
                        ori_weight=0.0,
                    )
                )
            factors.append(pk.costs.limit_cost(self.robot, joint_var, weight=10.0))

            problem = jaxls.LeastSquaresProblem(factors, [joint_var]).analyze()
            sol = problem.solve(
                initial_vals=jaxls.VarValues.make([joint_var.with_value(init_q)]),
                linear_solver="dense_cholesky",
                verbose=False,
                termination=jaxls.TerminationConfig(
                    max_iterations=64,
                    early_termination=False,
                ),
                trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
            )
            return sol[joint_var]

        return jax.vmap(solve_single)(initial_q, target_link_mats)


def load_rows(data_root):
    rows = []
    with open(data_root, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_indices(total, max_samples, seed, random_sample):
    limit = total if max_samples <= 0 else min(total, max_samples)
    if random_sample:
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(total, size=limit, replace=False).tolist())
    return list(range(limit))


def object_name_for_row(row):
    return row.get("motion_task_id") or row.get("task_id")


def sanitize_name(name, max_len=140):
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))
    return safe[:max_len].strip("._-") or "object"


def parse_indices_expr(expr, total):
    if not expr:
        return None
    indices = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in {2, 3}:
                raise ValueError(f"Invalid index range: {part}")
            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1]) if pieces[1] else total
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(part))
    out = []
    seen = set()
    for idx in indices:
        if idx < 0:
            idx = total + idx
        if idx < 0 or idx >= total:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {total}")
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def indices_from_result_file(path, total):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = result.get("samples", [])
    if not samples:
        raise ValueError(f"No samples[] found in {path}")
    indices = []
    seen = set()
    for sample in samples:
        idx = int(sample["index"])
        if idx < 0 or idx >= total:
            raise IndexError(f"Index {idx} from {path} out of bounds for dataset of size {total}")
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def load_object_names(path):
    if not path:
        return []
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def select_dataset_indices(rows, args):
    total = len(rows)
    if args.indices_from_result:
        candidates = indices_from_result_file(args.indices_from_result, total)
    elif args.indices:
        candidates = parse_indices_expr(args.indices, total)
    else:
        candidates = list(range(total))

    exact_names = set(args.object_name or [])
    exact_names.update(load_object_names(args.object_file))
    substrings = args.object_substr or []

    if exact_names or substrings:
        filtered = []
        for idx in candidates:
            name = object_name_for_row(rows[idx])
            if exact_names and name in exact_names:
                filtered.append(idx)
            elif substrings and any(token in name for token in substrings):
                filtered.append(idx)
        candidates = filtered

    if args.random_sample:
        rng = np.random.default_rng(args.seed)
        candidates = rng.permutation(np.asarray(candidates, dtype=np.int64)).tolist()

    if args.max_objects > 0:
        selected = []
        per_object_count = defaultdict(int)
        object_order = []
        object_seen = set()
        for idx in candidates:
            name = object_name_for_row(rows[idx])
            if name not in object_seen:
                if len(object_order) >= args.max_objects:
                    continue
                object_seen.add(name)
                object_order.append(name)
            if per_object_count[name] >= args.samples_per_object:
                continue
            per_object_count[name] += 1
            selected.append(idx)
        candidates = selected
    elif args.max_samples > 0:
        candidates = candidates[: args.max_samples]

    return list(candidates)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def target_pose_from_motion_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    obj_pos = data["target_root_pos"].flatten()
    obj_rot = data["target_root_rot"].flatten()
    return np.concatenate([obj_pos, obj_rot]).astype(float).tolist()


def scale_from_motion_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return float(np.asarray(data.get("scale", 1.0)).reshape(-1)[0])


def safe_rotation_error_deg(pred_euler, gt_euler):
    try:
        pred = R.from_euler("XYZ", np.asarray(pred_euler, dtype=np.float64))
        gt = R.from_euler("XYZ", np.asarray(gt_euler, dtype=np.float64))
        return float((pred.inv() * gt).magnitude() * 180.0 / np.pi)
    except Exception:
        return float("nan")


@torch.no_grad()
def q_diagnostics(hand_model, q_pred, q_gt, device):
    q_pred_np = np.array(q_pred, dtype=np.float32, copy=True)
    q_gt_np = np.array(q_gt, dtype=np.float32, copy=True)

    root_translation_error = float(np.linalg.norm(q_pred_np[:3] - q_gt_np[:3]))
    root_rotation_error_deg = safe_rotation_error_deg(q_pred_np[3:6], q_gt_np[3:6])
    joint_abs = np.abs(q_pred_np[6:] - q_gt_np[6:])

    pred_tensor = torch.as_tensor(q_pred_np, dtype=torch.float32, device=device).unsqueeze(0)
    gt_tensor = torch.as_tensor(q_gt_np, dtype=torch.float32, device=device).unsqueeze(0)
    pred_link = hand_model.get_link_se3(pred_tensor)
    gt_link = hand_model.get_link_se3(gt_tensor)
    link_err = torch.linalg.norm(
        pred_link[:, :3, 3] - gt_link[:, :3, 3],
        dim=-1,
    ).detach().cpu().numpy()

    return {
        "root_translation_error": root_translation_error,
        "root_rotation_error_deg": root_rotation_error_deg,
        "joint_mae": float(joint_abs.mean()) if joint_abs.size else 0.0,
        "joint_max_error": float(joint_abs.max()) if joint_abs.size else 0.0,
        "link_pos_mean_error": float(link_err.mean()) if link_err.size else 0.0,
        "link_pos_max_error": float(link_err.max()) if link_err.size else 0.0,
    }


def summarize(items):
    keys = [
        "root_translation_error",
        "root_rotation_error_deg",
        "joint_mae",
        "joint_max_error",
        "link_pos_mean_error",
        "link_pos_max_error",
    ]
    out = {"count": len(items)}
    for key in keys:
        values = [
            float(item["diagnostics"][key])
            for item in items
            if np.isfinite(float(item["diagnostics"].get(key, float("nan"))))
        ]
        if not values:
            out[key] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
        }
    return out


def prepare_isaac_assets(object_name, first_npz_path, work_root):
    data = np.load(first_npz_path, allow_pickle=True)
    urdf_dir = os.path.abspath(os.path.join(work_root, f"object_urdf_{uuid.uuid4().hex}"))
    os.makedirs(urdf_dir, exist_ok=True)
    nf.prepare_object_urdf(object_name, data, urdf_dir)
    return urdf_dir


def write_val_data(joints, target_poses, work_root):
    path = os.path.abspath(os.path.join(work_root, f"isaac_eval_{uuid.uuid4().hex}.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"joint": joints, "target_pose": target_poses}, f)
    return path


def combine_phase_videos(env_dir):
    close_path = Path(env_dir) / "close.mp4"
    force_path = Path(env_dir) / "force.mp4"
    combined_path = Path(env_dir) / "result.mp4"
    phase_paths = [p for p in [close_path, force_path] if p.exists() and p.stat().st_size > 0]
    if not phase_paths:
        return None

    ffmpeg = "/usr/bin/ffmpeg"
    if os.path.exists(ffmpeg):
        concat_list = Path(env_dir) / "concat_videos.txt"
        concat_list.write_text(
            "".join(f"file '{path}'\n" for path in phase_paths),
            encoding="utf-8",
        )
        copy_cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(combined_path),
        ]
        ret = subprocess.run(copy_cmd, capture_output=True, text=True)
        if ret.returncode == 0 and combined_path.exists() and combined_path.stat().st_size > 0:
            return str(combined_path)

        if len(phase_paths) == 2:
            reencode_cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(phase_paths[0]),
                "-i",
                str(phase_paths[1]),
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map",
                "[v]",
                "-pix_fmt",
                "yuv420p",
                str(combined_path),
            ]
            ret = subprocess.run(reencode_cmd, capture_output=True, text=True)
            if ret.returncode == 0 and combined_path.exists() and combined_path.stat().st_size > 0:
                return str(combined_path)
            return {"error": ret.stderr[-1000:] or ret.stdout[-1000:] or "ffmpeg concat failed"}

    try:
        import imageio.v2 as imageio

        wrote_frame = False
        writer = imageio.get_writer(
            combined_path,
            format="FFMPEG",
            fps=10,
            codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        try:
            for phase_path in phase_paths:
                reader = imageio.get_reader(phase_path)
                try:
                    for frame in reader:
                        writer.append_data(frame)
                        wrote_frame = True
                finally:
                    reader.close()
        finally:
            writer.close()
        if wrote_frame and combined_path.exists():
            return str(combined_path)
    except Exception as exc:
        return {"error": repr(exc)}
    return None


def video_info_for_env(env_dir, combine_videos):
    env_dir = Path(env_dir)
    info = {"video_dir": str(env_dir)}
    for name in ["close", "force"]:
        path = env_dir / f"{name}.mp4"
        if path.exists():
            info[f"{name}_video"] = str(path)
    if combine_videos:
        combined = combine_phase_videos(env_dir)
        if isinstance(combined, dict):
            info["combined_video_error"] = combined["error"]
        elif combined:
            info["result_video"] = combined
    pngs = sorted(env_dir.glob("*.png"))
    if pngs:
        info["png_frames"] = len(pngs)
    return info


def write_record_metadata(path, payload):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_success_list(success):
    tensor = torch.as_tensor(success)
    return [bool(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def data_root_for_args(args):
    if args.data_root:
        return Path(args.data_root)
    if args.split == "train":
        return DEFAULT_TRAIN_JSONL
    return DEFAULT_TEST_JSONL


def build_vis_records(dataset, rows, predictions, max_vis_samples, prefer_failures=True):
    if max_vis_samples <= 0:
        return []

    failures = [x for x in predictions if not x.get("success", False)]
    successes = [x for x in predictions if x.get("success", False)]
    ordered = failures + successes if prefer_failures else predictions
    ordered = ordered[:max_vis_samples]

    records = []
    for item in ordered:
        sample = dataset[item["index"]]
        row = rows[item["index"]]
        records.append(
            {
                "index": item["index"],
                "split": item["split"],
                "object_name": item["object_name"],
                "object_glb": row["object_glb"],
                "motion_npz": row["motion_npz"],
                "lang_anno": sample["lang_anno"],
                "success": item.get("success", False),
                "q_pred": torch.as_tensor(item["q_pred"], dtype=torch.float32),
                "q_gt": torch.as_tensor(item["q_gt"], dtype=torch.float32),
                "object_pc": sample["object_pc"].detach().cpu(),
                "target_pose": torch.as_tensor(item["target_pose"], dtype=torch.float32),
                "object_scale": float(item["object_scale"]),
                "diagnostics": item["diagnostics"],
                "isaac_error": item.get("isaac_error"),
            }
        )
    return records


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TRO language model with inference + IK + Isaac validation."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--samples-per-object", type=int, default=1)
    parser.add_argument("--indices", default=None, help="Dataset indices, e.g. 0,7,20:40.")
    parser.add_argument("--indices-from-result", default=None, help="Reuse samples[].index from a previous result json.")
    parser.add_argument("--object-name", action="append", default=[], help="Exact object/task name. Can be repeated.")
    parser.add_argument("--object-file", default=None, help="Text file with one exact object/task name per line.")
    parser.add_argument("--object-substr", action="append", default=[], help="Substring filter for object/task names. Can be repeated.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--record-videos", action="store_true", help="Save Isaac videos like Data-Filter/filter_demo.py.")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--no-combine-videos", dest="combine_videos", action="store_false")
    parser.add_argument("--noisy-std", type=float, default=0.0)
    parser.add_argument("--max-vis-samples", type=int, default=64)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-vis", default=None)
    parser.set_defaults(combine_videos=True)
    args = parser.parse_args()
    if args.samples_per_object < 1:
        raise ValueError("--samples-per-object must be >= 1")
    set_all_seeds(args.seed)

    os.chdir(ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / "tmp" / "isaac_eval"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = data_root_for_args(args)

    config = OmegaConf.load(args.config)
    config.dataset.data_root = str(data_root)
    config.dataset.num_workers = 0
    config.model.mode = "test"
    config.model.inference_config = {"inference_step": int(args.inference_steps)}

    print(f"[setup] split={args.split}")
    print(f"[setup] device={device} isaac_gpu={args.gpu}")
    print(f"[setup] ckpt={args.ckpt}")
    print(f"[setup] data_root={data_root}")

    dataset = TROFuncDataset(
        data_root=str(data_root),
        hand_types=list(config.dataset.hand_types),
        noisy_std=args.noisy_std,
        seed=args.seed,
    )
    rows = load_rows(data_root)
    if len(rows) != len(dataset):
        raise RuntimeError(f"jsonl rows ({len(rows)}) != dataset length ({len(dataset)})")

    indices = select_dataset_indices(rows, args)
    print(f"[setup] selected_samples={len(indices)} / dataset={len(dataset)}")
    print(f"[setup] selected_objects={len({object_name_for_row(rows[i]) for i in indices})}")
    if not indices:
        raise RuntimeError("No samples selected. Check --split/--data-root/object filters/indices.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_json:
        output_json = Path(args.output_json)
    else:
        output_json = output_dir / f"{args.split}_{stamp}_n{len(indices)}_isaac.json"
    if args.output_vis:
        output_vis = Path(args.output_vis)
    else:
        output_vis = output_json.with_name(output_json.stem + "_vis.pt")

    video_root = None
    if args.record_videos:
        if args.video_dir:
            video_root = Path(args.video_dir)
        else:
            video_root = output_json.with_name(output_json.stem + "_videos")
        video_root.mkdir(parents=True, exist_ok=True)
        print(f"[setup] video_root={video_root}")

    model = RobotGraph(**config.model).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    print(
        f"[setup] loaded epoch={int(ckpt.get('epoch', -1))} "
        f"global_step={int(ckpt.get('global_step', -1))}"
    )

    robot_name = list(config.dataset.hand_types)[0]
    hand_model = create_hand_model(robot_name, device=device)
    target_links = list(hand_model.link_names)
    ik_solver = CompatiblePyrokiRetarget(str(hand_model.urdf_path), target_links)
    batch_retarget = jax.jit(ik_solver.solve_retarget)

    predictions = []
    t0 = time.time()
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        samples = [dataset[i] for i in batch_indices]
        batch = move_batch_to_device(collate_fn(samples), device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            predict_link_pose_dict = model.inference(batch)

        target_mat_list = [predict_link_pose_dict[name] for name in target_links]
        target_mats = torch.stack(target_mat_list, dim=1)
        batch_count = target_mats.shape[0]
        initial_q = hand_model.get_initial_q().unsqueeze(0).expand(batch_count, -1)

        predict_q_jnp = batch_retarget(
            initial_q=jnp.array(initial_q.detach().cpu().numpy()),
            target_link_mats=jnp.array(target_mats.detach().cpu().numpy()),
        )
        jax.block_until_ready(predict_q_jnp)
        predict_q = np.asarray(predict_q_jnp, dtype=np.float32)

        for local_i, source_idx in enumerate(batch_indices):
            row = rows[source_idx]
            motion = np.load(row["motion_npz"], allow_pickle=True)
            q_gt = dataset.align_motion_to_model_root(motion, robot_name)
            q_pred = predict_q[local_i]
            diagnostics = q_diagnostics(hand_model, q_pred, q_gt, device)
            object_name = object_name_for_row(row)
            predictions.append(
                {
                    "split": args.split,
                    "index": int(source_idx),
                    "object_name": object_name,
                    "object_glb": row["object_glb"],
                    "motion_npz": row["motion_npz"],
                    "q_pred": q_pred.astype(float).tolist(),
                    "q_gt": q_gt.astype(float).tolist(),
                    "target_pose": target_pose_from_motion_npz(row["motion_npz"]),
                    "object_scale": scale_from_motion_npz(row["motion_npz"]),
                    "diagnostics": diagnostics,
                }
            )
        print(f"[inference] {len(predictions)}/{len(indices)} samples", flush=True)

    grouped = defaultdict(list)
    for item in predictions:
        grouped[item["object_name"]].append(item)

    all_success = []
    per_object = []
    for object_idx, (object_name, items) in enumerate(sorted(grouped.items()), 1):
        work_root = tmp_dir / f"isaac_eval_{uuid.uuid4().hex}"
        os.makedirs(work_root, exist_ok=True)
        val_data_path = None
        record_dir = None
        try:
            urdf_dir = prepare_isaac_assets(object_name, items[0]["motion_npz"], work_root)
            val_data_path = write_val_data(
                [x["q_pred"] for x in items],
                [x["target_pose"] for x in items],
                work_root,
            )
            if video_root is not None:
                record_dir = video_root / f"{object_idx:04d}_{sanitize_name(object_name)}"
                record_dir.mkdir(parents=True, exist_ok=True)
                write_record_metadata(
                    record_dir / "metadata.json",
                    {
                        "object_name": object_name,
                        "split": args.split,
                        "ckpt": str(args.ckpt),
                        "data_root": str(data_root),
                        "samples": items,
                    },
                )
                with open(val_data_path, "r", encoding="utf-8") as src, open(
                    record_dir / "val_data.json", "w", encoding="utf-8"
                ) as dst:
                    dst.write(src.read())
            success = validate_isaac(
                robot_name=robot_name,
                object_name=object_name,
                object_urdf_root=urdf_dir,
                val_data_path=val_data_path,
                uid=f"isaac_eval_{uuid.uuid4().hex}",
                gpu=args.gpu,
                use_gui=args.use_gui,
                record_dir=str(record_dir) if record_dir is not None else None,
            )
            success_list = safe_success_list(success)
            isaac_error = None
        except Exception as exc:
            success_list = [False] * len(items)
            isaac_error = repr(exc)
        finally:
            if val_data_path and os.path.exists(val_data_path):
                os.remove(val_data_path)
            if work_root.exists():
                import shutil

                shutil.rmtree(work_root)

        for env_idx, (item, ok) in enumerate(zip(items, success_list)):
            item["success"] = ok
            if record_dir is not None:
                item["record_dir"] = str(record_dir)
                item.update(video_info_for_env(record_dir / f"env_{env_idx}", args.combine_videos))
            if isaac_error:
                item["isaac_error"] = isaac_error
        all_success.extend(success_list)
        rate = float(sum(success_list) / max(1, len(success_list)))
        if record_dir is not None:
            write_record_metadata(
                record_dir / "metadata.json",
                {
                    "object_name": object_name,
                    "split": args.split,
                    "success": int(sum(success_list)),
                    "samples": len(success_list),
                    "success_rate": rate,
                    "items": items,
                    "isaac_error": isaac_error,
                },
            )
        per_object.append(
            {
                "object_name": object_name,
                "samples": len(success_list),
                "success": int(sum(success_list)),
                "success_rate": rate,
                "isaac_error": isaac_error,
                "record_dir": str(record_dir) if record_dir is not None else None,
            }
        )
        print(
            f"[isaac] {object_idx}/{len(grouped)} {object_name}: "
            f"{sum(success_list)}/{len(success_list)} ({rate:.4f})",
            flush=True,
        )
        if isaac_error:
            print(f"[isaac_error] {object_name}: {isaac_error}", flush=True)

    successes = [x for x in predictions if x.get("success", False)]
    failures = [x for x in predictions if not x.get("success", False)]
    elapsed = time.time() - t0

    result = {
        "ckpt": str(args.ckpt),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "ckpt_global_step": int(ckpt.get("global_step", -1)),
        "split": args.split,
        "data_root": str(data_root),
        "device": str(device),
        "isaac_gpu": args.gpu,
        "inference_steps": args.inference_steps,
        "random_sample": bool(args.random_sample),
        "seed": args.seed,
        "selection": {
            "max_samples": args.max_samples,
            "max_objects": args.max_objects,
            "samples_per_object": args.samples_per_object,
            "indices": args.indices,
            "indices_from_result": args.indices_from_result,
            "object_name": args.object_name,
            "object_file": args.object_file,
            "object_substr": args.object_substr,
        },
        "dataset_size": len(dataset),
        "selected_samples": len(indices),
        "objects": len(grouped),
        "success": int(sum(all_success)),
        "total": len(all_success),
        "success_rate": float(sum(all_success) / max(1, len(all_success))),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(all_success) / max(1e-8, elapsed),
        "success_diagnostics": summarize(successes),
        "failure_diagnostics": summarize(failures),
        "record_videos": bool(args.record_videos),
        "video_root": str(video_root) if video_root is not None else None,
        "per_object": per_object,
        "samples": predictions,
        "note": "Physical Isaac validation after model inference + Pyroki IK. Diagnostics compare predicted q against the filtered motion q_gt.",
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = output_json.with_suffix(output_json.suffix + ".tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_json, output_json)

    vis_records = build_vis_records(
        dataset=dataset,
        rows=rows,
        predictions=predictions,
        max_vis_samples=args.max_vis_samples,
        prefer_failures=True,
    )
    torch.save(
        {
            "result_json": str(output_json),
            "split": args.split,
            "ckpt": str(args.ckpt),
            "records": vis_records,
        },
        output_vis,
    )

    summary = {k: v for k, v in result.items() if k not in {"samples", "per_object"}}
    summary["result_json"] = str(output_json)
    summary["vis_pt"] = str(output_vis)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
