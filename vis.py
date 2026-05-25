#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from utils.tro_hand_model import create_hand_model


ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = ROOT / "results" / "isaac_eval_latest"


def latest_vis_file():
    files = sorted(DEFAULT_EVAL_DIR.glob("*_vis.pt"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No *_vis.pt found in {DEFAULT_EVAL_DIR}")
    return files[-1]


def load_records_from_result_json(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for sample in data.get("samples", []):
        record = dict(sample)
        if "q_pred" in record:
            record["q_pred"] = torch.as_tensor(record["q_pred"], dtype=torch.float32)
        if "q_gt" in record:
            record["q_gt"] = torch.as_tensor(record["q_gt"], dtype=torch.float32)
        if "target_pose" in record:
            record["target_pose"] = torch.as_tensor(record["target_pose"], dtype=torch.float32)
        records.append(record)
    return {
        "result_json": str(path),
        "split": data.get("split"),
        "ckpt": data.get("ckpt"),
        "records": records,
    }


def load_vis_data(vis_pt, result_json):
    if result_json is not None:
        return load_records_from_result_json(result_json), Path(result_json)
    vis_path = latest_vis_file() if vis_pt == "latest" else Path(vis_pt)
    return torch.load(vis_path, map_location="cpu", weights_only=False), vis_path


def load_mesh_any(path):
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type {type(loaded)} from {path}")
    return loaded


def pose_matrix(target_pose):
    target_pose = np.asarray(target_pose, dtype=np.float64).reshape(-1)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = target_pose[:3]
    mat[:3, :3] = R.from_quat(target_pose[3:7]).as_matrix()
    return mat


def object_mesh_for_record(record):
    mesh = load_mesh_any(record["object_glb"]).copy()
    scale = float(record.get("object_scale", 1.0))
    mesh.vertices = mesh.vertices * scale
    mesh.apply_transform(pose_matrix(record["target_pose"]))
    return mesh


def translated(mesh, offset):
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices + offset.reshape(1, 3)
    return mesh


def add_mesh(server, name, mesh, color, opacity=1.0):
    server.scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        color=color,
        opacity=opacity,
    )


def add_point_cloud(server, name, points, color):
    server.scene.add_point_cloud(
        name,
        points=np.asarray(points, dtype=np.float32),
        colors=np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 3), (len(points), 1)),
        point_size=0.004,
    )


def diagnostic_value(record, key):
    return float(record.get("diagnostics", {}).get(key, 0.0))


def choose_records(records, only_failures, only_successes, start, count, sort_by):
    filtered = records
    if only_failures:
        filtered = [x for x in filtered if not bool(x.get("success", False))]
    if only_successes:
        filtered = [x for x in filtered if bool(x.get("success", False))]
    if sort_by:
        filtered = sorted(filtered, key=lambda x: diagnostic_value(x, sort_by), reverse=True)
    return filtered[start : start + count]


def print_record(i, record):
    status = "SUCCESS" if record.get("success", False) else "FAIL"
    diag = record.get("diagnostics", {})
    print("=" * 88)
    print(f"[{i}] {status} split={record.get('split')} index={record.get('index')}")
    print(f"object: {record.get('object_name')}")
    print(f"motion: {record.get('motion_npz')}")
    print("language:")
    for line in record.get("lang_anno", []):
        print(f"  - {line}")
    print("diagnostics:")
    for key in [
        "root_translation_error",
        "root_rotation_error_deg",
        "joint_mae",
        "joint_max_error",
        "link_pos_mean_error",
        "link_pos_max_error",
    ]:
        if key in diag:
            print(f"  {key}: {diag[key]}")
    if record.get("isaac_error"):
        print(f"isaac_error: {record['isaac_error']}")
    if record.get("result_video"):
        print(f"result_video: {record['result_video']}")
    elif record.get("close_video") or record.get("force_video"):
        print(f"close_video: {record.get('close_video')}")
        print(f"force_video: {record.get('force_video')}")


def main():
    parser = argparse.ArgumentParser(description="Visualize TRO Isaac eval records.")
    parser.add_argument("--vis-pt", default="latest")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--only-failures", action="store_true")
    parser.add_argument("--only-successes", action="store_true")
    parser.add_argument("--sort-by", default=None, choices=[
        "root_translation_error",
        "root_rotation_error_deg",
        "joint_mae",
        "joint_max_error",
        "link_pos_mean_error",
        "link_pos_max_error",
    ])
    parser.add_argument("--spacing", type=float, default=0.45)
    parser.add_argument("--no-gt", action="store_true")
    parser.add_argument("--no-point-cloud", action="store_true")
    args = parser.parse_args()

    data, source_path = load_vis_data(args.vis_pt, args.result_json)
    records = data.get("records", [])
    if not records:
        raise RuntimeError(f"No records found in {source_path}")

    selected = choose_records(
        records,
        only_failures=args.only_failures,
        only_successes=args.only_successes,
        start=args.start,
        count=args.count,
        sort_by=args.sort_by,
    )
    if not selected:
        raise RuntimeError("No records selected. Try removing filters or changing --start.")

    print(f"source: {source_path}")
    if data.get("result_json"):
        print(f"result_json: {data['result_json']}")

    for local_i, record in enumerate(selected):
        print_record(local_i, record)

    print("Starting viser server and loading hand/object meshes...")
    server = viser.ViserServer(host=args.host, port=args.port)
    hand_model = create_hand_model("shadow_hand", device=torch.device("cpu"))

    for local_i, record in enumerate(selected):
        offset = np.array([local_i * args.spacing, 0.0, 0.0], dtype=np.float32)
        root = f"/sample_{local_i}_{'ok' if record.get('success') else 'fail'}"

        object_mesh = translated(object_mesh_for_record(record), offset)
        add_mesh(server, f"{root}/object", object_mesh, color=(160, 160, 160), opacity=0.38)

        pred_mesh = hand_model.get_trimesh_q(record["q_pred"])["visual"]
        pred_mesh = translated(pred_mesh, offset)
        pred_color = (38, 122, 255) if record.get("success") else (230, 68, 68)
        add_mesh(server, f"{root}/pred_hand", pred_mesh, color=pred_color, opacity=0.84)

        if not args.no_gt:
            gt_mesh = hand_model.get_trimesh_q(record["q_gt"])["visual"]
            gt_mesh = translated(gt_mesh, offset)
            add_mesh(server, f"{root}/gt_hand", gt_mesh, color=(45, 180, 98), opacity=0.34)

        if not args.no_point_cloud and "object_pc" in record:
            points = record["object_pc"].detach().cpu().numpy() + offset.reshape(1, 3)
            add_point_cloud(server, f"{root}/object_pc", points, color=(255, 196, 65))

    print("=" * 88)
    print(f"Open the viser page from the terminal output. Server: http://{args.host}:{args.port}")
    print("Color: red/blue = prediction, green = ground truth, gray = object mesh, yellow = object point cloud.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
