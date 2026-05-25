import os
import glob
import json
import trimesh
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

from utils.tro_hand_model import create_hand_model


isaac_to_pk_link_dict = {
    "shadow_hand": {
        "isaac": [
            "FFJ4", "FFJ3", "FFJ2", "FFJ1",
            "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
            "MFJ4", "MFJ3", "MFJ2", "MFJ1",
            "RFJ4", "RFJ3", "RFJ2", "RFJ1",
            "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
        ],
        "pk": [
            "FFJ4", "FFJ3", "FFJ2", "FFJ1",
            "MFJ4", "MFJ3", "MFJ2", "MFJ1",
            "RFJ4", "RFJ3", "RFJ2", "RFJ1",
            "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
            "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
        ],
    }
}


def isaac_to_pk(raw_joint_vals, hand):
    isaac_names = isaac_to_pk_link_dict[hand]["isaac"]
    pk_names = isaac_to_pk_link_dict[hand]["pk"]
    name2idx_isaac = {name: i for i, name in enumerate(isaac_names)}
    pk_vals = np.array([raw_joint_vals[name2idx_isaac[name]] for name in pk_names], dtype=np.float32)

    if hand == "shadow_hand":
        pk_vals[0] *= -1
        pk_vals[4] *= -1
        pk_vals[-2] *= -1
        pk_vals[-1] *= -1
    return pk_vals

class TROFuncDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        hand_types: list,
        noisy_std: float = 0.002,
        max_links: int = 25,
        num_object_points: int = 2048, # 注意：需与你 yaml 里的 object_patch 匹配
        dtype: torch.dtype = torch.float32,
        seed: int = 0
    ):
        super().__init__()
        self.data_root = data_root
        self.hand_types = hand_types
        
        # 机器人名字到 ID 的映射
        self.robot2id = {name: i for i, name in enumerate(self.hand_types)}

        self.max_links = max_links
        self.noisy_std = noisy_std
        self.num_object_points = num_object_points
        self.dtype = dtype
        self.seed = seed
        self.hand_models = {
            hand_type: create_hand_model(hand_type, device=torch.device("cpu"))
            for hand_type in self.hand_types
        }

        # ==========================================
        # 1. 遍历并加载所有的 map.jsonl
        # ==========================================
        self.data_list = []
        
        # 兼容直接读取总表文件的情况
        if os.path.isfile(self.data_root) and self.data_root.endswith('.jsonl'):
            jsonl_files = [self.data_root]
        else:
            jsonl_pattern = os.path.join(self.data_root, "000-*", "map.jsonl")
            jsonl_files = glob.glob(jsonl_pattern)
            
            if len(jsonl_files) == 0:
                fallback_pattern = os.path.join(self.data_root, "map.jsonl")
                jsonl_files = glob.glob(fallback_pattern)

        # 排序以保证多卡训练时的数据集切分顺序一致性
        jsonl_files.sort() 

        for jsonl_path in tqdm(jsonl_files, desc="Loading JSONL Annotations"):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        # 核心校验：只保留包含语言标注的合法数据
                        if "grasp_description" in item:
                            self.data_list.append(item)
        
        print(f"Successfully loaded {len(self.data_list)} tasks from {len(jsonl_files)} JSONL files.")

    def __len__(self):
        return len(self.data_list)

    def annotation_to_lines(self, annotation):
        """将 JSON 字典中的抓取特征，转换为固定长度的自然语言描述列表"""
        obj = annotation.get("object_info", {})
        contact = annotation.get("contact_info", {})
        func = annotation.get("functional_context", {})

        # ---- object info ----
        obj_name = obj.get("name", "unknown")
        obj_color = obj.get("color", "unknown")
        # ---- target part ----
        target_part = annotation.get("target_part", "unknown")
        # ---- contact info ----
        finger_count = contact.get("finger_count", "unknown")
        grasp_style = contact.get("grasp_style", "unknown")
        contact_fingers = contact.get("contact_fingers", [])
        fingers_str = " and ".join(contact_fingers) if contact_fingers else "none"
        # ---- functional context ----
        intent = func.get("intent", "none")
        constraints = func.get("constraints", "none")
        spatial_relation = obj.get("spatial_relation", "none")

        lines = [
            f"Object: {obj_name}, color: {obj_color}.",
            f"Target part: {target_part}.",
            f"Contact: {finger_count} fingers, {grasp_style}, fingers: {fingers_str}.",
            f"Intent: {intent}.",
            f"Constraints: {constraints}.",
            f"Spatial relation: {spatial_relation}."
        ]
        return lines

    def _load_target_vec(self, retarget_json_path):
        """直接从 JSON 文件读取 6D 姿态矩阵"""
        poses = torch.zeros((self.max_links, 6), dtype=self.dtype)
        try:
            with open(retarget_json_path, 'r', encoding='utf-8') as f:
                pose_data = json.load(f)
            
            # 提取每个连杆的 [平移(3), 旋转(3)]
            for i, link_pose in enumerate(pose_data):
                if i >= self.max_links:
                    break
                trans = link_pose.get('translation', [0.0, 0.0, 0.0])
                rot = link_pose.get('rotation', [0.0, 0.0, 0.0])
                
                pose = np.concatenate((trans, rot), axis=0)
                poses[i] = torch.from_numpy(pose)
        except Exception as e:
            pass
        return poses

    def get_se3_from_trans_and_quat(self, trans, quat):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = np.asarray(trans, dtype=np.float32)
        T[:3, :3] = R.from_quat(np.asarray(quat, dtype=np.float32)).as_matrix()
        return T

    def get_se3_from_trans_and_rpy(self, trans, rpy):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = np.asarray(trans, dtype=np.float32)
        T[:3, :3] = R.from_euler("XYZ", np.asarray(rpy, dtype=np.float32)).as_matrix()
        return T

    def align_motion_to_model_root(self, motion_data, robot_name):
        if robot_name != "shadow_hand":
            raise NotImplementedError(f"motion_npz path is only implemented for shadow_hand, got {robot_name}")

        obj_pos = motion_data["target_root_pos"].flatten()
        obj_rot = motion_data["target_root_rot"].flatten()
        T_obj_world = self.get_se3_from_trans_and_quat(obj_pos, obj_rot)
        T_palm_rel = self.get_se3_from_trans_and_quat(
            motion_data["hand_root_rel_pos"].flatten(),
            motion_data["hand_root_rel_rot"].flatten(),
        )
        T_world_raw = T_obj_world @ T_palm_rel

        # Keep this alignment exactly in sync with new_filter.py/filter_demo.py.
        T_world_palm_offset = self.get_se3_from_trans_and_rpy(
            trans=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            rpy=np.array([0.0, -1.5707963267948966, 0.0], dtype=np.float32),
        )
        T_world_palm = T_world_raw @ T_world_palm_offset

        T_vyaw_cyl = self.get_se3_from_trans_and_rpy(
            trans=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            rpy=np.array([0.0, 0.0, 1.57079], dtype=np.float32),
        )
        T_world_base = T_world_palm @ np.linalg.inv(T_vyaw_cyl)

        base_trans = T_world_base[:3, 3]
        base_rpy = R.from_matrix(T_world_base[:3, :3]).as_euler("XYZ")
        mapped_joints = isaac_to_pk(motion_data["joint_vals"].flatten(), robot_name)
        return np.concatenate([base_trans, base_rpy, mapped_joints], axis=-1).astype(np.float32)

    def matrix_to_pose_vec(self, link_se3):
        poses = torch.zeros((self.max_links, 6), dtype=self.dtype)
        link_count = min(self.max_links, int(link_se3.shape[0]))
        for i in range(link_count):
            mat = link_se3[i].detach().cpu().numpy()
            trans = mat[:3, 3]
            rotvec = R.from_matrix(mat[:3, :3]).as_rotvec()
            poses[i] = torch.from_numpy(np.concatenate([trans, rotvec], axis=0)).to(self.dtype)
        return poses

    def load_motion_target_vec(self, motion_data, robot_name):
        q = self.align_motion_to_model_root(motion_data, robot_name)
        q_tensor = torch.from_numpy(q).to(dtype=torch.float32).unsqueeze(0)
        link_se3 = self.hand_models[robot_name].get_link_se3(q_tensor)
        return self.matrix_to_pose_vec(link_se3)

    def transform_object_points_with_motion(self, points, motion_data):
        scale = float(np.asarray(motion_data.get("scale", 1.0)).reshape(-1)[0])
        points = np.asarray(points, dtype=np.float32) * scale

        T_obj_world = self.get_se3_from_trans_and_quat(
            motion_data["target_root_pos"].flatten(),
            motion_data["target_root_rot"].flatten(),
        )
        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        return (T_obj_world @ points_h.T).T[:, :3].astype(np.float32)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. 动态获取机械手类型
        robot_name = item.get("robot_name", self.hand_types[0])
        robot_id = self.robot2id.get(robot_name, 0)

        # 2. 语言标注处理
        lang_anno = item["grasp_description"]["description"]
        lang_line_list = self.annotation_to_lines(lang_anno)

        motion_data = None
        if "motion_npz" in item and item["motion_npz"]:
            motion_data = np.load(item["motion_npz"], allow_pickle=True)
            target_vec = self.load_motion_target_vec(motion_data, robot_name)
        else:
            target_vec = self._load_target_vec(item["retarget_json"])

        # 4. 读取与采样物体点云
        glb_path = item["object_glb"]
        try:
            mesh = trimesh.load(glb_path, force='mesh', process=False)
            points, _ = trimesh.sample.sample_surface(mesh, self.num_object_points)
            if motion_data is not None:
                points = self.transform_object_points_with_motion(points, motion_data)
            object_pc = torch.from_numpy(points).to(self.dtype)
        except Exception as e:
            object_pc = torch.zeros((self.num_object_points, 3), dtype=self.dtype)

        # 5. 添加少许高斯噪声做数据增强
        object_pc += torch.randn(object_pc.shape) * self.noisy_std

        return {
            "robot_id": torch.tensor([robot_id], dtype=torch.long),
            "target_vec": target_vec,                               # (L, 6)
            "object_pc": object_pc,                                 # (Nobj, 3)
            "lang_anno": lang_line_list                             # List[str]
        }

def collate_fn(batch):
    """自定义批量打包逻辑，规避 PyTorch 无法直接 stack 字符串列表的问题"""
    out = {}
    for key in batch[0].keys():
        if key == "lang_anno":
            continue
        if torch.is_tensor(batch[0][key]):
            out[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            out[key] = [b[key] for b in batch]

    out["lang_anno"] = [b["lang_anno"] for b in batch]
    return out

def create_dataloader(cfg, is_train: bool):
    dataset = TROFuncDataset(
        data_root=cfg.data_root,
        hand_types=cfg.hand_types,
        seed=0
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=is_train,             # 训练时打乱数据，防止过拟合
        drop_last=is_train,           # 丢弃不完整的末尾 batch
        collate_fn=collate_fn,
        pin_memory=True,              # 加速向 GPU 显存拷贝的速度
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    return dataloader
