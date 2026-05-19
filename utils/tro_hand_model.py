import os
import sys
import json
import math
import random
import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.func_utils import farthest_point_sampling
from utils.mesh_utils import load_link_geometries
from utils.rotation import *


class HandModel:
    def __init__(
        self,
        robot_name,
        urdf_path,
        device,
        link_num_points=256
    ):
        self.robot_name = robot_name
        self.urdf_path = urdf_path
        self.device = device

        self.pk_chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=device)
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        self.links_pc = None
  
        self.meshes = load_link_geometries(robot_name, self.urdf_path, self.pk_chain.get_link_names())

        self.vertices = {}
        for link_name, link_mesh in self.meshes.items():
            if '_collision_ball_' in link_name:
                continue
            v = link_mesh.sample(link_num_points)
            self.vertices[link_name] = v
        self.link_names = list(self.vertices.keys())
        self.frame_status = None

    def get_joint_orders(self):
        return [joint.name for joint in self.pk_chain.get_joints()]

    def update_status(self, q):
        if q.shape[-1] != self.dof:
            q = q_rot6d_to_q_euler(q)
        self.frame_status = self.pk_chain.forward_kinematics(q.to(self.device))

    def get_link_se3(self, q=None):
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        self.update_status(q)
        all_link_se3 = []
        for link_name in self.link_names:
            se3 = self.frame_status[link_name].get_matrix()[0].to(q.device)
            all_link_se3.append(se3)
        all_link_se3 = torch.stack(all_link_se3, dim=0)
        return all_link_se3

    def get_canonical_q(self):
        """ For visualization purposes only. """
        lower, upper = self.pk_chain.get_joint_limits()
        canonical_q = torch.tensor(lower) * 0.75 + torch.tensor(upper) * 0.25
        canonical_q[:6] = 0
        return canonical_q

    def get_initial_q(self, q=None, max_angle: float = math.pi / 6):
        """
        Compute the robot initial joint value q based on the target grasp.
        Root translation is not considered since the point cloud will be normalized to zero-mean.

        :param q: (6 + DOF,) or (9 + DOF,), joint values (euler/rot6d representation)
        :param max_angle: float, maximum angle of the random rotation
        :return: initial q: (6 + DOF,), euler representation
        """
        if q is None:  # random sample root rotation and joint values
            q_initial = torch.zeros(self.dof, dtype=torch.float32, device=self.device)

            q_initial[3:6] = (torch.rand(3) * 2 - 1) * torch.pi
            q_initial[5] /= 2

            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            portion = random.uniform(0.65, 0.85)
            q_initial[6:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
        else:
            if len(q) == self.dof:
                q = q_euler_to_q_rot6d(q)
            q_initial = q.clone()

            # compute random initial rotation
            direction = - q_initial[:3] / torch.norm(q_initial[:3])
            angle = torch.tensor(random.uniform(0, max_angle), device=q.device)  # sample rotation angle
            axis = torch.randn(3).to(q.device)  # sample rotation axis
            axis -= torch.dot(axis, direction) * direction  # ensure orthogonality
            axis = axis / torch.norm(axis)
            random_rotation = axisangle_to_matrix(axis, angle).to(q.device)
            rotation_matrix = random_rotation @ rot6d_to_matrix(q_initial[3:9])
            q_initial[3:9] = matrix_to_rot6d(rotation_matrix)

            # compute random initial joint values
            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            portion = random.uniform(0.65, 0.85)
            q_initial[9:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
            # q_initial[9:] = torch.zeros_like(q_initial[9:], dtype=q.dtype, device=q.device)

            q_initial = q_rot6d_to_q_euler(q_initial)

        return q_initial

    def get_trimesh_q(self, q):
        """ Return the hand trimesh object corresponding to the input joint value q. """
        self.update_status(q)

        scene = trimesh.Scene()
        for link_name in self.vertices:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        parts = {}
        for link_name in self.meshes:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            part_mesh = self.meshes[link_name].copy().apply_transform(mesh_transform_matrix)
            parts[link_name] = part_mesh

        return_dict = {
            'visual': trimesh.Trimesh(vertices=all_vertices, faces=all_faces),
            'parts': parts
        }
        return return_dict

    def get_trimesh_se3(self, transform, index):
        """ Return the hand trimesh object corresponding to the input transform. """

        scene = trimesh.Scene()
        for link_name in transform:
            mesh_transform_matrix = transform[link_name][index].detach().cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        return trimesh.Trimesh(vertices=all_vertices, faces=all_faces)


def create_hand_model(
    robot_name,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_points=256
):
    json_path = os.path.join(ROOT_DIR, 'data/data_urdf/robot/urdf_assets_meta.json')
    urdf_assets_meta = json.load(open(json_path))
    urdf_path = os.path.join(ROOT_DIR, urdf_assets_meta['urdf_path'][robot_name])
    hand_model = HandModel(robot_name, urdf_path, device, num_points)
    return hand_model
