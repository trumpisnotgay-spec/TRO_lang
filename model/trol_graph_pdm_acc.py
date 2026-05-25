#################################
# T(R,O,E) in score matching scheme
#################################
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from time import time
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer

from bps_torch.bps import bps_torch
from model.vqvae.vq_vae import VQVAE
from utils.rotation import *
from model.tro_denoiser import TROLGraphDenoiser
from utils.tro_hand_model import create_hand_model

class RobotGraph(nn.Module):

    def __init__(
        self,
        vqvae_cfg,
        vqvae_pretrain,
        object_patch,
        max_link_node,
        robot_links,
        inference_config,
        bps_config,
        N_t_training,
        diffusion_config,
        denoiser_config,
        embodiment,
        loss_config,
        lang_encoder_path,
        mode="train"
    ):

        super(RobotGraph, self).__init__()
        self.mode = mode
        # vqvae encoder
        self.vqvae = VQVAE(vqvae_cfg)
        if vqvae_pretrain is not None:
            state_dict = torch.load(vqvae_pretrain, map_location='cpu', weights_only=False)
            self.vqvae.load_state_dict(state_dict)
            print(f"Loaded pretrained VQVAE from {vqvae_pretrain}.")
        # vqvae fixed
        self.vqvae.requires_grad_(False)


        # language encoder
        self.language_encoder = SentenceTransformer(lang_encoder_path)
        self.language_encoder.requires_grad_(False)

        # meta
        self.embodiment = embodiment
        self.hand_dict = {}
        for hand_name in self.embodiment:
            self.hand_dict[hand_name] = create_hand_model(hand_name)

        self.object_patch = object_patch
        self.max_link_node = max_link_node
        
        # link embedding
        self.robot_links = robot_links
        self.link_embed_dim = bps_config.n_bps_points + 4   # link bps, centroid, scale
        self.bps = bps_torch(**bps_config)
        self.link_token_encoder = nn.Sequential(
            nn.Linear(self.link_embed_dim, self.link_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.link_embed_dim, self.link_embed_dim)
        )

        self.N_t_training = N_t_training
        self.init_diffusion(diffusion_config)
        self.denoiser = TROLGraphDenoiser(
            object_patch=self.object_patch,
            max_link_node=self.max_link_node,
            **OmegaConf.to_container(denoiser_config, resolve=True)
        )
  
        self.mode = mode
        self.link_embeddings = self.construct_bps()
        if self.mode == "train":
            self.loss_config = loss_config
        elif self.mode == "test":
            inference_config = OmegaConf.to_container(inference_config, resolve=True)
            self.inference_step = inference_config["inference_step"]

    def construct_bps(self):
        
        link_embedding_dict = {}
        for embodiment, hand_model in self.hand_dict.items():
            links_vertices = hand_model.vertices
            embodiment_bps = []
            for link_name, vertices in links_vertices.items():
                vertices = torch.from_numpy(vertices).to(torch.float32)
                centroid, scale = self._unit_ball_(vertices)
                vertices = (vertices - centroid) / scale
                link_bps = self.bps.encode(
                    vertices,
                    feature_type=['dists'],
                    x_features=None,
                    custom_basis=None
                )['dists']
                device = link_bps.device
                link_bps = torch.cat([
                    link_bps, 
                    centroid.to(device=device), 
                    scale.view(1, 1).to(device=device)
                ], dim=-1)
                embodiment_bps.append(link_bps)
            link_embedding_dict[embodiment] = torch.cat(embodiment_bps, dim=0)
        return link_embedding_dict
            
    def _unit_ball_(self, pc):

        centroid = torch.mean(pc, dim=0, keepdim=True)
        pc = pc - centroid
        max_radius = pc.norm(dim=-1).max()
        return centroid, max_radius

    def _normalize_pc_(self, pc):

        # recenter
        B, N, _ = pc.shape
        centroids = torch.mean(pc, dim=1, keepdim=True)
        pc = pc - centroids

        scale, _ = torch.max(torch.abs(pc), dim=1, keepdim=True)
        scale, _ = torch.max(scale, dim=2, keepdim=True)
        pc = pc / scale

        return pc, centroids, scale

    def _normalize_pc_by_scale(self, pc, centroids, scale):
        pc = pc - centroids
        pc = pc / scale
        
        return pc, centroids, scale

    def init_diffusion(self, cfg):

        self.M = cfg["M"]
        self.scheduling = cfg["scheduling"]
        if self.scheduling == "linear":
            self.beta_min, self.beta_max = cfg["beta_min"], cfg["beta_max"]
            betas = torch.linspace(self.beta_min, self.beta_max, self.M)
        else:
            raise NotImplementedError()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer(
            "alpha_bars",
            torch.tensor([torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]),
        )
        self.ddim_steps = cfg["ddim_steps"]
        self.eta = cfg["ddim_eta"]
        self.noise_lambda = cfg["lambda"]

    def expand_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return (
            x[:, None, :, :]
            .expand(-1, self.N_t_training, -1, -1)
            .reshape(B * self.N_t_training, N, -1)
        )

    def _expand_and_reshape_(self, x, name):

        shape = x.shape
        B = x.shape[0]
        if len(shape) == 3:  # Node
            return (
                x[:, None, :, :]
                .expand(-1, self.N_t_training, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2])
            )
        elif len(shape) == 4:  # Edge
            return (
                x[:, None, :, :, :]
                .expand(-1, self.N_t_training, -1, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2], shape[3])
            )
        elif len(shape) == 2:  # Node Mask
            return (
                x[:, None, :]
                .expand(-1, self.N_t_training, -1)
                .reshape(B * self.N_t_training, shape[1])
            )
        else:
            raise ValueError(f"Unsupported shape for {name}: {shape}")

    def forward(self, batch, eps=1e-12):

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        ## Graph Construction

        # Object Node
        with torch.no_grad():
            normal_pc, centroids, scale = self._normalize_pc_(object_pc)
            object_tokens = self.vqvae.encode(normal_pc) 
            
        object_nodes = torch.cat([
            object_tokens["xyz"],
            scale.expand(-1, self.object_patch, -1),
            object_tokens["z_q"]
        ], dim=-1)  # [B, P, 3+1+64]

        # language node
        language_nodes = []
        for anno in batch['lang_anno']:
            language_nodes.append(self.language_encoder.encode(anno))
        language_nodes = torch.from_numpy(np.stack(language_nodes, axis=0)).to(device)

        # Target Link Node
        target_vec = batch["target_vec"]
        link_target_poses = torch.zeros(
            [B, self.max_link_node, 6], device=device, dtype=dtype
        )
        link_robot_embeds = torch.zeros(
            [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
        )
        link_node_masks = torch.zeros(
            [B, self.max_link_node], device=device, dtype=torch.bool
        )
        
        robot_name = self.embodiment[int(batch["robot_id"][0].item())]
        num_link = self.robot_links[robot_name]
        target_trans = target_vec[:, :, :3]
        target_rot = target_vec[:, :, 3:]

        target_trans = (target_trans - centroids) / scale
        link_target_poses = torch.cat([target_trans, target_rot], dim=-1)

        link_bps = self.link_embeddings[robot_name].to(device=device, dtype=dtype)
        link_embed = self.link_token_encoder(link_bps)
        link_robot_embeds[:, :num_link, :] = link_embed.unsqueeze(0).expand(B, -1, -1)
        link_node_masks[:, :num_link] = True

        robot_nodes = torch.cat([
            link_target_poses, 
            link_robot_embeds
        ], dim=-1)  # [B, L, 6+64]

        ## Forward Process
        t = np.random.randint(0, self.M, (B * self.N_t_training)) 
        V_O = self._expand_and_reshape_(object_nodes, "V_O")             # [B*T, P, 68]
        V_R = self._expand_and_reshape_(robot_nodes, "V_R")              # [B*T, R, 70]
        V_L = self._expand_and_reshape_(language_nodes, "V_L")           # [B*T, L, 384]
        V_R_trans, V_R_rot, V_R_embed = V_R[:, :, :3], V_R[:, :, 3:6], V_R[:, :, 6:]  

        eta_V_R_trans = torch.randn_like(V_R_trans)
        eta_V_R_rot = torch.randn_like(V_R_rot)
        a_bar = self.alpha_bars[t][:, None, None]
        
        noisy_trans = a_bar.sqrt() * V_R_trans + (1 - a_bar).sqrt() * eta_V_R_trans
        noisy_rot = a_bar.sqrt() * V_R_rot + (1 - a_bar).sqrt() * eta_V_R_rot
        noisy_V_R = torch.cat([noisy_trans, noisy_rot, V_R_embed], dim=-1)

        # update graph edges
        noisy_V_R_pose = noisy_V_R[:, :, :6]
        noisy_E_RR = relative_pose_6d(noisy_V_R_pose, noisy_V_R_pose)

        object_positions = V_O[:, :, :3]
        B, P, _ = object_positions.shape
        object_rots = torch.zeros((B, P, 3), device=device, dtype=object_positions.dtype)
        object_6D_pose = torch.cat([object_positions, object_rots], dim=-1)
        noisy_E_OR = relative_pose_6d(noisy_V_R_pose, object_6D_pose)

        ## Backward Denoising  
        pred_noise = self.denoiser(
            V_O,
            V_L,
            noisy_V_R,
            noisy_E_OR,
            noisy_E_RR,
            t
        )
        
        M_V_R = self._expand_and_reshape_(link_node_masks, "M_V_R").float()           # [B*T, L]
        pred_trans_noise = pred_noise[:, :, :3]
        pred_rot_noise = pred_noise[:, :, 3:]

        error_trans_noise = (eta_V_R_trans - pred_trans_noise) ** 2
        error_trans_noise = error_trans_noise.mean(dim=-1)
        loss_trans_noise = (error_trans_noise * M_V_R).sum() / (M_V_R.sum() + eps)

        error_rot_noise = (eta_V_R_rot - pred_rot_noise) ** 2
        error_rot_noise = error_rot_noise.mean(dim=-1)
        loss_rot_noise = (error_rot_noise * M_V_R).sum() / (M_V_R.sum() + eps)

        total_loss = self.loss_config["trans_weight"] * loss_trans_noise + self.loss_config["rot_weight"] * loss_rot_noise
        loss_dict = {
            "loss_rot": loss_rot_noise,
            "loss_trans": loss_trans_noise,
            "loss_total": total_loss
        }
        return loss_dict

    def inference(self, batch, eps=1e-8):

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype
        robot_name = self.embodiment[int(batch["robot_id"][0].item())]
        num_link = self.robot_links[robot_name]
        link_names = self.hand_dict[robot_name].link_names

        # Object Node
        with torch.no_grad():
            normal_pc, centroids, scale = self._normalize_pc_(object_pc)
            object_tokens = self.vqvae.encode(normal_pc)
            
        object_nodes = torch.cat([
            object_tokens["xyz"],
            scale.expand(-1, self.object_patch, -1),
            object_tokens["z_q"]
        ], dim=-1)  # [B, P, 3+1+64]

        # language node
        language_nodes = []
        for anno in batch['lang_anno']:
            language_nodes.append(self.language_encoder.encode(anno))
        language_nodes = torch.from_numpy(np.stack(language_nodes, axis=0)).to(device)

        ## Start from complete noise of link node (pose)
        noisy_V_R_trans = torch.randn(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        noisy_V_R_rot = torch.randn(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        link_robot_embeds = torch.zeros(
            [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
        )

        link_bps = self.link_embeddings[robot_name].to(device=device, dtype=dtype)
        link_embed = self.link_token_encoder(link_bps)
        link_robot_embeds[:, :num_link, :] = link_embed.unsqueeze(0).expand(B, -1, -1)
        noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1)

        ## Formulate edges
        noisy_V_R_pose = noisy_V_R[:, :, :6]
        noisy_E_RR = relative_pose_6d(noisy_V_R_pose, noisy_V_R_pose)

        object_positions = object_nodes[:, :, :3]
        B, P, _ = object_positions.shape
        object_rots = torch.zeros((B, P, 3), device=device, dtype=object_positions.dtype)
        object_6D_pose = torch.cat([object_positions, object_rots], dim=-1)
        noisy_E_OR = relative_pose_6d(noisy_V_R_pose, object_6D_pose)

        ## Reverse DDPM
        step = self.M // self.inference_step
        ddim_t = torch.arange(self.M - 1, -1, -step, device=device, dtype=torch.long)

        for i, diffuse_step in enumerate(ddim_t):
            
            diffuse_step = diffuse_step.item()
            # predict score
            pred_noise = self.denoiser(
                object_nodes,
                language_nodes,
                noisy_V_R,
                noisy_E_OR,
                noisy_E_RR,
                t=torch.full(
                    (object_nodes.shape[0],),
                    diffuse_step,
                    dtype=torch.long,
                    device=object_nodes.device
                )
            )

            pred_trans_noise = pred_noise[:, :, :3]
            pred_rot_noise = pred_noise[:, :, 3:]

            # predict x_0
            a_bar_t = self.alpha_bars[diffuse_step]
            if i == len(ddim_t) - 1:
                a_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)
            else:
                a_bar_prev = self.alpha_bars[ddim_t[i + 1]]
            
            x_t_trans = noisy_V_R_trans
            x_t_rot = noisy_V_R_rot
            x_0_trans = (x_t_trans - (1 - a_bar_t).sqrt() * pred_trans_noise) / a_bar_t.sqrt()
            x_0_rot = (x_t_rot - (1 - a_bar_t).sqrt() * pred_rot_noise) / a_bar_t.sqrt()

            sigma_t = self.eta * torch.sqrt(((1 - a_bar_prev) / (1 - a_bar_t)) * (1 - a_bar_t / a_bar_prev))            
            ddim_coeffient = torch.sqrt(1 - a_bar_prev - sigma_t ** 2)

            if i == len(ddim_t) - 1:
                z_trans = torch.zeros_like(x_0_trans)
                z_rot = torch.zeros_like(x_0_rot)
            else:
                z_trans = torch.randn_like(x_0_trans)
                z_rot = torch.randn_like(x_0_rot)

            x_prev_trans = a_bar_prev.sqrt() * x_0_trans + ddim_coeffient * pred_trans_noise + sigma_t * z_trans * self.noise_lambda
            x_prev_rot = a_bar_prev.sqrt() * x_0_rot + ddim_coeffient * pred_rot_noise + sigma_t * z_rot * self.noise_lambda

            noisy_V_R_trans = x_prev_trans
            noisy_V_R_rot = x_prev_rot
 
            # update node and edge
            noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1)
            noisy_V_R_pose = noisy_V_R[:, :, :6]
            noisy_E_OR = relative_pose_6d(noisy_V_R_pose, object_6D_pose)
            noisy_E_RR = relative_pose_6d(noisy_V_R_pose, noisy_V_R_pose)
       
            # export
            if i == len(ddim_t) - 1:
                pred_trans = noisy_V_R_trans * scale + centroids
                pred_rot = noisy_V_R_rot
                pred_pose = torch.cat([pred_trans, pred_rot], dim=-1)

                predict_link_pose_dict = {}
                predict_link_pose = vector_to_matrix(pred_pose[:, :num_link])
                for link_id, link_name in enumerate(link_names):
                    predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]

                return predict_link_pose_dict
