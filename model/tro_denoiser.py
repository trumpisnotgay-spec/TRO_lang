# V5: from v2, remove the v_mask, every node is valid
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NaiveMLP(torch.nn.Module):
    def __init__(
        self, 
        z_dim, 
        out_dim, 
        hidden_dims,
        output_activation=None
    ):
        super().__init__()
        self.output_activation = output_activation

        c_wide_out_cnt = z_dim
        self.layers = nn.ModuleList()
        c_in = z_dim
        for i, c_out in enumerate(hidden_dims):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(c_in, c_out),
                    nn.LayerNorm(c_out),
                    nn.LeakyReLU(),
                )
            )
            c_wide_out_cnt += c_out
            c_in = c_out
        self.out_fc0 = nn.Linear(c_wide_out_cnt, out_dim)

        if output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif output_activation is None:
            self.out_act = None
        else:
            raise ValueError("Unsupported output_activation")
        return

    def forward(self, x):
        f_list = [x]
        for l in self.layers:
            x = l(x)
            f_list.append(x)
        f = torch.cat(f_list, -1)
        y = self.out_fc0(f)

        if self.out_act is not None:
            y = self.out_act(y)
        return y

class EdgeValueNet(nn.Module):

    def __init__(self, Do: int, Dr: int, De: int, Dout: int):
        super().__init__()
        self.Wo = nn.Linear(Do, Dout, bias=False)
        self.Wr = nn.Linear(Dr, Dout, bias=False)
        self.We = nn.Linear(De, Dout, bias=True)
        self.W2 = nn.Linear(Dout, Dout, bias=True)

    def forward(self, o, r, e):
        x = self.Wo(o) + self.Wr(r) + self.We(e)
        x = F.silu(x)
        return self.W2(x)
    
class BindNet(nn.Module):
    def __init__(self, d_x: int, d_t: int):
        super().__init__()
        self.Wx = nn.Linear(d_x, d_x, bias=True)
        self.Wt = nn.Linear(d_t, d_x, bias=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return F.silu(self.Wx(x) + self.Wt(t).view(t.shape[0], *([1] * (x.ndim - 2)), -1))
    
class EdgeUpdateNet(nn.Module):
    def __init__(self, d_edge: int):
        super().__init__()
        self.Wv = nn.Linear(d_edge, d_edge, bias=True)
        self.Wp = nn.Linear(d_edge, d_edge, bias=False)
        self.W2 = nn.Linear(d_edge, d_edge, bias=True)

    def forward(self, value: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        x = self.Wv(value) + self.Wp(pool)
        x = F.silu(x)
        return self.W2(x)


class TROGraphLayer(nn.Module):

    def __init__(
        self,
        v_object_dim,
        v_robot_dim,
        t_embed_dim,
        e_or_dim,
        e_rr_dim,
        c_atten_head,
        update_edge=True
    ) -> None:
        super().__init__()

        self.c_atten_head = c_atten_head
        self.update_edge = update_edge

        # head dims
        assert e_or_dim % c_atten_head == 0
        assert e_rr_dim % c_atten_head == 0
        self.or_head_dim = e_or_dim // c_atten_head
        self.rr_head_dim = e_rr_dim // c_atten_head

        # =========================
        # PE binding
        # =========================
        self.v_object_binding_fc = BindNet(v_object_dim, t_embed_dim)
        self.v_robot_binding_fc = BindNet(v_robot_dim, t_embed_dim)
        self.e_or_binding_fc = BindNet(e_or_dim, t_embed_dim)
        self.e_rr_binding_fc = BindNet(e_rr_dim, t_embed_dim)

        # =========================
        # OR attention
        # =========================
        self.or_query_fc = nn.Linear(v_robot_dim, e_or_dim)
        self.or_key_fc = nn.Linear(v_object_dim, e_or_dim)
        self.or_value_fc = EdgeValueNet(v_object_dim, v_robot_dim, e_or_dim, e_or_dim)
        self.or_r_self = nn.Sequential(nn.Linear(v_robot_dim, e_or_dim), nn.SiLU())
        
        # =========================
        # RR attention
        # =========================
        self.rr_query_fc = nn.Linear(v_robot_dim, e_rr_dim)
        self.rr_key_fc = nn.Linear(v_robot_dim, e_rr_dim)
        self.rr_value_fc = EdgeValueNet(v_robot_dim,  v_robot_dim, e_rr_dim, e_rr_dim)
        self.rr_r_self = nn.Sequential(nn.Linear(v_robot_dim, e_rr_dim), nn.SiLU())

        # =========================
        # Edge Update Net
        # =========================
        if self.update_edge:
            self.or_edge_out = EdgeUpdateNet(e_or_dim)
            self.rr_edge_out = EdgeUpdateNet(e_rr_dim)

    def forward(
        self,
        object_node_f,  # [B, P, v_object_dim]
        robot_node_f,   # [B, L, v_robot_dim]
        or_edge_f,      # [B, L, P, e_or_dim]
        rr_edge_f,      # [B, L, L, e_rr_dim]
        t_embed,        # [B, t_embed_dim]
    ):
        B, P, _ = object_node_f.shape
        _, L, _ = robot_node_f.shape
        H = self.c_atten_head
        Dh_or = self.or_head_dim
        Dh_rr = self.rr_head_dim

        # ========
        # Binding
        # ========
        object_node_f = self.v_object_binding_fc(object_node_f, t_embed)
        robot_node_f = self.v_robot_binding_fc(robot_node_f, t_embed)

        or_edge_f = self.e_or_binding_fc(or_edge_f, t_embed)
        rr_edge_f = self.e_rr_binding_fc(rr_edge_f, t_embed)

        # ============================================================
        # OR attention (robot attends to object)
        # ============================================================
        # Q: [B, L, H, Dh_or]
        Q_or = self.or_query_fc(robot_node_f).view(B, L, H, Dh_or)
        # K: [B, P, H, Dh_or]
        K_or = self.or_key_fc(object_node_f).view(B, P, H, Dh_or)

        # edge value: [B, L, P, e_or_dim]
        o_b = object_node_f[:, None, :, :]        # [B, 1, P, Do]
        r_b = robot_node_f[:, :, None, :]         # [B, L, 1, Dr]
        or_value = self.or_value_fc(o_b, r_b, or_edge_f)  # [B, L, P, Eo]
        
        Q_ = Q_or.reshape(B * L, H, 1, Dh_or)
        K_ = (
            K_or.permute(0, 2, 1, 3)
            .unsqueeze(1)
            .expand(B, L, H, P, Dh_or)
            .reshape(B * L, H, P, Dh_or)
        )
        V_ = (
            or_value.view(B, L, P, H, Dh_or)
            .permute(0, 1, 3, 2, 4)
            .reshape(B * L, H, P, Dh_or)
        )

        out = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)
        agg_or = out.squeeze(2).reshape(B, L, H * Dh_or)      # [B, L, e_or_dim]
        robot_node_f = agg_or + self.or_r_self(robot_node_f)  # [B, L, e_or_dim]

        # ============================================================
        # RR attention (robot attends to robot)
        # ============================================================
        Q_rr = self.rr_query_fc(robot_node_f).view(B, L, H, Dh_rr)  # [B, L, H, Dh_rr]
        K_rr = self.rr_key_fc(robot_node_f).view(B, L, H, Dh_rr)    # [B, L, H, Dh_rr]

        self_b = robot_node_f[:, None, :, :]      # [B, 1, L, Dr]
        nei_b  = robot_node_f[:, :, None, :]      # [B, L, 1, Dr]
        rr_value = self.rr_value_fc(self_b, nei_b, rr_edge_f)  # [B, L, L, Er]

        Q_ = Q_rr.reshape(B * L, H, 1, Dh_rr)
        K_ = (
            K_rr.permute(0, 2, 1, 3)
            .unsqueeze(1)
            .expand(B, L, H, L, Dh_rr)
            .reshape(B * L, H, L, Dh_rr)
        )
        V_ = (
            rr_value.view(B, L, L, H, Dh_rr)
            .permute(0, 1, 3, 2, 4)
            .reshape(B * L, H, L, Dh_rr)
        )

        out = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)
        agg_rr = out.squeeze(2).reshape(B, L, H * Dh_rr)      # [B, L, e_rr_dim]
        robot_node_f = agg_rr + self.rr_r_self(robot_node_f)  # [B, L, e_rr_dim]

        # Edge Update
        if self.update_edge:

            or_edge_f_pool = 0.5 * (or_edge_f.mean(1, keepdim=True) + or_edge_f.mean(2, keepdim=True))
            or_edge_f = self.or_edge_out(or_value, or_edge_f_pool)  # [B, L, P, e_or_dim]

            rr_edge_f_pool = 0.5 * (rr_edge_f.mean(1, keepdim=True) + rr_edge_f.mean(2, keepdim=True))
            rr_edge_f = self.rr_edge_out(rr_value, rr_edge_f_pool)  # [B, L, L, e_rr_dim]

            return robot_node_f, or_edge_f, rr_edge_f
    
        else:
            return robot_node_f, None, None


class TROGraphDenoiser(torch.nn.Module):

    def __init__(
        self,
        # Initial Config
        M: int = 1000,
        object_patch: int = 25,
        max_link_node: int = 25,
        # Embedding
        t_embed_dim: int = 200,
        # Input Encoder
        V_object_dims: list = [3, 1, 64],
        V_robot_dims: list = [3, 3, 64],
        E_or_dims: list = [3, 3],
        E_rr_dims: list = [3, 3],
        # Backbone
        v_conv_dim=384,
        e_conv_dim=384,
        c_atten_head=32,
        num_layers=6,
        v_out_hidden_dims=[256, 128],
        out_dim=[3, 3]
    ) -> None:
        super().__init__()

        self.M = M
        self.object_patch = object_patch
        self.max_link_node = max_link_node

        # Position Embedding
        self.t_embed_dim = t_embed_dim
        self.register_buffer("t_embed", self.sinusoidal_embedding(self.M, self.t_embed_dim))

        # Node Embedding
        self.V_object_dims = V_object_dims
        self.V_robot_dims = V_robot_dims
        self.V_object_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, v_conv_dim) for in_dim in self.V_object_dims]
        )
        self.V_robot_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, v_conv_dim) for in_dim in self.V_robot_dims]
        )

        # Edge Embedding
        self.E_or_dims = E_or_dims
        self.E_rr_dims = E_rr_dims
        self.E_or_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, e_conv_dim) for in_dim in self.E_or_dims]
        )
        self.E_rr_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, e_conv_dim) for in_dim in self.E_rr_dims]
        )

        # Backbone
        self.v_conv_dim = v_conv_dim
        self.e_conv_dim = e_conv_dim
        self.c_atten_head = c_atten_head
        self.num_layers = num_layers
        
        self.graph_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.graph_layers.append(
                TROGraphLayer(
                    v_object_dim=v_conv_dim,
                    v_robot_dim=v_conv_dim,
                    t_embed_dim=t_embed_dim,
                    e_or_dim=e_conv_dim,
                    e_rr_dim=e_conv_dim,
                    c_atten_head=c_atten_head,
                    update_edge=(i < self.num_layers - 1)
                )
            )
        self.v_robot_wide_fc = nn.Linear(v_conv_dim * (1 + num_layers), v_conv_dim)

        # Output layers
        self.out_dim = out_dim
        self.v_robot_output_module = nn.ModuleList()

        for _, f_dim in enumerate(self.out_dim):
            self.v_robot_output_module.append(
                NaiveMLP(
                    f_dim + self.v_conv_dim, 
                    f_dim,
                    v_out_hidden_dims
                )
            )

    def sinusoidal_embedding(self, n, d):
        # Returns the standard positional embedding
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
        wk = wk.reshape((1, d))
        t = torch.arange(n).reshape((n, 1))
        embedding[:, ::2] = torch.sin(t * wk[:, ::2])
        embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
        return embedding

    def _encoder_(self, x, in_dims, in_layers):
        # Encode node and edge
        assert len(in_dims) == len(in_layers)
        cur = 0
        feat = None
        for i, d in enumerate(in_dims):
            partial = x[..., cur : cur + d]
            out = in_layers[i](partial)
            if feat is None:
                feat = out
            else:
                feat = feat + out
            cur += d
        return feat

    def forward(
        self, 
        V_O, 
        noisy_V_R,
        noisy_E_OR,
        noisy_E_RR,
        t,
    ):

        # Position Embedding
        B = t.shape[0]
        t_embed = self.t_embed[t]  # [B, 200]

        # Initial Encoding
        object_node_f = self._encoder_(
            V_O,
            self.V_object_dims,
            self.V_object_in_layers
        )  # [B, P, F]
        robot_node_f = self._encoder_(
            noisy_V_R,
            self.V_robot_dims,
            self.V_robot_in_layers
        )  # [B, L, F]


        or_edge_f = self._encoder_(
            noisy_E_OR,
            self.E_or_dims,
            self.E_or_in_layers
        )  # [B, L, P, F]
        rr_edge_f = self._encoder_(
            noisy_E_RR,
            self.E_rr_dims,
            self.E_rr_in_layers
        )  # [B, L, L, F]

        noisy_robot_node_f_list = [robot_node_f]

        for layer_id, layer in enumerate(self.graph_layers):
            robot_node_f, or_edge_f, rr_edge_f = layer(
                object_node_f,
                robot_node_f,
                or_edge_f,
                rr_edge_f,
                t_embed
            )
            noisy_robot_node_f_list.append(robot_node_f)

        update_robot_node_f = self.v_robot_wide_fc(
            torch.cat(noisy_robot_node_f_list, dim=-1)
        )  # [B, L, F]

        # Output
        v_robot_pred = []
        cur = 0
        for layer_id, f_dim in enumerate(self.out_dim):
            v_robot_pred.append(
                self.v_robot_output_module[layer_id](
                    torch.cat([update_robot_node_f, noisy_V_R[..., cur : cur + f_dim]], dim=-1)
                )
            )
            cur += f_dim
        v_robot_pred = torch.cat(v_robot_pred, dim=-1)
        return v_robot_pred


class TROEGraphLayer(nn.Module):

    def __init__(
        self,
        v_object_dim,
        v_robot_dim,
        v_env_dim,
        t_embed_dim,
        e_or_dim,
        e_rr_dim,
        e_er_dim,
        c_atten_head,
        update_edge=True
    ) -> None:
        super().__init__()

        self.c_atten_head = c_atten_head
        self.update_edge = update_edge

        # head dims
        assert e_or_dim % c_atten_head == 0
        assert e_rr_dim % c_atten_head == 0
        assert e_er_dim % c_atten_head == 0
        self.or_head_dim = e_or_dim // c_atten_head
        self.rr_head_dim = e_rr_dim // c_atten_head
        self.er_head_dim = e_er_dim // c_atten_head

        # =========================
        # PE binding
        # =========================
        self.v_object_binding_fc = BindNet(v_object_dim, t_embed_dim)
        self.v_robot_binding_fc = BindNet(v_robot_dim, t_embed_dim)
        self.v_env_binding_fc = BindNet(v_env_dim, t_embed_dim)
        self.e_or_binding_fc = BindNet(e_or_dim, t_embed_dim)
        self.e_rr_binding_fc = BindNet(e_rr_dim, t_embed_dim)
        self.e_er_binding_fc = BindNet(e_er_dim, t_embed_dim)

        # =========================
        # OR attention
        # =========================
        self.or_query_fc = nn.Linear(v_robot_dim, e_or_dim)
        self.or_key_fc = nn.Linear(v_object_dim, e_or_dim)
        self.or_value_fc = EdgeValueNet(v_object_dim, v_robot_dim, e_or_dim, e_or_dim)
        self.or_r_self = nn.Sequential(nn.Linear(v_robot_dim, e_or_dim), nn.SiLU())

        # =========================
        # ER attention (env -> robot)
        # =========================
        self.er_query_fc = nn.Linear(v_robot_dim, e_er_dim)
        self.er_key_fc = nn.Linear(v_env_dim, e_er_dim)
        self.er_value_fc = EdgeValueNet(v_env_dim, v_robot_dim, e_er_dim, e_er_dim)
        self.er_r_self = nn.Sequential(nn.Linear(v_robot_dim, e_er_dim), nn.SiLU())
        
        # =========================
        # RR attention
        # =========================
        self.rr_query_fc = nn.Linear(v_robot_dim, e_rr_dim)
        self.rr_key_fc = nn.Linear(v_robot_dim, e_rr_dim)
        self.rr_value_fc = EdgeValueNet(v_robot_dim,  v_robot_dim, e_rr_dim, e_rr_dim)
        self.rr_r_self = nn.Sequential(nn.Linear(v_robot_dim, e_rr_dim), nn.SiLU())

        # =========================
        # Edge Update Net
        # =========================
        if self.update_edge:
            self.or_edge_out = EdgeUpdateNet(e_or_dim)
            self.er_edge_out = EdgeUpdateNet(e_er_dim)
            self.rr_edge_out = EdgeUpdateNet(e_rr_dim)

    def forward(
        self,
        object_node_f,  # [B, P, v_object_dim]
        robot_node_f,   # [B, L, v_robot_dim]
        env_node_f,     # [B, E, v_env_dim]
        or_edge_f,      # [B, L, P, e_or_dim]
        rr_edge_f,      # [B, L, L, e_rr_dim]
        er_edge_f,      # [B, L, E, e_er_dim]
        t_embed,        # [B, t_embed_dim]
    ):
        B, P, _ = object_node_f.shape
        _, L, _ = robot_node_f.shape
        _, E, _ = env_node_f.shape
        H = self.c_atten_head
        Dh_or = self.or_head_dim
        Dh_er = self.er_head_dim
        Dh_rr = self.rr_head_dim

        # ========
        # Binding
        # ========
        object_node_f = self.v_object_binding_fc(object_node_f, t_embed)
        robot_node_f = self.v_robot_binding_fc(robot_node_f, t_embed)
        env_node_f = self.v_env_binding_fc(env_node_f, t_embed)

        or_edge_f = self.e_or_binding_fc(or_edge_f, t_embed)
        rr_edge_f = self.e_rr_binding_fc(rr_edge_f, t_embed)
        er_edge_f = self.e_er_binding_fc(er_edge_f, t_embed)

        # ============================================================
        # OR attention (robot attends to object)
        # ============================================================
        # Q: [B, L, H, Dh_or]
        Q_or = self.or_query_fc(robot_node_f).view(B, L, H, Dh_or)
        # K: [B, P, H, Dh_or]
        K_or = self.or_key_fc(object_node_f).view(B, P, H, Dh_or)

        # edge value: [B, L, P, e_or_dim]
        o_b = object_node_f[:, None, :, :]        # [B, 1, P, Do]
        r_b = robot_node_f[:, :, None, :]         # [B, L, 1, Dr]
        or_value = self.or_value_fc(o_b, r_b, or_edge_f)  # [B, L, P, Eo]
        
        Q_ = Q_or.reshape(B * L, H, 1, Dh_or)
        K_ = (
            K_or.permute(0, 2, 1, 3)
            .unsqueeze(1)
            .expand(B, L, H, P, Dh_or)
            .reshape(B * L, H, P, Dh_or)
        )
        V_ = (
            or_value.view(B, L, P, H, Dh_or)
            .permute(0, 1, 3, 2, 4)
            .reshape(B * L, H, P, Dh_or)
        )

        out = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)
        agg_or = out.squeeze(2).reshape(B, L, H * Dh_or)      # [B, L, e_or_dim]
        robot_node_f = agg_or + self.or_r_self(robot_node_f)  # [B, L, e_or_dim]

        # ============================================================
        # ER attention (robot attends to env)
        # ============================================================
        Q_er = self.er_query_fc(robot_node_f).view(B, L, H, Dh_er)  # [B, L, H, Dh_er]
        K_er = self.er_key_fc(env_node_f).view(B, E, H, Dh_er)      # [B, E, H, Dh_er]

        e_b = env_node_f[:, None, :, :]           # [B, 1, E, De]
        r_b = robot_node_f[:, :, None, :]         # [B, L, 1, Dr]
        er_value = self.er_value_fc(e_b, r_b, er_edge_f)  # [B, L, E, Ee]

        Q_ = Q_er.reshape(B * L, H, 1, Dh_er)
        K_ = (
            K_er.permute(0, 2, 1, 3)
            .unsqueeze(1)
            .expand(B, L, H, E, Dh_er)
            .reshape(B * L, H, E, Dh_er)
        )
        V_ = (
            er_value.view(B, L, E, H, Dh_er)
            .permute(0, 1, 3, 2, 4)
            .reshape(B * L, H, E, Dh_er)
        )

        out = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)
        agg_er = out.squeeze(2).reshape(B, L, H * Dh_er)      # [B, L, e_er_dim]
        robot_node_f = agg_er + self.er_r_self(robot_node_f)  # [B, L, e_er_dim]

        # ============================================================
        # RR attention (robot attends to robot)
        # ============================================================
        Q_rr = self.rr_query_fc(robot_node_f).view(B, L, H, Dh_rr)  # [B, L, H, Dh_rr]
        K_rr = self.rr_key_fc(robot_node_f).view(B, L, H, Dh_rr)    # [B, L, H, Dh_rr]

        self_b = robot_node_f[:, None, :, :]      # [B, 1, L, Dr]
        nei_b  = robot_node_f[:, :, None, :]      # [B, L, 1, Dr]
        rr_value = self.rr_value_fc(self_b, nei_b, rr_edge_f)  # [B, L, L, Er]

        Q_ = Q_rr.reshape(B * L, H, 1, Dh_rr)
        K_ = (
            K_rr.permute(0, 2, 1, 3)
            .unsqueeze(1)
            .expand(B, L, H, L, Dh_rr)
            .reshape(B * L, H, L, Dh_rr)
        )
        V_ = (
            rr_value.view(B, L, L, H, Dh_rr)
            .permute(0, 1, 3, 2, 4)
            .reshape(B * L, H, L, Dh_rr)
        )

        out = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)
        agg_rr = out.squeeze(2).reshape(B, L, H * Dh_rr)      # [B, L, e_rr_dim]
        robot_node_f = agg_rr + self.rr_r_self(robot_node_f)  # [B, L, e_rr_dim]

        # Edge Update
        if self.update_edge:

            or_edge_f_pool = 0.5 * (or_edge_f.mean(1, keepdim=True) + or_edge_f.mean(2, keepdim=True))
            or_edge_f = self.or_edge_out(or_value, or_edge_f_pool)  # [B, L, P, e_or_dim]

            er_edge_f_pool = 0.5 * (er_edge_f.mean(1, keepdim=True) + er_edge_f.mean(2, keepdim=True))
            er_edge_f = self.er_edge_out(er_value, er_edge_f_pool)  # [B, L, E, e_er_dim]
                
            rr_edge_f_pool = 0.5 * (rr_edge_f.mean(1, keepdim=True) + rr_edge_f.mean(2, keepdim=True))
            rr_edge_f = self.rr_edge_out(rr_value, rr_edge_f_pool)  # [B, L, L, e_rr_dim]

            return robot_node_f, or_edge_f, rr_edge_f, er_edge_f
    
        else:
            return robot_node_f, None, None, None


class TROEGraphDenoiser(torch.nn.Module):

    def __init__(
        self,
        # Initial Config
        M: int = 1000,
        object_patch: int = 25,
        env_patch: int = 25,
        max_link_node: int = 25,
        # Embedding
        t_embed_dim: int = 200,
        # Input Encoder
        V_object_dims: list = [3, 1, 64],
        V_robot_dims: list = [3, 3, 64],
        V_env_dims: list = [3, 1, 64],
        E_or_dims: list = [3, 3],
        E_rr_dims: list = [3, 3],
        E_er_dims: list = [3, 3],
        # Backbone
        v_conv_dim=384,
        e_conv_dim=384,
        c_atten_head=32,
        num_layers=6,
        v_out_hidden_dims=[256, 128],
        out_dim=[3, 3]
    ) -> None:
        super().__init__()

        self.M = M
        self.object_patch = object_patch
        self.env_patch = env_patch
        self.max_link_node = max_link_node

        # Position Embedding
        self.t_embed_dim = t_embed_dim
        self.register_buffer("t_embed", self.sinusoidal_embedding(self.M, self.t_embed_dim))

        # Node Embedding
        self.V_object_dims = V_object_dims
        self.V_robot_dims = V_robot_dims
        self.V_env_dims = V_env_dims
        self.V_object_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, v_conv_dim) for in_dim in self.V_object_dims]
        )
        self.V_robot_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, v_conv_dim) for in_dim in self.V_robot_dims]
        )
        self.V_env_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, v_conv_dim) for in_dim in self.V_env_dims]
        )

        # Edge Embedding
        self.E_or_dims = E_or_dims
        self.E_rr_dims = E_rr_dims
        self.E_er_dims = E_er_dims
        self.E_or_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, e_conv_dim) for in_dim in self.E_or_dims]
        )
        self.E_rr_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, e_conv_dim) for in_dim in self.E_rr_dims]
        )
        self.E_er_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, e_conv_dim) for in_dim in self.E_er_dims]
        )

        # Backbone
        self.v_conv_dim = v_conv_dim
        self.e_conv_dim = e_conv_dim
        self.c_atten_head = c_atten_head
        self.num_layers = num_layers
        
        self.graph_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.graph_layers.append(
                TROEGraphLayer(
                    v_object_dim=v_conv_dim,
                    v_robot_dim=v_conv_dim,
                    v_env_dim=v_conv_dim,
                    t_embed_dim=t_embed_dim,
                    e_or_dim=e_conv_dim,
                    e_rr_dim=e_conv_dim,
                    e_er_dim=e_conv_dim,
                    c_atten_head=c_atten_head,
                    update_edge=(i < self.num_layers - 1)
                )
            )
        self.v_robot_wide_fc = nn.Linear(v_conv_dim * (1 + num_layers), v_conv_dim)

        # Output layers
        self.out_dim = out_dim
        self.v_robot_output_module = nn.ModuleList()

        for _, f_dim in enumerate(self.out_dim):
            self.v_robot_output_module.append(
                NaiveMLP(
                    f_dim + self.v_conv_dim, 
                    f_dim,
                    v_out_hidden_dims
                )
            )

    def sinusoidal_embedding(self, n, d):
        # Returns the standard positional embedding
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
        wk = wk.reshape((1, d))
        t = torch.arange(n).reshape((n, 1))
        embedding[:, ::2] = torch.sin(t * wk[:, ::2])
        embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
        return embedding

    def _encoder_(self, x, in_dims, in_layers):
        # Encode node and edge
        assert len(in_dims) == len(in_layers)
        cur = 0
        feat = None
        for i, d in enumerate(in_dims):
            partial = x[..., cur : cur + d]
            out = in_layers[i](partial)
            if feat is None:
                feat = out
            else:
                feat = feat + out
            cur += d
        return feat
        
    def forward(
        self, 
        V_O, 
        noisy_V_R,
        V_E,
        noisy_E_OR,
        noisy_E_RR,
        noisy_E_ER,
        t,
    ):

        # Position Embedding
        B = t.shape[0]
        t_embed = self.t_embed[t]  # [B, 200]

        # Initial Encoding
        object_node_f = self._encoder_(
            V_O,
            self.V_object_dims,
            self.V_object_in_layers
        )  # [B, P, F]
        robot_node_f = self._encoder_(
            noisy_V_R,
            self.V_robot_dims,
            self.V_robot_in_layers
        )  # [B, L, F]
        env_node_f = self._encoder_(
            V_E,
            self.V_env_dims,
            self.V_env_in_layers
        )  # [B, E, F]

        or_edge_f = self._encoder_(
            noisy_E_OR,
            self.E_or_dims,
            self.E_or_in_layers
        )  # [B, L, P, F]
        rr_edge_f = self._encoder_(
            noisy_E_RR,
            self.E_rr_dims,
            self.E_rr_in_layers
        )  # [B, L, L, F]
        er_edge_f = self._encoder_(
            noisy_E_ER,
            self.E_er_dims,
            self.E_er_in_layers
        )  # [B, L, E, F]

        noisy_robot_node_f_list = [robot_node_f]

        for layer_id, layer in enumerate(self.graph_layers):
            robot_node_f, or_edge_f, rr_edge_f, er_edge_f = layer(
                object_node_f,
                robot_node_f,
                env_node_f,
                or_edge_f,
                rr_edge_f,
                er_edge_f,
                t_embed
            )
            noisy_robot_node_f_list.append(robot_node_f)

        update_robot_node_f = self.v_robot_wide_fc(
            torch.cat(noisy_robot_node_f_list, dim=-1)
        )  # [B, L, F]

        # Output
        v_robot_pred = []
        cur = 0
        for layer_id, f_dim in enumerate(self.out_dim):
            v_robot_pred.append(
                self.v_robot_output_module[layer_id](
                    torch.cat([update_robot_node_f, noisy_V_R[..., cur : cur + f_dim]], dim=-1)
                )
            )
            cur += f_dim
        v_robot_pred = torch.cat(v_robot_pred, dim=-1)
        return v_robot_pred
        
        
class TROLGraphLayer(nn.Module):

    def __init__(
        self,
        v_object_dim,
        v_robot_dim,
        v_language_dims,
        t_embed_dim,
        e_or_dim,
        e_rr_dim,
        lr_attn_dim,
        c_atten_head,
        update_edge=True
    ) -> None:
        super().__init__()

        self.c_atten_head = c_atten_head
        self.update_edge = update_edge

        # head dims
        assert e_or_dim % c_atten_head == 0
        assert e_rr_dim % c_atten_head == 0
        assert lr_attn_dim % c_atten_head == 0
        self.or_head_dim = e_or_dim // c_atten_head
        self.rr_head_dim = e_rr_dim // c_atten_head
        self.lr_head_dim = lr_attn_dim // c_atten_head

        # =========================
        # PE binding
        # =========================
        self.v_object_binding_fc = BindNet(v_object_dim, t_embed_dim)
        self.v_robot_binding_fc = BindNet(v_robot_dim, t_embed_dim)
        self.e_or_binding_fc = BindNet(e_or_dim, t_embed_dim)
        self.e_rr_binding_fc = BindNet(e_rr_dim, t_embed_dim)

        # =========================
        # OR attention
        # =========================
        self.or_query_fc = nn.Linear(v_robot_dim, e_or_dim)
        self.or_key_fc = nn.Linear(v_object_dim, e_or_dim)
        self.or_value_fc = EdgeValueNet(v_object_dim, v_robot_dim, e_or_dim, e_or_dim)
        self.or_r_self = nn.Sequential(nn.Linear(v_robot_dim, e_or_dim), nn.SiLU())
        
        # =========================
        # RR attention
        # =========================
        self.rr_query_fc = nn.Linear(v_robot_dim, e_rr_dim)
        self.rr_key_fc = nn.Linear(v_robot_dim, e_rr_dim)
        self.rr_value_fc = EdgeValueNet(v_robot_dim,  v_robot_dim, e_rr_dim, e_rr_dim)
        self.rr_r_self = nn.Sequential(nn.Linear(v_robot_dim, e_rr_dim), nn.SiLU())

        # =========================
        # LR attention
        # =========================
        self.rl_query_fc = nn.Linear(v_robot_dim, lr_attn_dim)
        self.rl_key_fc = nn.Linear(v_language_dims, lr_attn_dim)
        self.rl_value_fc = nn.Linear(v_language_dims, lr_attn_dim)
        self.rl_r_self = nn.Sequential(nn.Linear(v_robot_dim, lr_attn_dim), nn.SiLU())

        # =========================
        # Edge Update Net
        # =========================
        if self.update_edge:
            self.or_edge_out = EdgeUpdateNet(e_or_dim)
            self.rr_edge_out = EdgeUpdateNet(e_rr_dim)

    def forward(
        self,
        object_node_f,      # [B, P, v_object_dim]
        robot_node_f,       # [B, L, v_robot_dim]
        languange_node_f,   # [B, LAN, v_lan_dim]
        or_edge_f,          # [B, L, P, e_or_dim]
        rr_edge_f,          # [B, L, L, e_rr_dim]
        t_embed,            # [B, t_embed_dim]
    ):
        B, P, _ = object_node_f.shape
        _, L, _ = robot_node_f.shape
        _, LAN, _ = languange_node_f.shape
        H = self.c_atten_head
        Dh_or = self.or_head_dim
        Dh_rr = self.rr_head_dim
        Dh_lr = self.lr_head_dim

        # ========
        # Binding
        # ========
        object_node_f = self.v_object_binding_fc(object_node_f, t_embed)
        robot_node_f = self.v_robot_binding_fc(robot_node_f, t_embed)

        or_edge_f = self.e_or_binding_fc(or_edge_f, t_embed)
        rr_edge_f = self.e_rr_binding_fc(rr_edge_f, t_embed)

        # ============================================================
        # LR attention (robot attends to object)
        # ============================================================
        # Q: [B, L, H, Dh]
        Q_lr = self.rl_query_fc(robot_node_f).view(B, L, H, Dh_lr)
        # K: [B, LAN, H, Dh]
        K_lr = self.rl_key_fc(languange_node_f).view(B, LAN, H, Dh_lr)
        # V: [B, LAN, H, Dh]
        V_lr = self.rl_value_fc(languange_node_f).view(B, LAN, H, Dh_lr)   
        
        Q_ = Q_lr.reshape(B * L, H, 1, Dh_lr)
        K_ = (
            K_lr.permute(0, 2, 1, 3)
            .unsqueeze(1)
            .expand(B, L, H, LAN, Dh_lr)
            .reshape(B * L, H, LAN, Dh_lr)
        )
        V_ = (
            V_lr.permute(0, 2, 1, 3)
            .unsqueeze(1)
            .expand(B, L, H, LAN, Dh_lr)
            .reshape(B * L, H, LAN, Dh_lr)
        )

        out = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)
        agg_lr = out.squeeze(2).reshape(B, L, H * Dh_lr)      # [B, L, rl_attn_dim]
        robot_node_f = agg_lr + self.rl_r_self(robot_node_f)  # [B, L, rl_attn_dim]

        # ============================================================
        # OR attention (robot attends to object)
        # ============================================================
        # Q: [B, L, H, Dh_or]
        Q_or = self.or_query_fc(robot_node_f).view(B, L, H, Dh_or)
        # K: [B, P, H, Dh_or]
        K_or = self.or_key_fc(object_node_f).view(B, P, H, Dh_or)

        # edge value: [B, L, P, e_or_dim]
        o_b = object_node_f[:, None, :, :]        # [B, 1, P, Do]
        r_b = robot_node_f[:, :, None, :]         # [B, L, 1, Dr]
        or_value = self.or_value_fc(o_b, r_b, or_edge_f)  # [B, L, P, Eo]
        
        Q_ = Q_or.reshape(B * L, H, 1, Dh_or)
        K_ = (
            K_or.permute(0, 2, 1, 3)
            .unsqueeze(1)
            .expand(B, L, H, P, Dh_or)
            .reshape(B * L, H, P, Dh_or)
        )
        V_ = (
            or_value.view(B, L, P, H, Dh_or)
            .permute(0, 1, 3, 2, 4)
            .reshape(B * L, H, P, Dh_or)
        )

        out = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)
        agg_or = out.squeeze(2).reshape(B, L, H * Dh_or)      # [B, L, e_or_dim]
        robot_node_f = agg_or + self.or_r_self(robot_node_f)  # [B, L, e_or_dim]

        # ============================================================
        # RR attention (robot attends to robot)
        # ============================================================
        Q_rr = self.rr_query_fc(robot_node_f).view(B, L, H, Dh_rr)  # [B, L, H, Dh_rr]
        K_rr = self.rr_key_fc(robot_node_f).view(B, L, H, Dh_rr)    # [B, L, H, Dh_rr]

        self_b = robot_node_f[:, None, :, :]      # [B, 1, L, Dr]
        nei_b  = robot_node_f[:, :, None, :]      # [B, L, 1, Dr]
        rr_value = self.rr_value_fc(self_b, nei_b, rr_edge_f)  # [B, L, L, Er]

        Q_ = Q_rr.reshape(B * L, H, 1, Dh_rr)
        K_ = (
            K_rr.permute(0, 2, 1, 3)
            .unsqueeze(1)
            .expand(B, L, H, L, Dh_rr)
            .reshape(B * L, H, L, Dh_rr)
        )
        V_ = (
            rr_value.view(B, L, L, H, Dh_rr)
            .permute(0, 1, 3, 2, 4)
            .reshape(B * L, H, L, Dh_rr)
        )

        out = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)
        agg_rr = out.squeeze(2).reshape(B, L, H * Dh_rr)      # [B, L, e_rr_dim]
        robot_node_f = agg_rr + self.rr_r_self(robot_node_f)  # [B, L, e_rr_dim]

        # Edge Update
        if self.update_edge:

            or_edge_f_pool = 0.5 * (or_edge_f.mean(1, keepdim=True) + or_edge_f.mean(2, keepdim=True))
            or_edge_f = self.or_edge_out(or_value, or_edge_f_pool)  # [B, L, P, e_or_dim]

            rr_edge_f_pool = 0.5 * (rr_edge_f.mean(1, keepdim=True) + rr_edge_f.mean(2, keepdim=True))
            rr_edge_f = self.rr_edge_out(rr_value, rr_edge_f_pool)  # [B, L, L, e_rr_dim]

            return robot_node_f, or_edge_f, rr_edge_f
    
        else:
            return robot_node_f, None, None


class TROLGraphDenoiser(torch.nn.Module):

    def __init__(
        self,
        # Initial Config
        M: int = 1000,
        object_patch: int = 25,
        max_link_node: int = 25,
        # Embedding
        t_embed_dim: int = 200,
        # Input Encoder
        V_object_dims: list = [3, 64],
        V_robot_dims: list = [3, 3, 64],
        V_language_dims: list = [384],
        E_or_dims: list = [3, 3],
        E_rr_dims: list = [3, 3],
        # Backbone
        v_conv_dim=384,
        e_conv_dim=384,
        c_atten_head=32,
        num_layers=6,
        v_out_hidden_dims=[256, 128],
        out_dim=[3, 3, 1]
    ) -> None:
        super().__init__()

        self.M = M
        self.object_patch = object_patch
        self.max_link_node = max_link_node

        # Position Embedding
        self.t_embed_dim = t_embed_dim
        self.register_buffer("t_embed", self.sinusoidal_embedding(self.M, self.t_embed_dim))

        # Node Embedding
        self.V_object_dims = V_object_dims
        self.V_robot_dims = V_robot_dims
        self.V_language_dims = V_language_dims
        self.V_object_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, v_conv_dim) for in_dim in self.V_object_dims]
        )
        self.V_robot_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, v_conv_dim) for in_dim in self.V_robot_dims]
        )
        self.V_language_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, v_conv_dim) for in_dim in self.V_language_dims]
        )

        # Edge Embedding
        self.E_or_dims = E_or_dims
        self.E_rr_dims = E_rr_dims
        self.E_or_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, e_conv_dim) for in_dim in self.E_or_dims]
        )
        self.E_rr_in_layers = nn.ModuleList(
            [nn.Linear(in_dim, e_conv_dim) for in_dim in self.E_rr_dims]
        )

        # Backbone
        self.v_conv_dim = v_conv_dim
        self.e_conv_dim = e_conv_dim
        self.c_atten_head = c_atten_head
        self.num_layers = num_layers
        
        self.graph_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.graph_layers.append(
                TROLGraphLayer(
                    v_object_dim=v_conv_dim,
                    v_robot_dim=v_conv_dim,
                    v_language_dims=v_conv_dim,
                    t_embed_dim=t_embed_dim,
                    e_or_dim=e_conv_dim,
                    e_rr_dim=e_conv_dim,
                    lr_attn_dim=e_conv_dim,
                    c_atten_head=c_atten_head,
                    update_edge=(i < self.num_layers - 1)
                )
            )
        self.v_robot_wide_fc = nn.Linear(v_conv_dim * (1 + num_layers), v_conv_dim)

        # Output layers
        self.out_dim = out_dim
        self.v_robot_output_module = nn.ModuleList()

        for _, f_dim in enumerate(self.out_dim):
            self.v_robot_output_module.append(
                NaiveMLP(
                    f_dim + self.v_conv_dim, 
                    f_dim,
                    v_out_hidden_dims
                )
            )

    def sinusoidal_embedding(self, n, d):
        # Returns the standard positional embedding
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
        wk = wk.reshape((1, d))
        t = torch.arange(n).reshape((n, 1))
        embedding[:, ::2] = torch.sin(t * wk[:, ::2])
        embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
        return embedding

    def _encoder_(self, x, in_dims, in_layers):
        # Encode node and edge
        assert len(in_dims) == len(in_layers)
        cur = 0
        feat = None
        for i, d in enumerate(in_dims):
            partial = x[..., cur : cur + d]
            out = in_layers[i](partial)
            if feat is None:
                feat = out
            else:
                feat = feat + out
            cur += d
        return feat

    def forward(
        self, 
        V_O,
        V_L,
        noisy_V_R,
        noisy_E_OR,
        noisy_E_RR,
        t,
    ):

        # Position Embedding
        t_embed = self.t_embed[t]  # [B, 200]

        # Initial Encoding
        object_node_f = self._encoder_(
            V_O,
            self.V_object_dims,
            self.V_object_in_layers
        )  # [B, P, F]
        robot_node_f = self._encoder_(
            noisy_V_R,
            self.V_robot_dims,
            self.V_robot_in_layers
        )  # [B, L, F]
        language_node_f = self._encoder_(
            V_L,
            self.V_language_dims,
            self.V_language_in_layers
        )  # [B, LAN, F]

        or_edge_f = self._encoder_(
            noisy_E_OR,
            self.E_or_dims,
            self.E_or_in_layers
        )  # [B, L, P, F]
        rr_edge_f = self._encoder_(
            noisy_E_RR,
            self.E_rr_dims,
            self.E_rr_in_layers
        )  # [B, L, L, F]

        noisy_robot_node_f_list = [robot_node_f]

        for layer_id, layer in enumerate(self.graph_layers):
            robot_node_f, or_edge_f, rr_edge_f = layer(
                object_node_f,
                robot_node_f,
                language_node_f,
                or_edge_f,
                rr_edge_f,
                t_embed
            )
            noisy_robot_node_f_list.append(robot_node_f)

        update_robot_node_f = self.v_robot_wide_fc(
            torch.cat(noisy_robot_node_f_list, dim=-1)
        )  # [B, L, F]

        # Output
        v_robot_pred = []
        cur = 0
        for layer_id, f_dim in enumerate(self.out_dim):
            v_robot_pred.append(
                self.v_robot_output_module[layer_id](
                    torch.cat([update_robot_node_f, noisy_V_R[..., cur : cur + f_dim]], dim=-1)
                )
            )
            cur += f_dim
        v_robot_pred = torch.cat(v_robot_pred, dim=-1)
        return v_robot_pred
