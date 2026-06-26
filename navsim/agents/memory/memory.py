import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        return x


class LinearProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        return self.projection(x)


class Memory2DPositionEmbedding(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.half_dim = embed_dim // 2

    def forward(self, coords):
        """
        Args:
            coords (Tensor): [B, N, 2] -> Relative position (x, y)
        Returns:
            pe (Tensor): [B, N, embed_dim]
        """
        div_term = torch.exp(torch.arange(0, self.half_dim, 2) * (-math.log(10000.0) / self.half_dim))
        div_term = div_term.to(coords.device)

        # X (Sine/Cosine)
        pos_x = torch.zeros(coords.shape[0], coords.shape[1], self.half_dim, device=coords.device)
        x_coords = coords[:, :, 0].unsqueeze(-1)  # [B, N, 1]
        pos_x[:, :, 0::2] = torch.sin(x_coords * div_term)
        pos_x[:, :, 1::2] = torch.cos(x_coords * div_term)

        # Y (Sine/Cosine)
        pos_y = torch.zeros(coords.shape[0], coords.shape[1], self.half_dim, device=coords.device)
        y_coords = coords[:, :, 1].unsqueeze(-1)  # [B, N, 1]
        pos_y[:, :, 0::2] = torch.sin(y_coords * div_term)
        pos_y[:, :, 1::2] = torch.cos(y_coords * div_term)

        # Concat [B, N, embed_dim]
        pe = torch.cat([pos_x, pos_y], dim=-1)
        return pe


class MemoryAugmentationModule(nn.Module):
    def __init__(
        self,
        embed_dim=512,  # Model's internal dimension
        memory_in_dim=768,  # Input dimension of visual priors
        persistent_mem_size=16,  # Number of learnable persistent tokens
        num_heads=8,
        dropout=0.5,
        dist_threshold=100.0,  # Contextual memory mask distance threshold (m)
    ):
        """
        It fuses current states with :
            1. Contextual Memory 
            2. Persistent Memory
        """
        super(MemoryAugmentationModule, self).__init__()

        self.embed_dim = embed_dim
        self.persistent_mem_size = persistent_mem_size

        self.memory_projection = Conv1dProjection(memory_in_dim, embed_dim)
        self.pos_embed_layer = Memory2DPositionEmbedding(embed_dim)

        self.persistent_memory = nn.Parameter(torch.randn(1, persistent_mem_size, embed_dim))

        # Q: Current Driving State, K/V: [Persistent Memory || Contextual Memory]
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        # Memory gate decides how much to fuse memory into the current state.
        self.memory_gate = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, embed_dim // 4), nn.GELU(), nn.Linear(embed_dim // 4, embed_dim), nn.Sigmoid()
        )
        nn.init.constant_(self.memory_gate[-2].bias, -2.0)

        self.fusion_norm = nn.LayerNorm(embed_dim)

        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.m_norm = nn.LayerNorm(embed_dim)
        self.dist_threshold = dist_threshold

    def forward(self, current_states, memory_embedding, memory_pos, return_info=False):

        # --- Contextual Memory Preparation ---
        batch_size = memory_embedding.shape[0]

        context_memory = self.memory_projection(memory_embedding)
        ctx_pos_embed = self.pos_embed_layer(memory_pos)
        context_memory = context_memory + ctx_pos_embed

        # --- Persistent Memory Preparation ---
        persistent_memory = self.persistent_memory.repeat(batch_size, 1, 1)

        # --- Memory Concat ---
        full_memory = torch.cat([persistent_memory, context_memory], dim=1)

        # --- Masking Logic ---
        context_padding_mask = memory_embedding.abs().sum(dim=-1) == 0
        dist = torch.norm(memory_pos, dim=-1)
        is_too_far = dist > self.dist_threshold
        context_padding_mask = context_padding_mask | is_too_far

        persistent_padding_mask = torch.zeros(
            batch_size, self.persistent_memory.shape[1], dtype=torch.bool, device=current_states.device
        )

        key_padding_mask = torch.cat([persistent_padding_mask, context_padding_mask], dim=1)

        # --- Attention ---
        c_norm = self.q_norm(current_states)
        full_memory = self.k_norm(full_memory)
        attended_memory, attn_weights = self.memory_attention(
            query=c_norm,
            key=full_memory,
            value=full_memory,
            need_weights=return_info,
            key_padding_mask=key_padding_mask,
        )

        m_norm = self.m_norm(attended_memory)

        # --- Memory Gate ---
        delta = c_norm - m_norm
        l2_dist = torch.norm(delta, p=2, dim=-1, keepdim=True) / math.sqrt(self.embed_dim)
        cos_sim = F.cosine_similarity(c_norm, m_norm, dim=-1).unsqueeze(-1)

        gate_input = torch.cat([c_norm, m_norm, l2_dist, cos_sim], dim=-1)
        gate = self.memory_gate(gate_input)

        fused_states = current_states + (gate * m_norm)
        current_states_out = self.fusion_norm(fused_states)

        if return_info:
            persist_attn = attn_weights[:, :, : self.persistent_mem_size]
            persist_mean = persist_attn.mean().detach()
            context_attn = attn_weights[:, :, self.persistent_mem_size :]

            valid_mask_float = (~context_padding_mask).float().unsqueeze(1)
            valid_attn_sum = (context_attn * valid_mask_float).sum()
            valid_token_count = valid_mask_float.expand_as(context_attn).sum()

            context_mean = (valid_attn_sum / (valid_token_count + 1e-6)).detach()

            info_dict = {
                "gate_value": gate.detach(),  # [B, Q, D]
                "attn_weights": attn_weights.detach(),  # [B, Q, K]
                "attn_mean_persist": persist_mean,  # Scalar Tensor
                "attn_mean_context": context_mean,  # Scalar Tensor
                "memory_contrib": (gate * m_norm).norm(dim=-1).detach(),
            }
            return current_states_out, info_dict
        return current_states_out
