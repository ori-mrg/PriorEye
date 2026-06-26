"""
Shared memory feature builder used by all agents (Hydra, Transfuser, DrivoR).
"""

import numpy as np
import torch

from navsim.agents.memory import memory_config
from navsim.agents.memory.memory_lookup import EmbeddingLookup
from navsim.common.dataclasses import AgentInput

EMBEDDING_MAP = {
    "DINOV2": (memory_config.NUPLAN_DINOV2_EMBEDDING_PICKLEFILE, 768),
    "SEGFORMER": (memory_config.NUPLAN_SEGFORMER_EMBEDDING_PICKLEFILE, 512),
    "SIGLIP2": (memory_config.NUPLAN_SIGLIP2_EMBEDDING_PICKLEFILE, 768),
}


class MemoryFeatureBuilder:
    """Handles EmbeddingLookup init and memory feature extraction."""

    def __init__(
        self,
        memory_embedding_model: str = "SEGFORMER",
        memory_corruption_test: bool = False,
        visualize: bool = False,
    ):
        self.model_name = memory_embedding_model.upper()
        self.model_key = self.model_name.lower()
        self._corruption_test = memory_corruption_test
        self._visualize = visualize

        embedding_pickle, embedding_dim = EMBEDDING_MAP[self.model_name]
        self._lookup = EmbeddingLookup(
            embedding_pickle_path=embedding_pickle,
            map_root=memory_config.NUPLAN_MAP_DIRECTORY,
            memory_embedding_dim=embedding_dim,
            map_version="nuplan-maps-v1.0",
            visualize=False,
        )

    def compute(self, agent_input: AgentInput):
        """
        Returns (embedding_tensor, pos_tensor, memory_node).
        memory_node is None unless visualize=True.
        """
        current_status = agent_input.ego_statuses[-1]
        ego_x, ego_y, ego_heading = agent_input.global_ego_pose

        cmd_idx = np.argmax(current_status.driving_command)
        command_map = {0: "turn_left", 1: "straight", 2: "turn_right", 3: "straight"}
        command_str = command_map.get(cmd_idx, "straight")
        map_name = agent_input.map_name

        # 1. Base retrieval
        embedding_np, _, node_list = self._lookup.find_memory_embedding(
            high_level_command=command_str, map_name=map_name, x=ego_x, y=ego_y, yaw=ego_heading
        )

        # - Corruption (visual)
        if self._corruption_test:
            OFFSET = 500.0
            theta = np.random.uniform(0, 2 * np.pi)
            dx = OFFSET * np.cos(theta)
            dy = OFFSET * np.sin(theta)

            embedding_other, _, _ = self._lookup.find_memory_embedding(
                high_level_command=command_str, map_name=map_name, x=ego_x + dx, y=ego_y + dy, yaw=ego_heading
            )
            mask = np.linalg.norm(embedding_np, axis=1) >= 1e-8
            embedding_np[mask] = embedding_other[mask]

        # 2. Compute relative position
        num_nodes = len(node_list)
        memory_pos_np = np.zeros((num_nodes, 2), dtype=np.float32)

        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)

        for i, node in enumerate(node_list):
            if node is not None:
                dx = node["x"] - ego_x
                dy = node["y"] - ego_y
                memory_pos_np[i, 0] = dx * cos_h - dy * sin_h
                memory_pos_np[i, 1] = dx * sin_h + dy * cos_h
        
        # - Corruption (spatial)
        if self._corruption_test:
            theta = np.random.uniform(0, 2 * np.pi, size=num_nodes)
            memory_pos_np[:, 0] = 500.0 * np.cos(theta)
            memory_pos_np[:, 1] = 500.0 * np.sin(theta)

        embedding_tensor = torch.from_numpy(embedding_np).float()
        pos_tensor = torch.from_numpy(memory_pos_np).float()

        # 3. Visualization nodes
        memory_node = None
        if self._visualize:
            vis_keys = ["gsv_image_path", "api_lat_lon", "gps_lat_lon", "map_name"]
            vis_node_list = []
            for node in node_list:
                if node is None:
                    vis_node_list.append({k: "" if k in ("gsv_image_path", "map_name") else (0.0, 0.0) for k in vis_keys})
                else:
                    vis_node_list.append({k: node.get(k, None) for k in vis_keys})
            memory_node = vis_node_list

        return embedding_tensor, pos_tensor, memory_node

    def compute_to_dict(self, agent_input: AgentInput, features: dict):
        """Convenience: compute and write results directly into features dict."""
        emb, pos, memory_node = self.compute(agent_input)
        features[f"memory_embedding_{self.model_key}"] = emb
        features[f"memory_pos_{self.model_key}"] = pos
        if memory_node is not None:
            features["memory_node"] = memory_node
