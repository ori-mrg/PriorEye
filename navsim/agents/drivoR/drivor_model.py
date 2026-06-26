from typing import Dict

import torch
import torch.nn as nn

from navsim.agents.drivoR.utils import pylogger

from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer

from navsim.agents.memory.memory import MemoryAugmentationModule

log = pylogger.get_pylogger(__name__)

# log.setLevel(logging.DEBUG)


class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.embed_dims = self._config.tf_d_model

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(
                torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features) * 1e-6,
                requires_grad=True,
            )

            # print("self.scene_embeds ", self.scene_embeds)

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(
                torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features) * 1e-6,
                requires_grad=True,
            )

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11 * 4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num * self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(
            num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(self.poses_num * 3, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        # get the trajectory decoders
        self.poses_num = config.num_poses
        self.state_size = 3
        ref_num = config.ref_num
        self.traj_head = nn.ModuleList(
            [MLP(config.tf_d_model, config.tf_d_ffn, traj_head_output_size) for _ in range(ref_num + 1)]
        )

        # scorer
        self.scorer = Scorer(config)

        self.b2d = config.b2d


        # ---  Memory Module Init ---
        self.use_memory = getattr(config, "use_memory", True)
        self.memory_embedding_model = getattr(config, "memory_embedding_model", "SIGLIP2")

        if self.use_memory:
            self.memory_embedding_dim = None
            if self.memory_embedding_model == "DINOV2":
                self.memory_embedding_dim = 768
            elif self.memory_embedding_model == "SEGFORMER":
                self.memory_embedding_dim = 512
            elif self.memory_embedding_model == "SIGLIP2":
                self.memory_embedding_dim = 768
            else:
                raise ValueError("Unsupported memory model")

            self._memory_module = MemoryAugmentationModule(embed_dim=config.tf_d_model, memory_in_dim=self.memory_embedding_dim)
        # -------------------------------------


    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]

        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")

        batch_size = ego_status.shape[0]

        scene_features = []
        # image features
        if self.num_cams > 0:

            if "image" in features:
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_scene_tokens = self.image_backbone(img, scene_tokens)

            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)
        log.debug(f"Scene features - {scene_features.shape}")


        # =================================================================
        #  Apply memory module
        # =================================================================
        if self.use_memory:
            model_key = self.memory_embedding_model.lower()
            emb_key = f"memory_embedding_{model_key}"
            pos_key = f"memory_pos_{model_key}"
            if emb_key in features and pos_key in features:
                memory_embedding = features[emb_key]
                memory_pos = features[pos_key]


                combined_query = torch.cat([ego_token, scene_features], dim=1) 

                fused_out, memory_info = self._memory_module(
                    current_states=combined_query,
                    memory_embedding=memory_embedding,
                    memory_pos=memory_pos,
                    return_info=True,
                )

                ego_token = fused_out[:, 0:1] 
                scene_features = fused_out[:, 1:]
            else:
                memory_info = {} 

        # =================================================================


        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode the trajectories at each step of the decoder
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        log.debug(f"Trajectory decoder - {len(token_list)}")
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i + 1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)

        traj_tokens = token_list[-1]
        proposals = proposal_list[-1]

        output = {}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        # scoring
        B, N, _, _ = proposals.shape

        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())  # (B, N, d_model)
        tr_out = self.scorer_attention(embedded_traj, scene_features)  # (B, N, d_model)
        tr_out = tr_out + ego_token
        (
            pred_logit,
            pred_logit2,
            pred_agents_states,
            pred_area_logit,
            bev_semantic_map,
            agent_states,
            agent_labels,
        ) = self.scorer(proposals, tr_out)

        output["pred_logit"] = pred_logit
        output["pred_logit2"] = pred_logit2
        output["pred_agents_states"] = pred_agents_states
        output["pred_area_logit"] = pred_area_logit
        output["bev_semantic_map"] = bev_semantic_map
        output["agent_states"] = agent_states
        output["agent_labels"] = agent_labels

        pdm_score = (
            self._config.noc * pred_logit["no_at_fault_collisions"].sigmoid().log()
            + self._config.dac * pred_logit["drivable_area_compliance"].sigmoid().log()
            + self._config.ddc * pred_logit["driving_direction_compliance"].sigmoid().log()
            + (
                self._config.ttc * pred_logit["time_to_collision_within_bound"].sigmoid()
                + self._config.ep * pred_logit["ego_progress"].sigmoid()
                + self._config.comfort * pred_logit["comfort"].sigmoid()
            ).log()
        )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        if self.use_memory:
            if 'memory_info' in locals():
                output["memory_info"] = memory_info
            
            if self._config.visualize and "memory_node" in features:
                output["memory_node"] = features["memory_node"]
                if 'memory_pos' in locals():
                    output["memory_pos"] = memory_pos

        return output