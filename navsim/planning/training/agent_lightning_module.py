# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import traceback
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dp.dp_agent import DPAgent
from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from navsim.agents.gtrs_dense.gtrs_agent import GTRSAgent
from navsim.agents.gtrs_dense.hydra_features import state2traj
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory

def _rowwise_isin(tensor_1: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """Helper function for DrivoR metrics."""
    matches = (tensor_1[:, None] == target_tensor)
    return torch.sum(matches, dim=1, dtype=torch.bool)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(
        self,
        agent: AbstractAgent,
        combined: bool = False,
    ):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.combined = combined
        self.agent = agent
        self.v_params = get_pacifica_parameters()
        self.for_viz = False

        dp_preds_path = os.getenv("DP_PREDS", None)
        if dp_preds_path == "none":
            dp_preds_path = None

        if dp_preds_path is not None:
            self.dp_preds_2hz = pickle.load(open(dp_preds_path, "rb"))
        else:
            traceback.print_exc()
            self.dp_preds_2hz = dict()

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        features, targets, tokens = batch
        prediction = self.agent.forward(features)

        if isinstance(self.agent, TransfuserAgent):
            loss_output = self.agent.compute_loss(features, targets, prediction)
        else:
            loss_output = self.agent.compute_loss(features, targets, prediction, tokens)

        if isinstance(loss_output, dict):
            loss = loss_output["loss"]
            loss_detail = {k: v for k, v in loss_output.items() if k != "loss"}
        elif isinstance(loss_output, tuple):
            loss, loss_detail = loss_output
        else:
            loss = loss_output
            loss_detail = {}
        is_train = logging_prefix == "train"

        name_underscore = f"{logging_prefix}_loss"
        if not is_train:
            name_underscore = f"{logging_prefix}_loss_epoch"

        name_slash = f"{logging_prefix}/loss"
        if not is_train:
            name_slash = f"{logging_prefix}/loss_epoch"

        for name in [name_underscore, name_slash]:
            self.log(name, loss, on_step=is_train, on_epoch=True, prog_bar=True, sync_dist=True)

        for k, v in loss_detail.items():
            # Style A: val_dp_loss
            k_underscore = f"{logging_prefix}_{k}"
            if not is_train:
                k_underscore += "_epoch"

            # Style B: val/dp_loss
            k_slash = f"{logging_prefix}/{k}"
            if not is_train:
                k_slash += "_epoch"

            for name in [k_underscore, k_slash]:
                self.log(name, v, on_step=is_train, on_epoch=True, sync_dist=True)
        # ==========================================

        if "memory_info" in prediction:
            mem_info = prediction["memory_info"]

            metrics = {
                "mem_attn_persist": mem_info["attn_mean_persist"],
                "mem_attn_context": mem_info["attn_mean_context"],
                "mem_gate_mean": mem_info["gate_value"].mean(),
                "mem_gate_std": mem_info["gate_value"].std(),
                "mem_contrib_mean": mem_info["memory_contrib"].mean(),
            }

            for name, value in metrics.items():
                log_name = f"{logging_prefix}/{name}"
                if not is_train:
                    log_name = f"{logging_prefix}/{name}_epoch"

                self.log(log_name, value, on_step=is_train, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def on_before_optimizer_step(self, optimizer):
        norms = pl.utilities.grad_norm(self, norm_type=2)

        total_norm = {k: v for k, v in norms.items() if "total" in k}

        self.log_dict(total_norm)

    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        if self.combined:
            return self.predict_step_combined(batch, batch_idx)
        if isinstance(self.agent, GTRSAgent):
            return self.predict_step_hydra(batch, batch_idx)
        elif isinstance(self.agent, DPAgent):
            return self.predict_step_dp_traj(batch, batch_idx)
        elif isinstance(self.agent, DrivoRAgent):
            return self.predict_step_drivor(batch, batch_idx)
        elif isinstance(self.agent, TransfuserAgent):
            return self.predict_step_transfuser(batch, batch_idx)
        else:
            raise ValueError("unsupported agent")

    def predict_step_dp_traj(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            # [B, PROPOSAL, HORIZON, 3]
            all_trajs = predictions["dp_pred"]

        # interpolate them to 10Hz
        all_interpolated_proposals = []
        for batch_idx, token in enumerate(tokens):
            ego_state = EgoState.build_from_rear_axle(
                StateSE2(*features["ego_pose"].cpu().numpy()[batch_idx]),
                tire_steering_angle=0.0,
                vehicle_parameters=self.v_params,
                time_point=TimePoint(0),
                rear_axle_velocity_2d=StateVector2D(*features["ego_velocity"].cpu().numpy()[batch_idx]),
                rear_axle_acceleration_2d=StateVector2D(*features["ego_acceleration"].cpu().numpy()[batch_idx]),
            )
            interpolated_proposals = []
            proposals = all_trajs[batch_idx].cpu().numpy()
            for proposal in proposals:
                traj = Trajectory(proposal, TrajectorySampling(time_horizon=4, interval_length=0.5))
                trans_traj = transform_trajectory(traj, ego_state)
                interpolated_traj = get_trajectory_as_array(
                    trans_traj, TrajectorySampling(num_poses=40, interval_length=0.1), ego_state.time_point
                )
                final_traj = state2traj(interpolated_traj)
                interpolated_proposals.append(final_traj)
            interpolated_proposals = np.array(interpolated_proposals)
            interpolated_proposals = torch.from_numpy(interpolated_proposals).float()[None]
            all_interpolated_proposals.append(interpolated_proposals)
        # B, 100, 40, 3
        all_interpolated_proposals = torch.cat(all_interpolated_proposals, 0)

        result = {}
        for idx, (proposals, interp_proposals, token) in enumerate(
            zip(all_trajs.cpu().numpy(), all_interpolated_proposals.cpu().numpy(), tokens)
        ):
            # randomly choose a dp proposal
            final_pose = proposals[0]
            if final_pose.shape[0] == 40:
                interval_length = 0.1
            else:
                interval_length = 0.5

            result[token] = {
                "trajectory": Trajectory(
                    final_pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)
                ),
                "proposals": proposals,
                "interpolated_proposal": interp_proposals,
            }
        return result

    def predict_step_hydra(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"].cpu().numpy()

            imis = predictions["imi"].softmax(-1).log().cpu().numpy()
            no_at_fault_collisions_all = predictions["no_at_fault_collisions"].sigmoid().log().cpu().numpy()
            drivable_area_compliance_all = predictions["drivable_area_compliance"].sigmoid().log().cpu().numpy()
            time_to_collision_within_bound_all = (
                predictions["time_to_collision_within_bound"].sigmoid().log().cpu().numpy()
            )
            ego_progress_all = predictions["ego_progress"].sigmoid().log().cpu().numpy()
            driving_direction_compliance_all = predictions["driving_direction_compliance"].sigmoid().log().cpu().numpy()
            lane_keeping_all = predictions["lane_keeping"].sigmoid().log().cpu().numpy()
            traffic_light_compliance_all = predictions["traffic_light_compliance"].sigmoid().log().cpu().numpy()

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        result = {}
        for (
            pose,
            imi,
            no_at_fault_collisions,
            drivable_area_compliance,
            time_to_collision_within_bound,
            ego_progress,
            driving_direction_compliance,
            lane_keeping,
            traffic_light_compliance,
            token,
        ) in zip(
            poses,
            imis,
            no_at_fault_collisions_all,
            drivable_area_compliance_all,
            time_to_collision_within_bound_all,
            ego_progress_all,
            driving_direction_compliance_all,
            lane_keeping_all,
            traffic_light_compliance_all,
            tokens,
        ):
            result[token] = {
                "trajectory": Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                "imi": imi,
                "no_at_fault_collisions": no_at_fault_collisions,
                "drivable_area_compliance": drivable_area_compliance,
                "time_to_collision_within_bound": time_to_collision_within_bound,
                "ego_progress": ego_progress,
                "driving_direction_compliance": driving_direction_compliance,
                "lane_keeping": lane_keeping,
                "traffic_light_compliance": traffic_light_compliance,
            }
        return result

    def predict_step_transfuser(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"].cpu().numpy()

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        result = {}
        for pose, token in zip(poses, tokens):
            result[token] = {
                "trajectory": Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
            }
        return result

    def predict_step_combined(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], not_used: int):
        features, targets, tokens = batch
        device = features["ego_pose"].device
        all_interpolated_proposals = []
        for batch_idx, token in enumerate(tokens):
            proposals = self.dp_preds_2hz[token]["proposals"]
            ego_state = EgoState.build_from_rear_axle(
                StateSE2(*features["ego_pose"].cpu().numpy()[batch_idx]),
                tire_steering_angle=0.0,
                vehicle_parameters=self.v_params,
                time_point=TimePoint(0),
                rear_axle_velocity_2d=StateVector2D(*features["ego_velocity"].cpu().numpy()[batch_idx]),
                rear_axle_acceleration_2d=StateVector2D(*features["ego_acceleration"].cpu().numpy()[batch_idx]),
            )
            interpolated_proposals = []
            for proposal in proposals:
                traj = Trajectory(proposal, TrajectorySampling(time_horizon=4, interval_length=0.5))
                trans_traj = transform_trajectory(traj, ego_state)
                interpolated_traj = get_trajectory_as_array(
                    trans_traj, TrajectorySampling(num_poses=40, interval_length=0.1), ego_state.time_point
                )
                final_traj = state2traj(interpolated_traj)
                interpolated_proposals.append(final_traj)
            interpolated_proposals = np.array(interpolated_proposals)
            interpolated_proposals = torch.from_numpy(interpolated_proposals).float().to(device)[None]
            all_interpolated_proposals.append(interpolated_proposals)
        # B, 100, 40, 3
        all_interpolated_proposals = torch.cat(all_interpolated_proposals, 0)

        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.evaluate_dp_proposals(features, all_interpolated_proposals)
            poses = predictions["trajectory"].cpu().numpy()
            imis = predictions["imi"].softmax(-1).log().cpu().numpy()
            no_at_fault_collisions_all = predictions["no_at_fault_collisions"].sigmoid().log().cpu().numpy()
            drivable_area_compliance_all = predictions["drivable_area_compliance"].sigmoid().log().cpu().numpy()
            time_to_collision_within_bound_all = (
                predictions["time_to_collision_within_bound"].sigmoid().log().cpu().numpy()
            )
            ego_progress_all = predictions["ego_progress"].sigmoid().log().cpu().numpy()
            driving_direction_compliance_all = predictions["driving_direction_compliance"].sigmoid().log().cpu().numpy()
            lane_keeping_all = predictions["lane_keeping"].sigmoid().log().cpu().numpy()
            traffic_light_compliance_all = predictions["traffic_light_compliance"].sigmoid().log().cpu().numpy()

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        result = {}
        for (
            pose,
            interpolated_proposal,
            imi,
            no_at_fault_collisions,
            drivable_area_compliance,
            time_to_collision_within_bound,
            ego_progress,
            driving_direction_compliance,
            lane_keeping,
            traffic_light_compliance,
            token,
        ) in zip(
            poses,
            all_interpolated_proposals.cpu().numpy(),
            imis,
            no_at_fault_collisions_all,
            drivable_area_compliance_all,
            time_to_collision_within_bound_all,
            ego_progress_all,
            driving_direction_compliance_all,
            lane_keeping_all,
            traffic_light_compliance_all,
            tokens,
        ):
            result[token] = {
                "trajectory": Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                "imi": imi,
                "no_at_fault_collisions": no_at_fault_collisions,
                "drivable_area_compliance": drivable_area_compliance,
                "time_to_collision_within_bound": time_to_collision_within_bound,
                "ego_progress": ego_progress,
                "driving_direction_compliance": driving_direction_compliance,
                "lane_keeping": lane_keeping,
                "traffic_light_compliance": traffic_light_compliance,
                "interpolated_proposal": interpolated_proposal,
            }

        if self.agent._config.visualize and self.agent._config.use_memory:
            memory_info_all = predictions.get("memory_info", {})
            memory_node_all = predictions["memory_node"]
            memory_pos_all = predictions["memory_pos"]
            num_nodes = len(memory_node_all)
            len(tokens)

            for i, token in enumerate(tokens):
                if token in result:

                    sample_memory_info = {}
                    for key, value in memory_info_all.items():
                        if isinstance(value, torch.Tensor):
                            if value.dim() > 0 and value.size(0) == len(tokens):
                                sample_memory_info[key] = value[i].detach().cpu().numpy()
                            else:
                                sample_memory_info[key] = value.detach().cpu().numpy()
                        else:
                            sample_memory_info[key] = value

                    sample_nodes = []
                    for n in range(num_nodes):
                        node_raw = memory_node_all[n]
                        node_data = {}

                        for key, val in node_raw.items():
                            if isinstance(val, torch.Tensor):
                                sub_val = val[i]
                                node_data[key] = sub_val.item() if sub_val.numel() == 1 else sub_val.cpu().numpy()

                            elif isinstance(val, (list, tuple)):
                                if key == "api_lat_lon" or key == "gps_lat_lon":
                                    lat_val = val[0][i].item() if torch.is_tensor(val[0]) else val[0][i]
                                    lon_val = val[1][i].item() if torch.is_tensor(val[1]) else val[1][i]
                                    node_data[key] = (lat_val, lon_val)
                                else:
                                    node_data[key] = val[i]
                            else:
                                node_data[key] = val

                        sample_nodes.append(node_data)

                    result[token]["memory_info"] = sample_memory_info
                    result[token]["memory_node"] = sample_nodes
                    result[token]["memory_pos"] = memory_pos_all[i]

        return result

    def predict_step_drivor(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"]
            if self.for_viz:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
                ego_status = features["ego_status"]
        result = {}
        for index, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
            proposal = Trajectory(pose)
            if self.for_viz:
                proposal_list = [proposal_list[index].cpu().numpy() for proposal_list in all_proposed_trajectories]
                result[token] = {
                    "trajectory": proposal,
                    "all_proposals": proposal_list,
                    "all_proposal_scores": final_scores[index],
                    "high_level_command": ego_status[index],
                }
            else:
                result[token] = {"trajectory": proposal}
        return result
