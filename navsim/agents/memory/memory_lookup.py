import math
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from navsim.agents.memory import memory_config
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

# NuPlan Imports
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

# User Utils Imports
from navsim.agents.memory.memory_util import _normalize_angle_rad, draw_arrow, load_data


class EmbeddingLookup:
    def __init__(
        self,
        embedding_pickle_path: str,
        map_root: str,
        map_version: str = "nuplan-maps-v1.0",
        memory_node_num: int = 20,
        memory_embedding_dim: int = 768,
        search_radius: float = 5.0,
        visualize: bool = False,
    ):
        self.memory_node_num = memory_node_num
        self.memory_embedding_dim = memory_embedding_dim
        self.path_length = memory_node_num * memory_config.DISTANCE_BIN_SIZE
        self.visualize = visualize

        self.maps_api = {}
        for map_name in memory_config.MAP_NAMES_NUPLAN:
            self.maps_api[map_name] = get_maps_api(map_root=map_root, map_version=map_version, map_name=map_name)

        self.all_embeddings_dict = load_data(embedding_pickle_path)
        self.search_radius = search_radius

    def _get_lane_object(self, map_api: AbstractMap, lane_id: str):
        lane_obj = None

        try:
            lane_obj = map_api.get_map_object(lane_id, SemanticMapLayer.LANE)
        except Exception:
            pass

        if lane_obj is None:
            try:
                lane_obj = map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)
            except Exception:
                pass

        return lane_obj

    def _get_lane_helpers(self, map_api: AbstractMap, lane_id: str, x: float = None, y: float = None):
        lane = self._get_lane_object(map_api, lane_id)
        if lane is None:
            return None, None

        discrete_path = lane.baseline_path.discrete_path
        if not discrete_path:
            return 0.0, 0.0

        poses = np.array([[p.x, p.y] for p in discrete_path])

        if len(poses) < 2:
            return 0.0, 0.0

        segment_distances = np.linalg.norm(np.diff(poses, axis=0), axis=1)
        total_length = np.sum(segment_distances)

        if x is None or y is None:
            return total_length, 0.0

        agent_coord = np.array([x, y])
        distances_to_agent = np.linalg.norm(poses - agent_coord, axis=1)
        closest_index = np.argmin(distances_to_agent)

        if closest_index == len(poses) - 1:
            return total_length, 0.0

        remaining_length = np.sum(segment_distances[closest_index:])
        return total_length, remaining_length

    def _find_paths_recursive(
        self,
        map_api: AbstractMap,
        current_lane_id: str,
        current_path: List[str],
        current_distance: float,
        max_distance: float,
        all_paths: List[List[str]],
    ):
        current_lane = self._get_lane_object(map_api, current_lane_id)
        if current_lane is None:
            return

        outgoing_edges = current_lane.outgoing_edges

        if not outgoing_edges:
            all_paths.append(list(current_path))
            return

        for edge in outgoing_edges:
            next_lane_id = edge.id

            # Prevent Cycle
            if next_lane_id in current_path:
                all_paths.append(list(current_path) + [next_lane_id])
                continue

            lane_length, _ = self._get_lane_helpers(map_api, next_lane_id)
            if lane_length is None:
                continue

            new_distance = current_distance + lane_length
            new_path = list(current_path) + [next_lane_id]

            if new_distance >= max_distance:
                all_paths.append(new_path)
            else:
                self._find_paths_recursive(
                    map_api,
                    next_lane_id,
                    new_path,
                    new_distance,
                    max_distance,
                    all_paths,
                )

    def find_closest_lane_by_pose(self, map_name: str, x: float, y: float, yaw: float) -> Optional[str]:

        MAX_ANGLE_THRESHOLD = np.pi / 3  # 60 deg
        map_api = self.maps_api[map_name]

        yaw = _normalize_angle_rad(yaw)

        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        proximal_objects = map_api.get_proximal_map_objects(Point2D(x, y), self.search_radius, layers)

        candidate_lanes = []
        for layer in layers:
            candidate_lanes.extend(proximal_objects.get(layer, []))

        best_lane_id = None
        min_distance = np.inf

        for lane in candidate_lanes:
            discrete_path = lane.baseline_path.discrete_path
            if not discrete_path:
                continue

            # (N, 3) -> x, y, heading
            lane_poses = np.array([[p.x, p.y, p.heading] for p in discrete_path])

            dists = np.linalg.norm(lane_poses[:, :2] - np.array([x, y]), axis=1)
            closest_idx = np.argmin(dists)
            distance_to_lane = dists[closest_idx]

            # Filtering heading
            lane_yaw = _normalize_angle_rad(lane_poses[closest_idx, 2])
            yaw_diff = abs(_normalize_angle_rad(yaw - lane_yaw))

            if yaw_diff > MAX_ANGLE_THRESHOLD:
                continue

            if distance_to_lane < min_distance:
                min_distance = distance_to_lane
                best_lane_id = lane.id

        return best_lane_id

    def find_lane_paths_until_distance(self, map_name: str, x: float, y: float, yaw: float) -> List[List[str]]:
        max_distance = self.path_length
        map_api = self.maps_api[map_name]

        start_lane_id = self.find_closest_lane_by_pose(map_name, x, y, yaw)

        if not start_lane_id:
            return []

        _, remaining_length_start_lane = self._get_lane_helpers(map_api, start_lane_id, x, y)

        if remaining_length_start_lane is None:
            return []

        all_paths = []
        current_path = [start_lane_id]

        if remaining_length_start_lane >= max_distance:
            all_paths.append(current_path)
            return all_paths

        self._find_paths_recursive(
            map_api,
            start_lane_id,
            current_path,
            remaining_length_start_lane,
            max_distance,
            all_paths,
        )

        return all_paths

    def visualize_memory_alignment(self, map_name, x, y, yaw, lane_ids, node_list, command):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis("equal")

        # 2. HD Map Lane
        map_api = self.maps_api[map_name]
        for lid in lane_ids:
            lane_obj = self._get_lane_object(map_api, lid)
            if lane_obj and lane_obj.baseline_path.discrete_path:
                pts = np.array([[p.x, p.y] for p in lane_obj.baseline_path.discrete_path])
                ax.plot(pts[:, 0], pts[:, 1], color="magenta", linewidth=1.5, alpha=0.3)

        # 3.  Ego
        draw_arrow(ax, x, y, yaw, 3, "red")
        ax.plot(x, y, "ro", markersize=8, label="Ego (Current)")

        final_nodes = [n for n in node_list if n is not None]

        if final_nodes:
            gsv_xs = [n["x"] for n in final_nodes]
            gsv_ys = [n["y"] for n in final_nodes]

            ax.scatter(
                gsv_xs, gsv_ys, c="black", marker="x", s=60, zorder=30, label=f"Selected Memory ({len(final_nodes)})"
            )

            ax.plot(gsv_xs, gsv_ys, "k--", alpha=0.4, linewidth=1)

            ax.text(gsv_xs[0], gsv_ys[0], "Start", fontsize=9, fontweight="bold")

        ax.set_title(f"Final Filtered Memory Alignment\nCommand: {command}")
        ax.legend()
        plt.grid(True, linestyle="--", alpha=0.4)

        # Save
        output_dir = os.path.join("./", "vis_final_memory")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"final_align_{map_name}_{command}_{int(x)}_{int(y)}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)
        print(f"Final filtered visualization saved to {output_dir}")

    def find_lane_via_command(self, high_level_command, map_name, x, y, yaw):
        map_api = self.maps_api[map_name]

        lane_candidates = self.find_lane_paths_until_distance(map_name, x, y, yaw)

        if not lane_candidates:
            return None

        lane_candidate_angles = {}
        lane_candidate_geometries = {}

        for lane_candidate in lane_candidates:
            try:
                current_lane = self._get_lane_object(map_api, lane_candidate[0])
                outgoing_lane = self._get_lane_object(map_api, lane_candidate[-1])

                if not current_lane or not outgoing_lane:
                    continue

                curr_path = current_lane.baseline_path.discrete_path
                out_path = outgoing_lane.baseline_path.discrete_path

                if not curr_path or not out_path:
                    continue

                current_lane_poses = np.array([[p.x, p.y] for p in curr_path])
                outgoing_lane_poses = np.array([[p.x, p.y] for p in out_path])

                all_segment_geoms = []
                for seg_id in lane_candidate:
                    seg_obj = self._get_lane_object(map_api, seg_id)
                    if seg_obj and seg_obj.baseline_path.discrete_path:
                        pts = np.array([[p.x, p.y] for p in seg_obj.baseline_path.discrete_path])
                        all_segment_geoms.append(pts)

                lane_candidate_geometries[tuple(lane_candidate)] = {
                    "current": current_lane_poses,
                    "outgoing": outgoing_lane_poses,
                    "all": all_segment_geoms,
                }

                vec_current = current_lane_poses[-1] - current_lane_poses[0]
                vec_outgoing = outgoing_lane_poses[-1] - outgoing_lane_poses[0]

                vec_current = vec_current / (np.linalg.norm(vec_current) + 1e-6)
                vec_outgoing = vec_outgoing / (np.linalg.norm(vec_outgoing) + 1e-6)

                dot = np.dot(vec_current, vec_outgoing)
                cross_z = vec_current[0] * vec_outgoing[1] - vec_current[1] * vec_outgoing[0]
                delta_angle_deg = np.rad2deg(np.arctan2(cross_z, dot))

                lane_candidate_angles[tuple(lane_candidate)] = delta_angle_deg

            except Exception as e:
                print(f"Error processing lane candidate {lane_candidate}: {e}")
                continue

        if not lane_candidate_angles:
            return None

        # Command Logic
        best_candidate_tuple = None
        if high_level_command == "turn_left":
            best_candidate_tuple = max(lane_candidate_angles, key=lane_candidate_angles.get)
        elif high_level_command == "turn_right":
            best_candidate_tuple = min(lane_candidate_angles, key=lane_candidate_angles.get)
        elif high_level_command == "straight":
            best_candidate_tuple = min(lane_candidate_angles, key=lambda k: abs(lane_candidate_angles[k]))

        return list(best_candidate_tuple) if best_candidate_tuple else None

    def find_memory_embedding(self, high_level_command, map_name, x, y, yaw):

        lane_ids = self.find_lane_via_command(high_level_command, map_name, x, y, yaw)
        TARGET_ROWS = self.memory_node_num
        distance_bin_size = memory_config.DISTANCE_BIN_SIZE
        MAX_DISTANCE = TARGET_ROWS * distance_bin_size
        EMBEDDING_DIM = self.memory_embedding_dim

        memory_embedding = np.zeros((TARGET_ROWS, EMBEDDING_DIM))
        node_list = [None] * TARGET_ROWS
        if lane_ids is None:
            return memory_embedding, lane_ids, node_list

        heading_x = math.cos(yaw)
        heading_y = math.sin(yaw)

        cumulative_distance = 0.0
        last_x, last_y = x, y
        bin_filled_status = [False] * TARGET_ROWS
        filled_bins_count = 0

        if map_name not in self.all_embeddings_dict:
            print(f"Warning: No embeddings found for map {map_name}")
            return memory_embedding, lane_ids, node_list

        map_embeddings = self.all_embeddings_dict[map_name]

        for lane_id in lane_ids:
            if filled_bins_count >= TARGET_ROWS:
                break

            if lane_id not in map_embeddings:
                continue

            nodes = map_embeddings[lane_id]
            sorted_node_keys = sorted(nodes.keys())

            for node_idx in sorted_node_keys:
                if filled_bins_count >= TARGET_ROWS:
                    break

                node = nodes[node_idx]
                node_x, node_y = node["x"], node["y"]
                vec_x = node_x - x
                vec_y = node_y - y
                dot_product = (heading_x * vec_x) + (heading_y * vec_y)

                if dot_product < 0:
                    continue

                segment_distance = math.hypot(node_x - last_x, node_y - last_y)
                cumulative_distance += segment_distance
                last_x, last_y = node_x, node_y

                if cumulative_distance > MAX_DISTANCE:
                    break

                bin_index = int(cumulative_distance // distance_bin_size)

                if 0 <= bin_index < TARGET_ROWS:
                    if not bin_filled_status[bin_index]:
                        embedding_data = np.array(node["embedding"]).squeeze()
                        memory_embedding[bin_index] = embedding_data
                        node_list[bin_index] = node
                        bin_filled_status[bin_index] = True
                        filled_bins_count += 1

        if self.visualize and lane_ids:
            self.visualize_memory_alignment(map_name, x, y, yaw, lane_ids, node_list, high_level_command)

        return memory_embedding, lane_ids, node_list

    def find_memory_embedding_knn(self, map_name, x, y, k=20):
        """
        Find memory closest.
        This function is for the test of memory corruption
        """
        TARGET_ROWS = self.memory_node_num
        EMBEDDING_DIM = self.memory_embedding_dim

        memory_embedding = np.zeros((TARGET_ROWS, EMBEDDING_DIM))
        node_list = [None] * TARGET_ROWS

        if map_name not in self.all_embeddings_dict:
            return memory_embedding, [], node_list

        all_nodes = []
        for lane_id, nodes in self.all_embeddings_dict[map_name].items():
            for node_idx, node in nodes.items():
                all_nodes.append(node)

        query_pt = np.array([x, y])
        node_coords = np.array([[n["x"], n["y"]] for n in all_nodes])
        distances = np.linalg.norm(node_coords - query_pt, axis=1)

        closest_indices = np.argsort(distances)[:TARGET_ROWS]

        for i, idx in enumerate(closest_indices):
            node = all_nodes[idx]
            memory_embedding[i] = np.array(node["embedding"]).squeeze()
            node_list[i] = node

        return memory_embedding, ["knn_mode"], node_list


def main():
    embedding_pickle = memory_config.NUPLAN_SIGLIP2_EMBEDDING_PICKLEFILE
    embedding_dim = 768

    lookup = EmbeddingLookup(
        embedding_pickle_path=embedding_pickle,
        map_root=memory_config.NUPLAN_MAP_DIRECTORY,
        memory_embedding_dim=embedding_dim,
        map_version="nuplan-maps-v1.0",
        visualize=True,
    )

    x = 664465.4126458991
    y = 3997855.348150607
    heading = 0.9747825400403761

    command = "turn_right"
    map_name = "us-nv-las-vegas-strip"
    print(f"\n[Query] : ({x:.2f}, {y:.2f}),  {heading:.2f}, Command: {command}")

    memory_embedding, lane_ids, node_list = lookup.find_memory_embedding(
        high_level_command=command, map_name=map_name, x=x, y=y, yaw=heading
    )

    print("\n" + "=" * 50)
    print("               RESULT SUMMARY")
    print("=" * 50)

    if lane_ids is None:
        print(">> Fail")
    else:
        print(f">> ({len(lane_ids)} lanes):")
        for i, lid in enumerate(lane_ids):
            print(f"   [{i}] {lid}")

        print("-" * 30)
        print(f">> Embedding Shape: {memory_embedding.shape}")


if __name__ == "__main__":
    main()
