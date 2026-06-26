import os
import gzip
import pickle
import torch
import math
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D

# Global cache to speed up map loading
MAP_CACHE = {}

def global_to_local(target_x, target_y, ego_x, ego_y, ego_yaw):
    """Global -> Local coordinate transformation"""
    dx = target_x - ego_x
    dy = target_y - ego_y
    cos_a = math.cos(-ego_yaw)
    sin_a = math.sin(-ego_yaw)
    return dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a

def decode_path_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        chars = [chr(c) for c in tensor.cpu().tolist() if c != 0]
        return "".join(chars)
    return str(tensor)

def load_gz_pickle(path):
    """Check GZIP header then load (Robust Load)"""
    is_gzipped = False
    with open(path, "rb") as f:
        magic_number = f.read(2)
        if magic_number == b'\x1f\x8b':
            is_gzipped = True
    
    def robust_load(file_obj):
        try:
            return torch.load(file_obj, map_location="cpu")
        except RuntimeError:
            file_obj.seek(0)
            return pickle.load(file_obj)

    if is_gzipped:
        with gzip.open(path, "rb") as f:
            return robust_load(f)
    else:
        with open(path, "rb") as f:
            return robust_load(f)

def render_local_map(ax, map_name, map_root, ego_pose, radius=80):
    """
    Fetches nearby map objects and draws them in local coordinates.
    """
    global MAP_CACHE
    
    # 1. Load Map API (with caching)
    if map_name not in MAP_CACHE:
        try:
            # map_version is usually "nuplan-maps-v1.0"
            MAP_CACHE[map_name] = get_maps_api(map_root, "nuplan-maps-v1.0", map_name)
        except Exception as e:
            print(f"[Warning] Failed to load map {map_name}: {e}")
            return

    map_api = MAP_CACHE[map_name]
    ego_x, ego_y, ego_yaw = ego_pose

    # 2. Query nearby objects (lanes, intersections, etc.)
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    proximal_objects = map_api.get_proximal_map_objects(Point2D(ego_x, ego_y), radius, layers)

    # 3. Draw
    for layer in layers:
        for obj in proximal_objects[layer]:
            # Extract polygon coordinates
            if hasattr(obj, 'polygon'):
                poly = obj.polygon
                if poly.is_empty: continue
                
                # Exterior coordinates (Global)
                x_pts, y_pts = poly.exterior.coords.xy
                
                # Convert to local
                local_xs = []
                local_ys = []
                for gx, gy in zip(x_pts, y_pts):
                    lx, ly = global_to_local(gx, gy, ego_x, ego_y, ego_yaw)
                    local_xs.append(lx)
                    local_ys.append(ly)
                
                # Plot (gray solid line)
                ax.plot(local_xs, local_ys, color='gray', alpha=0.5, linewidth=0.8, zorder=1)

# -----------------------------------------------------------

def visualize_token_dir(token_dir, output_dir, map_root):
    token = os.path.basename(token_dir)
    feat_path = os.path.join(token_dir, "transfuser_feature.gz")
    target_path = os.path.join(token_dir, "transfuser_target.gz")

    if not os.path.exists(feat_path) or not os.path.exists(target_path):
        return

    try:
        features = load_gz_pickle(feat_path)
        targets = load_gz_pickle(target_path)
        
        gt_traj = targets.get("trajectory", None)
        node_list_tensor = features.get("memory_node_list", None)
        
        if node_list_tensor is None: return 

        pkl_path = decode_path_tensor(node_list_tensor)
        if not os.path.exists(pkl_path):
            print(f"[Missing] Pickle file: {pkl_path}")
            return

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        
        # Legacy version compatibility
        if isinstance(pkl_data, list): return

        node_list = pkl_data.get("node_list", [])
        ego_pose = pkl_data.get("global_ego_pose", None)
        command = pkl_data.get("command", "unknown")
        map_name = pkl_data.get("map_name", "unknown")

        if ego_pose is None: return
        
        ego_x, ego_y, ego_yaw = ego_pose

        # -----------------------------------------------------------
        # Visualization (Local Frame)
        # -----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 10))

        if map_root:
            render_local_map(ax, map_name, map_root, ego_pose)

        # Ego
        ax.arrow(0, 0, 2.0, 0, head_width=0.5, head_length=0.5, fc='r', ec='r', zorder=20, label='Ego')
        ax.plot(0, 0, 'ro', markersize=5, zorder=20)

        # GT Future
        if gt_traj is not None:
            if isinstance(gt_traj, torch.Tensor): gt_traj = gt_traj.numpy()
            ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'g--', linewidth=2.5, alpha=0.9, label='GT Future', zorder=15)
            ax.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'g*', markersize=12, zorder=15)

        # Memory Path (Global -> Local)
        valid_nodes = [n for n in node_list if n is not None]
        if valid_nodes:
            lxs, lys = [], []
            for n in valid_nodes:
                lx, ly = global_to_local(n['x'], n['y'], ego_x, ego_y, ego_yaw)
                lxs.append(lx)
                lys.append(ly)
            ax.plot(lxs, lys, 'b-', linewidth=2, alpha=0.8, label='Memory Path', zorder=10)
            ax.scatter(lxs, lys, c='blue', s=20, zorder=10)

        ax.set_title(f"Token: {token}\nCmd: {command} | Map: {map_name}")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Fix range (Zoom)
        # ax.set_xlim(-20, 80)
        # ax.set_ylim(-50, 50)

        out_path = os.path.join(output_dir, f"viz_{token}.png")
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"Error drawing {token}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=str, required=True, help="Path to navsim cache logs (e.g. /repos/.../exp/navmini_cache)")
    
    # [Added] Map path argument (default value configurable)
    parser.add_argument("--map_root", type=str, default="/navsim_map", help="Path to nuplan maps root")
    
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    output_dir = os.path.join("./", "visualizations_gz_map")
    os.makedirs(output_dir, exist_ok=True)

    log_dirs = glob.glob(os.path.join(args.cache_root, "20*"))
    print(f"Found {len(log_dirs)} log directories.")

    count = 0
    for log_dir in tqdm(log_dirs):
        if not os.path.isdir(log_dir): continue
        token_dirs = [d for d in glob.glob(os.path.join(log_dir, "*")) if os.path.isdir(d)]
        
        for token_dir in token_dirs:
            if count >= args.limit: break
            # Pass map_root
            visualize_token_dir(token_dir, output_dir, args.map_root)
            count += 1
        
        if count >= args.limit: break

    print(f"\nSaved {count} images to {output_dir}")

if __name__ == "__main__":
    main()