import logging
import os
import pickle
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"
FRAME_INTERVAL = 1
target_log_names = [
    "2021.05.25.14.24.08_veh-25_03764_04034",
    '2021.05.25.17.54.41_veh-35_04111_04288',
    '2021.06.03.12.02.06_veh-35_00233_00609',
    '2021.06.28.13.53.26_veh-26_00492_00696',
    '2021.08.16.14.23.37_veh-45_00015_00132',
    '2021.08.30.13.45.25_veh-40_00610_00771',
    '2021.09.09.17.18.51_veh-48_01248_01450',
    '2021.09.16.13.53.10_veh-42_00077_00153',
    '2021.09.16.13.53.10_veh-42_00180_00342',
    '2021.09.16.14.39.34_veh-42_00032_00186',
    '2021.09.16.15.12.03_veh-42_01037_01434',
    '2021.09.29.14.44.26_veh-28_00238_00320',
    '2021.09.29.15.23.04_veh-28_00601_00802',
    '2021.09.29.18.19.40_veh-28_01268_01685',
    '2021.10.06.07.26.10_veh-52_00006_00398',
    '2021.10.06.07.26.10_veh-52_00422_00728',
    '2021.10.06.07.26.10_veh-52_00772_00917'
]



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    combined = cfg.get("combined_inference", False)

    print(f"Combined inference: {combined}")

    dump_path = os.getenv("SUBSCORE_PATH")
    print(f"Subscore/Trajectories saved to {dump_path}")

    # gpu inference
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.include_synthetic_scenes = False
    scene_filter.log_names = target_log_names
    scene_filter.frame_interval = FRAME_INTERVAL
    # scene_filter.max_scenes = None  # Remove specific count limit
    token_file = os.getenv("TOKEN_FILE", None)
    if token_file and os.path.exists(token_file):
        with open(token_file, "r") as f:
            tokens = [line.strip() for line in f if line.strip()]
        scene_filter.tokens = tokens
        scene_filter.log_names = None
        print(f"Filtering by {len(tokens)} tokens from {token_file}")
    else:
        scene_filter.tokens = None  # Unset specific token filter (to extract based on log_names)

    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=None,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
        is_training=False,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    predictions = trainer.predict(
        AgentLightningModule(agent=agent, combined=combined), dataloader, return_predictions=True
    )

    dist.barrier()
    all_predictions = [None for _ in range(dist.get_world_size())]

    if dist.is_initialized():
        dist.all_gather_object(all_predictions, predictions)
    else:
        all_predictions.append(predictions)

    if dist.get_rank() != 0:
        return None

    merged_predictions = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            merged_predictions.update(d)

    pickle.dump(merged_predictions, open(dump_path, "wb"))


if __name__ == "__main__":
    main()
