import logging
import os
import pickle
from pathlib import Path
from typing import Dict

import hydra
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle_gpu"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for GPU-based submission creation script.
    Uses PyTorch Lightning trainer.predict() for batch GPU inference,
    then creates a submission pickle.
    :param cfg: omegaconf dictionary
    """
    build_logger(cfg)
    combined = cfg.get('combined_inference', False)

    # GPU inference
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    scene_loader_inference = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
        is_training=False,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    # Scene loader without sensors for token split info
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    predictions = trainer.predict(
        AgentLightningModule(
            agent=agent,
            combined=combined,
        ),
        dataloader,
        return_predictions=True,
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

    # Split predictions into first stage and second stage
    tokens_stage_one = set(scene_loader.tokens_stage_one)
    tokens_stage_two = set(scene_loader.reactive_tokens_stage_two)

    first_stage_output: Dict[str, Trajectory] = {}
    for token in tokens_stage_one:
        if token in merged_predictions:
            first_stage_output[token] = merged_predictions[token]["trajectory"]

    second_stage_output: Dict[str, Trajectory] = {}
    for token in tokens_stage_two:
        if token in merged_predictions:
            second_stage_output[token] = merged_predictions[token]["trajectory"]

    logger.info(
        f"First stage: {len(first_stage_output)} predictions, "
        f"Second stage: {len(second_stage_output)} predictions"
    )

    submission = {
        "team_name": cfg.team_name,
        "authors": cfg.authors,
        "email": cfg.email,
        "institution": cfg.institution,
        "country / region": cfg.country,
        "first_stage_predictions": [first_stage_output],
        "second_stage_predictions": [second_stage_output],
    }

    # pickle and save dict
    save_path = Path(cfg.output_dir)
    filename = os.path.join(save_path, "submission.pkl")
    with open(filename, "wb") as file:
        pickle.dump(submission, file)
    logger.info(f"Your submission file was saved to {filename}")


if __name__ == "__main__":
    main()
