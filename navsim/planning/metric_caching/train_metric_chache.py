from __future__ import annotations

import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.io_utils import save_buffer
from shapely.geometry import LineString

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap


@dataclass
class MetricCache:
    """Dataclass for storing metric computation information."""

    file_path: Path
    # trajectory: InterpolatedTrajectory
    ego_state: EgoState

    observation: PDMObservation
    centerline: LineString
    route_lane_ids: List[str]
    drivable_area_map: PDMDrivableMap
    pdm_progress: np.array

    def dump(self) -> None:
        """Dump metric cache to pickle with lzma compression."""
        # TODO: check if file_path must really be pickled
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        save_buffer(self.file_path, lzma.compress(pickle_object, preset=0))
