"""Cache with the original NAVSIM v1 fields; separate from the v2 cache."""
import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.io_utils import save_buffer
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


@dataclass
class MetricCache:
    file_path: Path
    trajectory: InterpolatedTrajectory
    ego_state: EgoState
    observation: PDMObservation
    centerline: PDMPath
    route_lane_ids: List[str]
    drivable_area_map: PDMDrivableMap

    def dump(self) -> None:
        save_buffer(self.file_path, lzma.compress(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL), preset=0))
