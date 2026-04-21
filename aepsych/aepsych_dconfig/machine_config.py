import json
import os
from dataclasses import dataclass


@dataclass
class MachineConfig:
    network_disk_path: str
    stim_at_thres_path: str
    color_thres_base_dir: str
    flag_load_rgb: bool

    @classmethod
    def from_json(cls, path: str | None = None) -> "MachineConfig":
        if path is None:
            path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "local_config.json")
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Local config not found: {path}\n"
                "Copy local_config.json.template to local_config.json and fill in "
                "paths."
            )
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
