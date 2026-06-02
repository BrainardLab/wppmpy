import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AepsychIniConfig:
    """Experiment parameters parsed from the AEPsych .ini config file.

    Use ``from_expt_dir`` to locate the ini automatically from any script
    that lives in the same ``<proj>/<expt>/`` hierarchy as the ini.
    """

    lower_bound: float
    upper_bound: float
    eavc_target: float

    @property
    def grid_half_extent(self) -> float:
        """Display grid extent — slightly wider than the search bounds."""
        return self.upper_bound + 0.05

    @classmethod
    def from_path(cls, ini_path: Path) -> "AepsychIniConfig":
        cfg = configparser.ConfigParser()
        cfg.read(ini_path)
        return cls(
            lower_bound=float(cfg["delta_dim1"]["lower_bound"]),
            upper_bound=float(cfg["delta_dim1"]["upper_bound"]),
            eavc_target=float(cfg["EAVC"]["target"]),
        )

    @classmethod
    def from_expt_dir(cls, caller: Path, ini_filename: str) -> "AepsychIniConfig":
        """Locate the ini from any script in ``aepsych/<proj>/<expt>/`` or the
        mirrored ``analysis/<proj>/<expt>/`` tree, using the shared hierarchy."""
        repo_root = caller.resolve()
        # Walk up until we find the .git directory (repo root)
        for parent in (repo_root, *repo_root.parents):
            if (parent / ".git").exists():
                repo_root = parent
                break
        proj = caller.resolve().parent.parent.name  # e.g. "wppmopl"
        expt = caller.resolve().parent.name  # e.g. "ellipsoid3d"
        ini_path = repo_root / "aepsych" / proj / expt / "aepsych_config" / ini_filename
        if not ini_path.exists():
            raise FileNotFoundError(
                f"AEPsych ini not found at {ini_path}.\n"
                f"Check aepsych_config_file in local_config.json."
            )
        return cls.from_path(ini_path)
