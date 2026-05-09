import json
import os
from pathlib import Path
from typing import Any


class MachineConfig:
    """Machine-specific config loaded from local_config.json.

    Fields are arbitrary — access any key from the JSON as an attribute.
    Extra keys that a caller does not use are silently ignored, so different
    experiment directories can share one file format with different field sets.

    Typical usage in an experiment script::

        from aepsych_dconfig import MachineConfig
        machine = MachineConfig.from_json(Path(__file__))
        path = machine.network_disk_path
    """

    _data: dict[str, Any]

    def __init__(self, _data: dict[str, Any]) -> None:
        self._data = _data

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"MachineConfig has no field {name!r}. Available: {list(self._data)}"
            ) from None

    @classmethod
    def from_json(
        cls,
        caller: Path | str | None = None,
        *,
        config_filename: str = "local_config.json",
    ) -> "MachineConfig":
        """Load local_config.json using a three-tier search.

        Priority:
          1. ``WPPM_CONFIG_DIR`` environment variable — use that directory.
          2. Walk up from *caller*'s directory, checking each level, stopping
             at the directory that contains ``pyproject.toml`` (the package
             root).  The package-root level is included in the search.
          3. Raise ``FileNotFoundError``.

        Parameters
        ----------
        caller:
            Pass ``Path(__file__)`` from the calling script.  The search
            starts in that script's directory and walks toward the root.
            When ``caller`` is ``None`` the walk is skipped, so only the
            environment variable (priority 1) can succeed.
        """
        # Priority 1 — environment variable
        env_dir = os.environ.get("WPPM_CONFIG_DIR")
        if env_dir is not None:
            config_path = Path(env_dir) / config_filename
            if config_path.exists():
                return cls._load(config_path)
            raise FileNotFoundError(
                f"WPPM_CONFIG_DIR={env_dir!r} but {config_filename} not found "
                f"there.\nCopy {config_filename}.template to {config_filename} "
                "and fill in your machine-specific paths."
            )

        # Priority 2 — walk up from caller's directory
        if caller is not None:
            start = Path(caller).resolve()
            if start.is_file():
                start = start.parent
            for directory in (start, *start.parents):
                config_path = directory / config_filename
                if config_path.exists():
                    return cls._load(config_path)
                if (directory / "pyproject.toml").exists():
                    raise FileNotFoundError(
                        f"No {config_filename} found between {start} and the "
                        f"package root at {directory}.\n"
                        f"Copy {config_filename}.template to {config_filename} "
                        "and fill in your machine-specific paths."
                    )

        raise FileNotFoundError(
            f"No {config_filename} found.  Either:\n"
            "  1. export WPPM_CONFIG_DIR=/path/to/experiment/dir, or\n"
            "  2. call MachineConfig.from_json(Path(__file__)) from a script "
            "inside the experiment directory tree."
        )

    @classmethod
    def _load(cls, path: Path) -> "MachineConfig":
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        return cls(data)
