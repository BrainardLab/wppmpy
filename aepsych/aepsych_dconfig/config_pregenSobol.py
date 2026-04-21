#!/usr/bin/env python3
"""
Created on Sun Mar 22 21:26:30 2026

@author: fangfang

Configuration class for pre-generating Sobol trials used in color
discrimination experiments.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PregenSobolConfig:
    # dimensions
    stim_dims: int
    psyfield_dims: int

    # number of sessions and trials
    nSessions: int
    nTrials_sobol_perSession: int

    # optional metadata
    plane_2D: str | None = None
    file_date: str | None = None

    # sampling settings
    lb_sobol_trials: Sequence[float] | None = None
    ub_sobol_trials: Sequence[float] | None = None
    sobol_scaler: Sequence[float] | None = None

    # experiment structure
    flag_addCatchTrials: bool = False

    # optional catch trials
    delta_catchTrials_unique: np.ndarray | None = None
    percent_catchTrials: float | None = None

    # derived: number of repetitions of sobol_scaler needed to fill all trials
    num_repeats: int = field(init=False)

    def __post_init__(self) -> None:
        if self.sobol_scaler is None:
            raise ValueError("sobol_scaler must be provided")
        if self.nTrials_sobol_perSession % len(self.sobol_scaler) != 0:
            raise ValueError("nTrials must be multiple of sobol_scaler length")
        self.num_repeats = self.nTrials_sobol_perSession // len(self.sobol_scaler)
        if self.flag_addCatchTrials and self.delta_catchTrials_unique is None:
            raise ValueError(
                "Catch trials enabled but delta_catchTrials_unique not provided"
            )

    # --------------------------------------------------
    # factory constructors
    # --------------------------------------------------

    @classmethod
    def isoluminant_2D4D(cls) -> "PregenSobolConfig":
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D="Isoluminant plane",
            file_date="02242025",
            nTrials_sobol_perSession=1200,
            lb_sobol_trials=[-0.75, -0.75, -0.25, -0.25],
            ub_sobol_trials=[0.75, 0.75, 0.25, 0.25],
            sobol_scaler=[2 / 8, 3 / 8, 4 / 8],
            flag_addCatchTrials=False,
            nSessions=15,
        )

    @classmethod
    def rgbcube_3D_dichromat(cls) -> "PregenSobolConfig":
        return cls(
            stim_dims=3,
            psyfield_dims=3,
            nTrials_sobol_perSession=900,
            lb_sobol_trials=[-1, -1, -1 / 3],
            ub_sobol_trials=[1, 1, 1 / 3],
            sobol_scaler=[0.15, 0.45, 0.75],
            flag_addCatchTrials=False,
            nSessions=5,
        )

    @classmethod
    def LSisolating_dichromat(cls) -> "PregenSobolConfig":
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D="LSisolating plane",
            file_date="11172025",
            nTrials_sobol_perSession=2400,
            lb_sobol_trials=[-0.75, -0.75, -0.25, -0.25],
            ub_sobol_trials=[0.75, 0.75, 0.25, 0.25],
            sobol_scaler=[2 / 8, 3 / 8, 4 / 8],
            flag_addCatchTrials=True,
            percent_catchTrials=0.05,
            nSessions=30,
            delta_catchTrials_unique=np.array(
                [[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]]
            ),
        )

    @classmethod
    def LSisolating_dichromat_expanded(cls) -> "PregenSobolConfig":
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D="LSisolating plane",
            file_date="11172025",
            nTrials_sobol_perSession=2400,
            lb_sobol_trials=[-0.55, -0.7, -0.45, -0.3],
            ub_sobol_trials=[0.55, 0.7, 0.45, 0.3],
            sobol_scaler=[4 / 8, 6 / 8, 1],
            flag_addCatchTrials=True,
            percent_catchTrials=0.05,
            nSessions=30,
            delta_catchTrials_unique=np.array(
                [[-0.45, -0.3], [-0.45, 0.3], [0.45, -0.3], [0.45, 0.3]]
            ),
        )

    @classmethod
    def adaptation_round1(cls) -> "PregenSobolConfig":
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D="Isoluminant plane",
            file_date="10062025",
            nTrials_sobol_perSession=2400,
            lb_sobol_trials=[-0.75, -0.75, -0.25, -0.25],
            ub_sobol_trials=[0.75, 0.75, 0.25, 0.25],
            sobol_scaler=[2 / 8, 3 / 8, 4 / 8],
            flag_addCatchTrials=True,
            percent_catchTrials=0.05,
            nSessions=30,
            delta_catchTrials_unique=np.array(
                [[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]]
            ),
        )

    @classmethod
    def adaptation_round2(cls) -> "PregenSobolConfig":
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D="Isoluminant plane",
            file_date="02012026",
            nTrials_sobol_perSession=900,
            lb_sobol_trials=[-0.75, -0.75, -0.25, -0.25],
            ub_sobol_trials=[0.75, 0.75, 0.25, 0.25],
            sobol_scaler=[2 / 8, 3 / 8, 4 / 8],
            flag_addCatchTrials=True,
            percent_catchTrials=0.05,
            nSessions=40,
            delta_catchTrials_unique=np.array(
                [[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]]
            ),
        )

    def print_summary(self) -> None:
        print("---- Sobol Sampling Config ----")
        print(f"stim_dims                : {self.stim_dims}")
        print(f"psyfield_dims            : {self.psyfield_dims}")
        print(f"nTrials_sobol_perSession : {self.nTrials_sobol_perSession}")
        print(f"lb_sobol_trials          : {self.lb_sobol_trials}")
        print(f"ub_sobol_trials          : {self.ub_sobol_trials}")
        print(f"sobol_scaler             : {self.sobol_scaler}")
        print(f"flag_addCatchTrials      : {self.flag_addCatchTrials}")
        print(f"nSessions                : {self.nSessions}")
        if self.flag_addCatchTrials and self.delta_catchTrials_unique is not None:
            print("delta_catchTrials_unique :")
            print(self.delta_catchTrials_unique)
        else:
            print("delta_catchTrials_unique : None")
        print("--------------------------------")

    def to_legacy_dict(self) -> dict:
        return {
            "nTrials_sobol_perSession": self.nTrials_sobol_perSession,
            "lb_sobol_trials": self.lb_sobol_trials,
            "ub_sobol_trials": self.ub_sobol_trials,
            "sobol_scaler": self.sobol_scaler,
        }
