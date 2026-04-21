from dataclasses import dataclass, field

from aepsych_dconfig.config_pregenSobol import PregenSobolConfig


@dataclass
class ExptConfig(PregenSobolConfig):
    # AEPsych experiment structure (live-session trial counts)
    nTrials_mocs_perSession: int = 500
    nTrials_aepsych_perSession: int = 500
    nTrials_strat: list[int] = field(default_factory=lambda: [300, 300, 300, 6600])
    # Sobol scalers used during live AEPsych trials (distinct from sobol_scaler,
    # which governs pre-generation of MOCS stimuli)
    sobol_scaler_live: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0]
    )
    shuffle_sobol_scaler_max_strat: int = 3

    @classmethod
    def isoluminant_4d(cls) -> "ExptConfig":
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D="Isoluminant plane",
            file_date="02242025",
            nSessions=15,
            nTrials_sobol_perSession=1200,
            lb_sobol_trials=[-0.75, -0.75, -0.25, -0.25],
            ub_sobol_trials=[0.75, 0.75, 0.25, 0.25],
            sobol_scaler=[2 / 8, 3 / 8, 4 / 8],
            flag_addCatchTrials=False,
            nTrials_mocs_perSession=500,
            nTrials_aepsych_perSession=500,
            nTrials_strat=[300, 300, 300, 6600],
            sobol_scaler_live=[0.25, 0.5, 0.75, 1.0],
            shuffle_sobol_scaler_max_strat=3,
        )

    @classmethod
    def practice_isoluminant_4d(cls) -> "ExptConfig":
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D="Isoluminant plane",
            file_date="02242025",
            nSessions=1,
            nTrials_sobol_perSession=1200,
            lb_sobol_trials=[-0.75, -0.75, -0.25, -0.25],
            ub_sobol_trials=[0.75, 0.75, 0.25, 0.25],
            sobol_scaler=[2 / 8, 3 / 8, 4 / 8],
            flag_addCatchTrials=False,
            nTrials_mocs_perSession=20,
            nTrials_aepsych_perSession=20,
            nTrials_strat=[500, 500, 500, 1],
            sobol_scaler_live=[0.25, 0.5, 0.75, 1.0],
            shuffle_sobol_scaler_max_strat=3,
        )
