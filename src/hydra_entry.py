"""Hydra CLI for dataset jobs. Select hardware with launcher=<profile>."""


from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from hydra_run import launch


@hydra.main(version_base="1.3", config_path=str(Path(__file__).resolve().parents[1] / "config"), config_name="config")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    if config["stage"]["name"] == "extract_mineru":
        hydra_launcher = HydraConfig.get().launcher
        allocated_gpus = int(hydra_launcher.get("gpus_per_node", 0))
        if allocated_gpus < 1:
            raise ValueError("extract_mineru requires Hydra Submitit GPU resources (hydra.launcher.gpus_per_node>=1)")
        if allocated_gpus != int(config["launcher"]["gpus"]):
            raise ValueError("launcher.gpus and hydra.launcher.gpus_per_node must match for extract_mineru")
    launch(config, get_original_cwd(), native_slurm=HydraConfig.get().mode == RunMode.MULTIRUN, hydra_run_dir=HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
