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
    launch(config, get_original_cwd(), native_slurm=HydraConfig.get().mode == RunMode.MULTIRUN, hydra_run_dir=HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
