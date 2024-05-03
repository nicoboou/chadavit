import os

from argparse import Namespace
from typing import Any, Dict, Mapping, Optional, Union

import wandb
from lightning_fabric.utilities.logger import (
    _add_prefix,
    _convert_params,
    _flatten_dict,
    _sanitize_callable_params,
)
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only


class SLURMLogger(Logger):
    """
    Log to local file system in SLURM environment.

    Args:

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = True,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        prefix: str = "",
        entity: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `SLURMLogger` which will log all logs into `wandb` at the end of training at once, but `wandb` is not installed yet,"
                " install it with `pip install wandb`."  # pragma: no-cover
            )

        super().__init__()
        self._offline = offline
        self._prefix = prefix
        self._logged_model_time: Dict[str, float] = {}
        self._version = version
        self._entity = entity

        # paths are processed as strings
        if save_dir is not None:
            save_dir = os.fspath(save_dir)

        project = project or os.environ.get("WANDB_PROJECT", "lightning_logs")

        # set wandb init arguments
        self._wandb_init: Dict[str, Any] = {
            "name": name,
            "entity": entity,
            "project": project,
            "id": version or id,
            "mode": "offline" if offline else "online",
            "resume": "allow",
            "anonymous": ("allow" if anonymous else None),
        }
        self._wandb_init.update(**kwargs)

        # extract parameters
        self._project = self._wandb_init.get("project")
        self._save_dir = save_dir
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def version(self):
        # Return the experiment version, int or str.
        return self._version

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        self.hyperparams = params

    @rank_zero_only
    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        print("LOGGING METRICS")
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        # Write to file for logging, necessary for SLURM offline WandB logging
        print(f"current dir: {os.getcwd()}")
        print(f"saving dir: {self._save_dir}")
        with open(self._save_dir + "/training_logs.txt", "a+") as f:
            if step is not None:
                f.write(str(dict(metrics, **{"trainer/global_step": step})) + "\n")
            else:
                f.write(str(metrics) + "\n")

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status) -> None:
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
