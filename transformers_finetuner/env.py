import dataclasses
from dataclasses import dataclass, field
from json import dump
from pathlib import Path
import platform as pltfrm
from typing import Optional

import git
from git.exc import InvalidGitRepositoryError
import torch
import transformers
from torch import cuda


@dataclass
class Env:
    platform: str = pltfrm.platform()
    transformers_version: str = transformers.__version__
    python_version: str = pltfrm.python_version()
    toch_version: str = torch.__version__

    cuda_device_count: int = cuda.device_count() if cuda.is_available() else 0
    cuda_devices: Optional[str] = None
    cuda_capabilities: Optional[str] = None
    finetuner_commit: Optional[str] = None

    def __post_init__(self):
        if self.cuda_device_count:
            self.cuda_devices = "; ".join([cuda.get_device_name(i) for i in range(self.cuda_device_count)])
            self.cuda_capabilities = "; ".join([".".join(map(str, cuda.get_device_capability(i)))
                                                for i in range(self.cuda_device_count)])

        try:
            repo = git.Repo(path="../")
            self.finetuner_commit = repo.head.object.hexsha
        except InvalidGitRepositoryError:
            self.finetuner_commit = None

    @classmethod
    def dump(cls, output_file):
        env = cls()
        with Path(output_file).open("w", encoding="utf-8") as env_out:
            dump(dataclasses.asdict(env), env_out, indent=4, sort_keys=True)
