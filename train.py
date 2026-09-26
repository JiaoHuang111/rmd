# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


#!python

import os
import sys
from pathlib import Path

# ------------------------------------------------------------------------------------ #
# Project root resolution.
#
# This checkout is standalone: it has no .git / pyproject.toml, so
# pyrootutils.setup_root() would raise FileNotFoundError, and upstream's
# hardcoded fallback points outside this repository.
#
# Therefore: project root = the directory containing this file. An externally
# exported PROJECT_ROOT (e.g. to keep data and logs on another disk) is
# respected. The value feeds ${oc.env:PROJECT_ROOT} in
# "configs/paths/default.yaml", which determines the log / checkpoint output
# location.
# ------------------------------------------------------------------------------------ #

root = Path(__file__).resolve().parent
# Load variables from a .env file at the repository root, if one exists. This is
# where machine-specific settings belong; .env itself is gitignored, see
# .env.example for the recognised keys.
try:
    from dotenv import load_dotenv

    load_dotenv(root / ".env")
except ImportError:
    pass
os.environ.setdefault("PROJECT_ROOT", str(root))
# Make sure the byprot imported below is the one in this checkout. Without this,
# an editable install of a differently-located byprot can take precedence.
for _p in [root / "src"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
print(f"Project root: {root}")

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base="1.1",
    config_path=f"{root}/configs",
    config_name="config.yaml",
)
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from byprot import utils
    from byprot.training_pipeline import train

    # Applies optional utilities
    config = utils.extras(config)

    # Train model
    print('Function <main> success!')
    return train(config)


if __name__ == "__main__":
    main()
