# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


#!python

import os
import sys
from pathlib import Path

# ------------------------------------------------------------------------------------ #
# Project root resolution -- identical to the one in train.py, see there for the
# rationale. In short: this checkout is standalone (no .git / pyproject.toml), so
# pyrootutils.setup_root() cannot find the root; the directory containing this file
# is used instead, and an externally exported PROJECT_ROOT is respected.
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
# Make sure the byprot imported below is the one in this checkout.
for _p in [root / "src"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
print(f"Project root: {root}")

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402


@hydra.main(config_path=f"{root}/configs", config_name="test.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from byprot import utils
    from byprot.testing_pipeline import test

    # resolve user provided config
    config = utils.resolve_experiment_config(config)
    # Applies optional utilities
    config = utils.extras(config)

    # Evaluate model
    return test(config)


if __name__ == "__main__":
    main()
