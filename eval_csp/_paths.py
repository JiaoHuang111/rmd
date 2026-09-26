"""Path helpers shared by the eval_csp data-preparation scripts.

Two roots are resolved here, both overridable by environment variable so the
scripts stay usable from any checkout location:

  REPO_ROOT        the project root, inferred from this file's location
                   (eval_csp/_paths.py -> two levels up).
  CRYSTALLM_ROOT   a read-only CrystaLLM checkout supplying the benchmark CSVs
                   under resources/benchmarks/<ds>/{train,val,test}.csv. Those
                   CSVs are third-party data and are not vendored in this repo,
                   so the variable has to be set by the user before running any
                   script that reads them.

  EVAL_TOKENS_DIR  optional override for the tokenized corpus directory, for
                   when the tokenized data lives outside the checkout.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CRYSTALLM_ROOT = os.environ.get("CRYSTALLM_ROOT", "")


def crystallm_csv(ds, split):
    """Path of a CrystaLLM benchmark CSV, e.g. ``crystallm_csv("mp_20", "test")``."""
    if not _CRYSTALLM_ROOT:
        raise RuntimeError(
            "CRYSTALLM_ROOT is not set. Point it at a CrystaLLM checkout, e.g.\n"
            "    export CRYSTALLM_ROOT=/path/to/CrystaLLM-main\n"
            "The benchmark CSVs resources/benchmarks/<ds>/{train,val,test}.csv are "
            "read from there; they are third-party data and are not bundled here."
        )
    return os.path.join(_CRYSTALLM_ROOT, "resources", "benchmarks", ds, f"{split}.csv")


def crystallm_csvroot():
    """The benchmarks directory itself, for scripts iterating over datasets."""
    if not _CRYSTALLM_ROOT:
        raise RuntimeError(
            "CRYSTALLM_ROOT is not set. Point it at a CrystaLLM checkout, e.g.\n"
            "    export CRYSTALLM_ROOT=/path/to/CrystaLLM-main"
        )
    return os.path.join(_CRYSTALLM_ROOT, "resources", "benchmarks")


def tokens_dir(ds):
    """Directory of the tokenized corpus for a dataset (in-repo by default)."""
    override = os.environ.get("EVAL_TOKENS_DIR")
    if override:
        return override
    return os.path.join(REPO_ROOT, "data-bin", "crystalmols", f"tokens_{ds}")
