"""Custom OmegaConf resolvers used in Hydra YAML configs.

Call :func:`register_resolvers` once at module top of every ``runs/*.py``
script before any config interpolation is resolved.
"""

import math

from omegaconf import OmegaConf


def register_resolvers() -> None:
    """Register all custom resolvers used in YAML configs (idempotent)."""

    OmegaConf.register_new_resolver("prod", math.prod, replace=True)
    OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
    OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
