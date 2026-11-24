"""EnergyPlus Gym wrapper entry points.

EnergyPlus is an external runtime, so importing :mod:`building_gen.wrapper`
must not require ``pyenergyplus`` to be installed.  Runtime-dependent objects
are loaded only when requested.
"""

# CRITICAL: Setup EnergyPlus path BEFORE importing modules that depend on pyenergyplus
# This must be done first, before any imports that transitively import pyenergyplus
import os
import sys
import logging
from typing import TYPE_CHECKING

def setup_energyplus_path():
    """Setup EnergyPlus path from environment variable - must be called before pyenergyplus imports"""
    logger = logging.getLogger(__name__)

    if 'ENERGYPLUS_PATH' in os.environ:
        energyplus_path = os.environ['ENERGYPLUS_PATH']
        if os.path.exists(energyplus_path):
            # Add to sys.path for Python imports - THIS IS THE KEY FIX
            if energyplus_path not in sys.path:
                sys.path.append(energyplus_path)
                logger.debug(f"Added EnergyPlus to sys.path: {energyplus_path}")

            # Also add to PATH for executable lookup
            current_path = os.environ.get('PATH', '')
            if energyplus_path not in current_path:
                os.environ['PATH'] = f"{energyplus_path}:{current_path}"
                logger.debug(f"Added EnergyPlus to PATH: {energyplus_path}")
        else:
            logger.warning(f"EnergyPlus path does not exist: {energyplus_path}")
    else:
        logger.debug("ENERGYPLUS_PATH environment variable not set")
        # Try to find EnergyPlus in common container locations
        container_paths = [
            "/usr/local/EnergyPlus-24-1-0",
            "/usr/local/EnergyPlus-24-2-0",
            "/usr/local/EnergyPlus-23-2-0",
            "/opt/EnergyPlus-24-1-0"
        ]

        for path in container_paths:
            if os.path.exists(path):
                if path not in sys.path:
                    sys.path.append(path)
                    logger.info(f"Found and added EnergyPlus to sys.path: {path}")
                    os.environ['ENERGYPLUS_PATH'] = path
                    break

# Make a configured EnergyPlus installation discoverable, without importing it.
setup_energyplus_path()

if TYPE_CHECKING:
    from .config import EnergyPlusConfig
    from .create_simulator import create_simulator
    from .energyplus_env import EnergyPlusEnvironment
    from .ppo_trainer import PPOArgs, main, ppo_evaluate


def __getattr__(name: str):
    """Load optional Gymnasium, Torch, and EnergyPlus components on demand."""

    if name == "EnergyPlusConfig":
        from .config import EnergyPlusConfig

        return EnergyPlusConfig
    if name == "EnergyPlusEnvironment":
        from .energyplus_env import EnergyPlusEnvironment

        return EnergyPlusEnvironment
    if name == "create_simulator":
        from .create_simulator import create_simulator

        return create_simulator
    if name in {"PPOArgs", "main", "ppo_evaluate"}:
        from . import ppo_trainer

        return getattr(ppo_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
__all__ = [
    "EnergyPlusEnvironment",
    "EnergyPlusConfig",
    "PPOArgs",
    "main",
    "ppo_evaluate",
    "create_simulator",
    "setup_energyplus_path",
]
