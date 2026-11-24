"""
Action and observation space definitions for EnergyPlus environment
Compatible with template-based eplus_interface.py
"""

import gymnasium as gym
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from building_gen.accounting import JOULES_PER_KWH
from building_gen.wrapper.observation_encoding import (
    OBSERVATION_SCHEMA_ID,
    encode_observation_value,
)
from .config import EnergyPlusConfig


class CustomRescaleAction(gym.wrappers.RescaleAction):
    def __init__(self, env, min_action, max_action):
        """
        Initialize the CustomRescaleAction wrapper.

        Args:
            env: The environment to wrap
            min_action: The minimum value of the new action space
            max_action: The maximum value of the new action space
        """
        import logging
        logger = logging.getLogger("spaces")

        super().__init__(env, min_action, max_action)
        self.min_action = min_action
        self.max_action = max_action

        # Get the raw bounds from the environment's action space if available
        if hasattr(env, 'action_space') and hasattr(env.action_space, 'raw_low'):
            self.raw_low = env.action_space.raw_low
            self.raw_high = env.action_space.raw_high
            logger.info(f"CustomRescaleAction: Found raw bounds - low: {self.raw_low[:4]}, high: {self.raw_high[:4]}")
        else:
            # Fallback to the provided bounds
            self.raw_low = min_action
            self.raw_high = max_action
            logger.warning(f"CustomRescaleAction: Using fallback bounds - low: {self.raw_low[:4]}, high: {self.raw_high[:4]}")

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Scale an action from the normalized space (min_action to max_action)
        to the environment's original action space and clip it to stay within bounds.

        Args:
            action: Action in the normalized space (min_action to max_action)

        Returns:
            Action scaled to the environment's original action space and clipped
        """
        import logging
        logger = logging.getLogger("spaces")

        logger.info(f"=== ACTION SCALING DEBUG (EVERY STEP) ===")
        logger.info(f"Input action shape: {action.shape}")
        logger.info(f"Input action bounds: [{action.min():.3f}, {action.max():.3f}]")
        logger.info(f"Input action values: {action}")
        logger.info(f"Action wrapper bounds: min_action={self.min_action}, max_action={self.max_action}")
        logger.info(f"Raw temperature bounds: low={self.raw_low}, high={self.raw_high}")

        # Scale from [min_action, max_action] to [0, 1]
        norm_action = (action - self.min_action) / (self.max_action - self.min_action)
        logger.info(f"After normalization to [0,1]: bounds=[{norm_action.min():.3f}, {norm_action.max():.3f}]")
        logger.info(f"Normalized action values: {norm_action}")

        # Scale from [0, 1] to raw temperature bounds
        scaled_action = (
            norm_action * (self.raw_high - self.raw_low)
            + self.raw_low
        )
        logger.info(f"After scaling to raw bounds: bounds=[{scaled_action.min():.3f}, {scaled_action.max():.3f}]")
        logger.info(f"Scaled action values: {scaled_action}")

        # Clip the action to ensure it stays within the raw bounds
        clipped_action = np.clip(
            scaled_action, self.raw_low, self.raw_high
        )
        logger.info(f"After clipping: bounds=[{clipped_action.min():.3f}, {clipped_action.max():.3f}]")
        logger.info(f"Final clipped action values: {clipped_action}")

        # Show step-by-step calculation for each value
        for i in range(len(action)):
            logger.info(f"Value {i}: {action[i]:.3f} -> {norm_action[i]:.3f} -> {scaled_action[i]:.3f} -> {clipped_action[i]:.3f} (raw_low={self.raw_low[i]:.1f}, raw_high={self.raw_high[i]:.1f})")

        logger.info(f"=== END ACTION SCALING DEBUG ===")

        return clipped_action


class NormalizeObservation(gym.ObservationWrapper):
    """
    Custom observation wrapper that normalizes observations to a [0, 1] range according to the observation space bounds.
    Compatible with template-based observations.
    """

    def __init__(self, env: gym.Env, dtype: np.dtype = np.float64):
        """
        Args:
            env: The environment to wrap
            dtype: The dtype of the observation space
        """
        super().__init__(env)

        # Ensure the observation space is a Box
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"Expected observation space to be Box, got {type(env.observation_space)}"
            )

        # Store the original observation space bounds
        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high

        # Handle infinite bounds by replacing them with large finite values
        self.obs_low = np.where(np.isinf(self.obs_low), -1e10, self.obs_low)
        self.obs_high = np.where(np.isinf(self.obs_high), 1e10, self.obs_high)

        # Calculate the range of the observation space
        self.obs_range = self.obs_high - self.obs_low
        # Check for zero ranges which would cause division by zero
        zero_ranges = np.where(self.obs_range == 0)[0]
        if len(zero_ranges) > 0:
            raise ValueError(f"Observation space has zero range for dimensions {zero_ranges}. This would cause division by zero during normalization.")

        # Set the new observation space to be in the target range
        target_low = np.zeros(self.obs_low.shape, dtype=dtype)
        target_high = np.ones(self.obs_high.shape, dtype=dtype)

        self.observation_space = gym.spaces.Box(
            low=target_low,
            high=target_high,
            dtype=dtype
        )

    def observation(self, observation: Any) -> np.ndarray:
        """
        Normalize the observation to [0, 1] range.

        Args:
            observation: The observation to normalize

        Returns:
            The normalized observation
        """
        import logging
        logger = logging.getLogger("spaces")

        observation = np.asarray(observation, dtype=self.observation_space.dtype)

        logger.info(f"=== NORMALIZE OBSERVATION DEBUG ===")
        logger.info(f"Input observation shape: {observation.shape}")
        logger.info(f"Input observation bounds: [{observation.min():.3f}, {observation.max():.3f}]")
        logger.info(f"Input observation (first 10): {observation[:10]}")
        logger.info(f"Observation space bounds: low={self.obs_low.min():.3f}, high={self.obs_high.max():.3f}")
        logger.info(f"Observation range: {self.obs_range[:10]}")

        normalized_observation = (observation - self.obs_low) / self.obs_range

        logger.info(f"Normalized observation bounds: [{normalized_observation.min():.3f}, {normalized_observation.max():.3f}]")
        logger.info(f"Normalized observation (first 10): {normalized_observation[:10]}")
        logger.info(f"=== END NORMALIZE OBSERVATION DEBUG ===")

        return normalized_observation

    def denormalize(self, observation: np.ndarray) -> np.ndarray:
        """
        Denormalize the observation back to original range.

        Args:
            observation: The normalized observation

        Returns:
            The denormalized observation
        """
        return observation * self.obs_range + self.obs_low


class ActionRescaleWrapper(gym.ActionWrapper):
    """
    Custom action wrapper that rescales normalized [0,1] actions to real temperature values.
    Compatible with template-based interface.
    """

    def __init__(self, env, action_space_handler=None, normalized_min=0.0, normalized_max=1.0):
        """
        Initialize the action rescaling wrapper.

        Args:
            env: The environment to wrap
            action_space_handler: ActionSpace instance for temperature conversion (optional for template-based envs)
            normalized_min: Minimum value of normalized action space (default: 0.0)
            normalized_max: Maximum value of normalized action space (default: 1.0)
        """
        super().__init__(env)

        # Handle both old and new environment structures
        if action_space_handler is not None:
            # Old structure with action_space_handler
            self.action_space_obj = action_space_handler
            self.config = action_space_handler.config
        else:
            # New template-based structure - create config from environment
            self.action_space_obj = None
            # Create a minimal config for temperature bounds
            class TempConfig:
                def __init__(self):
                    self.heating_setpoint_range = (18.0, 23.0)
                    self.offset_range = (2.0, 4.0)
                    self.cooling_setpoint_range = (20.0, 27.0)
            self.config = TempConfig()

        self.norm_min = normalized_min
        self.norm_max = normalized_max

        # Store original bounds for conversion
        self.original_low = env.action_space.low.copy()
        self.original_high = env.action_space.high.copy()

        # Update environment's action space to [0,1] for RL agent
        new_low = np.full_like(env.action_space.low, normalized_min)
        new_high = np.full_like(env.action_space.high, normalized_max)
        self.action_space = gym.spaces.Box(
            low=new_low,
            high=new_high,
            dtype=env.action_space.dtype
        )

    def action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Convert normalized [0,1] actions to real temperature values.

        Args:
            normalized_action: Action in normalized space [0,1]

        Returns:
            Action scaled to real temperature values and clipped to valid bounds
        """
        # Clip to normalized range
        normalized_action = np.clip(normalized_action, self.norm_min, self.norm_max)

        # Scale from [0,1] to original action space bounds
        real_action = normalized_action * (self.original_high - self.original_low) + self.original_low

        # Use config bounds for final clipping
        for i in range(0, len(real_action), 2):
            # Heating setpoint - use config bounds
            heating_min, heating_max = self.config.heating_setpoint_range
            real_action[i] = np.clip(real_action[i], heating_min, heating_max)

            # Offset - use config bounds
            offset_min, offset_max = self.config.offset_range
            real_action[i + 1] = np.clip(real_action[i + 1], offset_min, offset_max)

        return real_action

    def reverse_action(self, real_action: np.ndarray) -> np.ndarray:
        """
        Convert real temperature values back to normalized [0,1] actions.

        Args:
            real_action: Action in real temperature values

        Returns:
            Action in normalized [0,1] range
        """
        # Scale from real bounds to [0,1]
        normalized_action = (real_action - self.original_low) / (self.original_high - self.original_low)
        normalized_action = normalized_action * (self.norm_max - self.norm_min) + self.norm_min

        return np.clip(normalized_action, self.norm_min, self.norm_max)


class ActionSpace:
    """
    Defines action space for setpoint control using offset approach
    Compatible with template-based interface
    """

    def __init__(self, config: EnergyPlusConfig, zone_names: List[str]):
        self.config = config
        self.zone_names = zone_names
        self.n_zones = len(zone_names)
        self.channel_names = tuple(
            getattr(
                config,
                "action_channel_names",
                ("heating_setpoint_c", "cooling_offset_c"),
            )
        )
        if self.channel_names != ("heating_setpoint_c", "cooling_offset_c"):
            raise ValueError(
                "ActionSpace requires channels ordered as "
                "('heating_setpoint_c', 'cooling_offset_c')"
            )
        self.transform_id = getattr(
            config,
            "action_transform_id",
            "linear_quantized_0_1c_legacy",
        )
        supported_transforms = {
            "linear_quantized_0_1c_legacy",
            "linear_unquantized_v1",
            "bounded_thermostat_15_30_v1",
            "bounded_thermostat_12_32_v1",
        }
        if self.transform_id not in supported_transforms:
            raise ValueError(
                f"unsupported action transform {self.transform_id!r}; "
                f"expected one of {sorted(supported_transforms)!r}"
            )

        # Validate required config attributes
        if not hasattr(config, 'heating_setpoint_range'):
            raise ValueError("Config missing required attribute 'heating_setpoint_range'")

        if not hasattr(config, 'offset_range'):
            raise ValueError("Config missing required attribute 'offset_range'")

        if self.transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1"):
            from building_gen.bounded_thermostat import validate_ranges
            validate_ranges(config.heating_setpoint_range,config.offset_range,self.transform_id)
        if self.transform_id == "linear_quantized_0_1c_legacy":
            bounds = (*config.heating_setpoint_range, *config.offset_range)
            if any(abs(value * 10.0 - round(value * 10.0)) > 1e-9 for value in bounds):
                raise ValueError(
                    "legacy tenth-degree action bounds must be aligned to 0.1 C"
                )

        if not hasattr(config, 'cooling_setpoint_range'):
            # Calculate based on heating + offset ranges
            max_cooling = config.heating_setpoint_range[1] + config.offset_range[1]
            min_cooling = config.heating_setpoint_range[0] + config.offset_range[0]
            config.cooling_setpoint_range = (min_cooling, max_cooling)

        # Action space: [heating_setpoint_1, offset_1, ..., heating_setpoint_n, offset_n]
        action_dim = self.n_zones * 2

        # Use config bounds consistently
        low = []
        high = []

        for _ in zone_names:
            # Heating setpoint bounds from config
            heating_min, heating_max = config.heating_setpoint_range
            low.append(heating_min)
            high.append(heating_max)

            # Offset bounds from config
            offset_min, offset_max = config.offset_range
            low.append(offset_min)
            high.append(offset_max)

        low_array = np.array(low, dtype=np.float32)
        high_array = np.array(high, dtype=np.float32)

        # Create normalized action space (0-1) for the agent
        # The CustomRescaleAction wrapper will handle conversion to raw temperatures
        normalized_low = np.zeros(action_dim, dtype=np.float32)
        normalized_high = np.ones(action_dim, dtype=np.float32)

        self.gym_space = gym.spaces.Box(
            low=normalized_low,
            high=normalized_high,
            dtype=np.float32
        )

        # Store the raw bounds for the CustomRescaleAction wrapper
        self.raw_low = low_array
        self.raw_high = high_array
        self.action_names = tuple(
            f"{zone_name}:{channel_name}"
            for zone_name in self.zone_names
            for channel_name in self.channel_names
        )
        channel_bounds = (
            tuple(float(value) for value in config.heating_setpoint_range),
            tuple(float(value) for value in config.offset_range),
        )
        self.endpoint_mapping = tuple(
            (
                f"{zone_name}:{channel_name}",
                ((0.0, physical_bounds[0]), (1.0, physical_bounds[1])),
            )
            for zone_name in self.zone_names
            for channel_name, physical_bounds in zip(
                self.channel_names,
                channel_bounds,
            )
        )

        if self.transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1"):
            # Coupled channels have no independent physical endpoint mapping.
            self.endpoint_mapping = ()
        # Attach raw bounds to the gym space so CustomRescaleAction can find them
        self.gym_space.raw_low = low_array
        self.gym_space.raw_high = high_array

    def parse_action(self, action: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Parse normalized [0,1] actions into real temperature setpoints with direct scaling.
        This bypasses wrapper complexity by doing the scaling here directly.
        """
        import logging
        logger = logging.getLogger("spaces")

        action = np.asarray(action)
        expected_size = len(self.zone_names) * 2
        if action.shape != (expected_size,):
            raise ValueError(
                f"Expected a flat ({expected_size},) action for "
                f"{len(self.zone_names)} zones, got shape {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Normalized actions must be finite")
        if np.any(action < 0.0) or np.any(action > 1.0):
            raise ValueError("Normalized actions must lie within [0, 1]")

        # DIRECT SCALING: Convert [0,1] actions to real temperature ranges
        heating_min, heating_max = self.config.heating_setpoint_range  # [18, 23]
        offset_min, offset_max = self.config.offset_range              # [2, 4]

        setpoints = {}

        for i, zone_name in enumerate(self.zone_names):
            # Get normalized actions [0,1]
            norm_heating = action[i * 2]      # Normalized heating setpoint
            norm_offset = action[i * 2 + 1]   # Normalized offset
            if self.transform_id == "linear_unquantized_v1":
                # Match the declared transform's Python-float arithmetic.
                # NumPy float32 intermediates can accumulate enough rounding
                # error to fail the independently checked cooling setpoint.
                norm_heating = float(norm_heating)
                norm_offset = float(norm_offset)

            # Scale to real temperature ranges
            heating_setpoint = heating_min + norm_heating * (heating_max - heating_min)  # [0,1] → [18,23]°C
            offset = offset_min + norm_offset * (offset_max - offset_min)                # [0,1] → [2,4]°C

            if self.transform_id == "linear_quantized_0_1c_legacy":
                heating_setpoint = round(heating_setpoint, 1)
                offset = round(offset, 1)

            # Calculate cooling setpoint
            cooling_setpoint = heating_setpoint + offset
            if self.transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1"):
                from building_gen.bounded_thermostat import decode
                heating_setpoint,cooling_setpoint=decode(float(norm_heating),float(norm_offset),self.transform_id)

            setpoints[zone_name] = (heating_setpoint, cooling_setpoint)

        logger.debug("Parsed thermostat setpoints: %r", setpoints)

        return setpoints

class ObservationSpace:
    """
    Defines observation space for environment state
    Compatible with template-based interface observations
    """

    def __init__(self, config: EnergyPlusConfig, zone_names: List[str]):
        self.config = config
        self.zone_names = zone_names
        self.n_zones = len(zone_names)
        self.encoding_id = OBSERVATION_SCHEMA_ID

        # Build observation keys based on what template interface produces
        self.observation_keys = self._get_observation_keys()

        # Calculate observation dimension
        obs_dim = len(self.observation_keys)

        # ObservationSpace emits the versioned policy encoding directly.
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32)

        self.gym_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

    def _get_observation_keys(self) -> List[str]:
        """
        Get ordered list of observation keys that match template interface output.
        Template interface produces flattened observations with these key patterns.
        """
        keys = []

        # Zone temperatures (template interface produces these)
        if getattr(self.config, 'include_zone_conditions', True):
            for zone in self.zone_names:
                keys.append(f"{zone}_temperature")

        # Outdoor conditions (template interface produces these)
        if getattr(self.config, 'include_outdoor_conditions', True):
            keys.extend(["outdoor_temperature", "outdoor_humidity"])

        # Energy consumption (template interface produces these via meters)
        if getattr(self.config, 'include_energy_consumption', True):
            keys.extend(["hvac_electricity", "hvac_gas"])

        # Time features (template interface produces these via functions)
        keys.extend(["hour_of_day", "day_of_week", "day_of_year"])

        return keys

    def pack_observation(self, obs_dict: Dict[str, float]) -> np.ndarray:
        """
        Pack observation dictionary into array.
        Compatible with flattened observations from template-based interface.

        Args:
            obs_dict: Dictionary of observation values from template interface

        Returns:
            Observation array

        Raises:
            ValueError: If any required observation is missing
        """
        import logging
        logger = logging.getLogger("spaces")

        logger.info(f"Packing observation: {len(obs_dict)} observations")

        # Log key observation values
        zone_temps = [obs for key, obs in obs_dict.items() if key.endswith('_temperature') and not key.startswith('outdoor')]
        if zone_temps:
            avg_temp = sum(zone_temps) / len(zone_temps)
            temp_range = max(zone_temps) - min(zone_temps)
            logger.info(f"Zone temperatures - avg: {avg_temp:.1f}°C, range: {temp_range:.1f}°C, count: {len(zone_temps)}")

        if 'outdoor_temperature' in obs_dict:
            logger.info(f"Outdoor temperature: {obs_dict['outdoor_temperature']:.1f}°C")

        if 'hvac_electricity' in obs_dict:
            logger.info(f"HVAC electricity: {obs_dict['hvac_electricity']:.1f} J/interval")

        if 'hvac_gas' in obs_dict:
            logger.info(f"HVAC gas: {obs_dict['hvac_gas']:.1f} J/interval")

        obs_array = []

        for key in self.observation_keys:
            if key not in obs_dict:
                logger.error(f"Missing required observation: {key}")
                logger.error(f"Available observations: {list(obs_dict.keys())}")
                logger.error(f"Expected observations: {self.observation_keys}")
                raise ValueError(f"Missing required observation: {key}")

            value = obs_dict[key]
            if value is None:
                logger.error(f"Observation '{key}' has None value")
                raise ValueError(f"Observation '{key}' has None value")

            encoded_value = encode_observation_value(
                key,
                value,
                floor_area_m2=self.config.building_floor_area,
                timesteps_per_hour=self.config.timesteps_per_hour,
            )
            obs_array.append(encoded_value)

        obs_array = np.array(obs_array, dtype=np.float32)
        logger.info(f"Packed observation array: shape={obs_array.shape}, min={obs_array.min():.3f}, max={obs_array.max():.3f}, mean={obs_array.mean():.3f}")

        return obs_array

    def unpack_observation(self, obs_array: np.ndarray) -> Dict[str, float]:
        """
        Unpack observation array into dictionary (for debugging)

        Args:
            obs_array: Observation array

        Returns:
            Dictionary of observation values
        """
        if len(obs_array) != len(self.observation_keys):
            raise ValueError(f"Expected {len(self.observation_keys)} observations, got {len(obs_array)}")

        obs_dict = {}
        for i, key in enumerate(self.observation_keys):
            obs_dict[key] = float(obs_array[i])

        return obs_dict

    def get_observation_info(self) -> Dict[str, Any]:
        """Get information about observation space"""
        return {
            "total_observations": len(self.observation_keys),
            "zone_count": self.n_zones,
            "observation_keys": self.observation_keys.copy(),
            "space_shape": self.gym_space.shape,
            "space_bounds": (float(self.gym_space.low[0]), float(self.gym_space.high[0])),
            "encoding": OBSERVATION_SCHEMA_ID,
            "includes_zones": getattr(self.config, 'include_zone_conditions', True),
            "includes_outdoor": getattr(self.config, 'include_outdoor_conditions', True),
            "includes_energy": getattr(self.config, 'include_energy_consumption', True)
        }

    def observation_summary(self, obs_dict: Dict[str, float]) -> str:
        """
        Create human-readable summary of observations

        Args:
            obs_dict: Observation dictionary

        Returns:
            Summary string
        """
        summary_parts = []

        # Zone temperatures
        if getattr(self.config, 'include_zone_conditions', True):
            zone_temps = []
            for zone in self.zone_names:
                temp_key = f"{zone}_temperature"
                if temp_key in obs_dict:
                    zone_temps.append(obs_dict[temp_key])

            if zone_temps:
                avg_temp = np.mean(zone_temps)
                temp_range = max(zone_temps) - min(zone_temps)
                summary_parts.append(f"Zones: {avg_temp:.1f}°C avg (range: {temp_range:.1f}°C)")

        # Outdoor conditions
        if getattr(self.config, 'include_outdoor_conditions', True):
            if "outdoor_temperature" in obs_dict and "outdoor_humidity" in obs_dict:
                outdoor_temp = obs_dict["outdoor_temperature"]
                outdoor_humidity = obs_dict["outdoor_humidity"]
                summary_parts.append(f"Outdoor: {outdoor_temp:.1f}°C, {outdoor_humidity:.1%} RH")

        # Energy consumption
        if getattr(self.config, 'include_energy_consumption', True):
            if "hvac_electricity" in obs_dict:
                hvac_elec = obs_dict["hvac_electricity"]
                summary_parts.append(f"HVAC: {hvac_elec:.0f} J/interval")

        return "; ".join(summary_parts) if summary_parts else "No observations"


# Compatibility functions for reference wrapper patterns
def extract_ordered_observations(obs: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """
    Extract observations in the same order as reference wrapper.
    This function bridges between template-based observations and the reference pattern.

    Args:
        obs: Flattened observation dict from template interface

    Returns:
        Tuple of (names, values) in reference wrapper order
    """
    # Get zone names from temperature observations
    zone_names = []
    for key in obs.keys():
        if key.endswith('_temperature') and not key.startswith('outdoor'):
            zone_name = key[:-12]  # Remove '_temperature' suffix
            zone_names.append(zone_name)
    zone_names.sort()  # Ensure consistent order

    # Build names and values in reference wrapper order
    names = []
    values = []

    # Zone temperatures
    for zone in zone_names:
        names.append(f"Zone Temperature {zone}")
        values.append(obs[f"{zone}_temperature"])

    # Outdoor conditions
    names.extend(["Outdoor Air Temperature", "Outdoor Air Relative Humidity"])
    values.extend([
        obs["outdoor_temperature"],
        obs["outdoor_humidity"]
    ])

    # Time features
    names.extend(["Current Time of Day", "Day of Year"])
    values.extend([
        obs["hour_of_day"] * 24,  # Convert back to hours
        obs["day_of_year"] * 365  # Convert back to days
    ])

    # Energy consumption
    names.extend(["HVAC Electricity Consumption", "HVAC Natural Gas Consumption"])
    values.extend([
        obs["hvac_electricity"],
        obs["hvac_gas"]
    ])

    return names, values


def observation_transform(obs: Dict[str, Any], floor_area: float = None) -> np.ndarray:
    """
    Transform observations using reference wrapper pattern.
    Compatible with template-based interface output.

    Args:
        obs: Flattened observation dict from template interface
        floor_area: Building floor area for energy normalization

    Returns:
        Transformed observation array

    Raises:
        ValueError: If floor_area is missing or invalid
    """
    names, values = extract_ordered_observations(obs)
    values = np.array(values)

    # Get indices of HVAC-related values (should be the last two elements)
    hvac_indices = [i for i, name in enumerate(names) if "HVAC" in name]

    # Validate floor area
    if floor_area is None:
        raise ValueError("floor_area is required for energy normalization")
    if floor_area <= 0:
        raise ValueError(f"floor_area must be positive, got {floor_area}")

    # Convert EnergyPlus interval joules to kWh/m2 without timestep scaling.
    if hvac_indices and floor_area > 0:
        values[hvac_indices] = values[hvac_indices] / JOULES_PER_KWH / floor_area

    return values


def create_observation_space(obs_template: Dict[str, Any]) -> Tuple[gym.spaces.Box, List[str]]:
    """
    Create observation space using reference wrapper pattern.
    Compatible with template-based observations.

    Args:
        obs_template: Template observation dict

    Returns:
        Tuple of (observation_space, observation_names)
    """
    names, values = extract_ordered_observations(obs_template)
    num_obs = len(values)

    # Create appropriate bounds for each observation type
    low_bounds = np.zeros(num_obs)
    high_bounds = np.ones(num_obs) * float('inf')

    # Count zones for indexing
    num_zones = len([n for n in names if "Zone Temperature" in n])
    idx = 0

    # Temperature bounds for zones
    low_bounds[idx:idx+num_zones] = 10.0
    high_bounds[idx:idx+num_zones] = 40.0
    idx += num_zones

    # Outdoor temperature bounds
    low_bounds[idx] = -50.0
    high_bounds[idx] = 50.0
    idx += 1

    # Humidity bounds
    low_bounds[idx] = 0.0
    high_bounds[idx] = 100.0
    idx += 1

    # Time bounds
    low_bounds[idx] = 0.0
    high_bounds[idx] = 24.0
    idx += 1

    # Day of year bounds
    low_bounds[idx] = 1.0
    high_bounds[idx] = 366.0
    idx += 1

    # HVAC bounds
    low_bounds[idx:] = 0.0
    high_bounds[idx:] = 40.0  # Normalized Wh/ft²

    return gym.spaces.Box(low_bounds, high_bounds), names


def action_transform(act: np.ndarray, actuators: Dict) -> Dict[str, float]:
    """
    Transform actions using reference wrapper pattern.
    Compatible with template-based interface expectations.

    Args:
        act: Action array [heating_setpoint, offset]
        actuators: Dict of actuator specifications (not used in template approach)

    Returns:
        Dict mapping zone_name -> (heating_setpoint, cooling_setpoint)
    """
    if len(act) != 2:
        raise ValueError(f"Expected 2 actions (heating_setpoint, offset), got {len(act)}")

    # Retain the historical behavior of this standalone helper. Scientific
    # benchmark tasks use ActionSpace with an explicit transform version.
    heating_setpoint = round(float(act[0]), 1)
    offset = round(float(act[1]), 1)
    cooling_setpoint = heating_setpoint + offset

    # For single zone control (reference wrapper pattern)
    # Template interface expects zone-specific format
    return {"Zone1": (heating_setpoint, cooling_setpoint)}


def create_action_space(actuators: Dict) -> gym.spaces.Box:
    """
    Create action space using reference wrapper pattern.

    Args:
        actuators: Dict of actuator specifications

    Returns:
        Action space for heating setpoint and offset
    """
    # Verify we have actuators for heating and cooling
    if len(actuators) < 2:
        raise ValueError(f"Expected at least 2 actuators (heating and cooling), got {len(actuators)}")

    # Temperature bounds: [heating_setpoint, offset]
    return gym.spaces.Box(np.array([15.0, 1.0]), np.array([25.0, 15.0]))
