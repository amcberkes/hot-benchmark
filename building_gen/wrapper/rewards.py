"""Reward functions for the EnergyPlus RL environment."""

from dataclasses import asdict
from typing import Any, Dict, List, Mapping

from building_gen.accounting import CarrierEnergyLedger, energy_intensity_kwh_per_m2
from building_gen.reward_foundation import (
    ENERGY_COMFORT_LEGACY_ALIAS,
    HOT_LEGACY_ENERGY_COMFORT_V1,
    HOT_OCCUPIED_ENERGY_COMFORT_V2,
    OccupiedEnergyComfortV2Components,
    OccupiedEnergyComfortV2Config,
    ZoneComfortSample,
    calculate_hot_occupied_energy_comfort_v2,
)
from .config import EnergyPlusConfig


class RewardFunction:
    """Base class for reward functions"""

    def __init__(self, config: EnergyPlusConfig, zone_names: List[str]):
        self.config = config
        self.zone_names = zone_names

    def calculate_reward(
        self,
        observations: Dict[str, float],
        actions: Dict[str, tuple],
        energy_ledger: CarrierEnergyLedger | None = None,
    ) -> float:
        """Calculate reward based on observations and actions"""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset episode-local reward state, if any."""

    def calculate_reward_with_components(
        self,
        observations: Dict[str, float],
        actions: Dict[str, tuple],
        energy_ledger: CarrierEnergyLedger | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Return the scalar reward and any auditable component values."""

        return self.calculate_reward(observations, actions, energy_ledger), {}


class EnergyComfortReward(RewardFunction):
    """Reward function balancing energy efficiency and comfort"""

    def __init__(self, config: EnergyPlusConfig, zone_names: List[str]):
        super().__init__(config, zone_names)
        self.previous_setpoints: Dict[str, tuple] = {}

    def reset(self) -> None:
        self.previous_setpoints = {}

    def calculate_reward(
        self,
        observations: Dict[str, float],
        actions: Dict[str, tuple],
        energy_ledger: CarrierEnergyLedger | None = None,
    ) -> float:
        """
        Calculate reward as negative weighted sum of:
        - Energy consumption penalty
        - Comfort violation penalty
        - Setpoint change penalty (for stability)
        """

        # DEBUG: Log observations and actions
        import logging

        logger = logging.getLogger(__name__)

        # Log key observation values for debugging
        zone_temps = [
            obs for key, obs in observations.items() if key.endswith("_temperature") and not key.startswith("outdoor")
        ]
        if zone_temps:
            avg_temp = sum(zone_temps) / len(zone_temps)
            temp_range = max(zone_temps) - min(zone_temps)
            logger.info(
                f"Zone temperatures - avg: {avg_temp:.1f}°C, range: {temp_range:.1f}°C, count: {len(zone_temps)}"
            )

        if "outdoor_temperature" in observations:
            logger.info(f"Outdoor temperature: {observations['outdoor_temperature']:.1f}°C")

        if energy_ledger is not None:
            logger.info(f"Reported HVAC energy: {energy_ledger.total_reported_kwh:.6f} kWh")

        if "hvac_gas" in observations:
            logger.info(f"HVAC gas: {observations['hvac_gas']:.1f} J/interval")

        logger.debug(f"Reward calculation - Observations: {observations}")
        logger.debug(f"Reward calculation - Actions: {actions}")

        # Energy penalty
        energy_penalty = self._calculate_energy_penalty(energy_ledger)
        logger.info(f"Energy penalty: {energy_penalty:.3f}")

        # Comfort penalty
        comfort_penalty = self._calculate_comfort_penalty(observations)
        logger.info(f"Comfort penalty: {comfort_penalty:.3f}")

        # Setpoint change penalty (if tracking previous setpoints)
        stability_penalty = self._calculate_stability_penalty(actions)
        logger.info(f"Stability penalty: {stability_penalty:.3f}")

        self.previous_setpoints = actions.copy()

        # Combined reward (negative because we want to minimize penalties)
        total_reward = -(
            self.config.energy_penalty_weight * energy_penalty
            + self.config.comfort_penalty_weight * comfort_penalty
            + self.config.stability_penalty_weight * stability_penalty
        )

        logger.info(f"Total reward: {total_reward:.3f}")
        logger.info(
            f"Weights - energy: {self.config.energy_penalty_weight}, comfort: {self.config.comfort_penalty_weight}"
        )

        return total_reward

    def _calculate_energy_penalty(self, energy_ledger: CarrierEnergyLedger | None) -> float:
        """Return interval site-energy intensity in kWh/m2."""

        if energy_ledger is None:
            raise ValueError("EnergyComfortReward requires an energy ledger")
        return energy_intensity_kwh_per_m2(energy_ledger, self.config.building_floor_area)

    def _calculate_comfort_penalty(self, observations: Dict[str, float]) -> float:
        """Calculate penalty based on comfort violations"""
        comfort_penalty = 0.0
        comfort_min, comfort_max = self.config.comfort_temp_range

        for zone_name in self.zone_names:
            zone_temp_key = f"{zone_name}_temperature"
            if zone_temp_key not in observations:
                raise ValueError(f"Missing required observation for comfort penalty: {zone_temp_key}")

            zone_temp = observations[zone_temp_key]

            # Calculate comfort violation
            if zone_temp < comfort_min:
                violation = comfort_min - zone_temp
            elif zone_temp > comfort_max:
                violation = zone_temp - comfort_max
            else:
                violation = 0.0

            # Quadratic penalty for violations
            comfort_penalty += violation**2

        return comfort_penalty

    def _calculate_stability_penalty(self, actions: Dict[str, tuple]) -> float:
        """Calculate penalty for large setpoint changes"""
        if not self.previous_setpoints:
            return 0.0
        penalty = 0.0
        for zone_name, (heating_setpoint, cooling_setpoint) in actions.items():
            if zone_name not in self.previous_setpoints:
                continue
            previous_heating, previous_cooling = self.previous_setpoints[zone_name]
            penalty += (heating_setpoint - previous_heating) ** 2
            penalty += (cooling_setpoint - previous_cooling) ** 2
        return penalty


class EnergyOnlyReward(RewardFunction):
    """Reward function focused only on energy efficiency"""

    def calculate_reward(
        self,
        observations: Dict[str, float],
        actions: Dict[str, tuple],
        energy_ledger: CarrierEnergyLedger | None = None,
    ) -> float:
        """Return negative interval site-energy intensity in kWh/m2."""

        if energy_ledger is None:
            raise ValueError("EnergyOnlyReward requires an energy ledger")
        return -energy_intensity_kwh_per_m2(energy_ledger, self.config.building_floor_area)


class OccupiedEnergyComfortRewardV2(RewardFunction):
    """Runtime adapter for the simulator-independent occupied reward v2."""

    def __init__(
        self,
        config: EnergyPlusConfig,
        zone_names: List[str],
        *,
        zone_floor_areas_m2: Mapping[str, float],
        energy_weight: float,
        comfort_weight: float,
        energy_scale_kwh_per_m2: float,
        comfort_scale_person_celsius_squared_hours_per_m2: float,
    ) -> None:
        super().__init__(config, zone_names)
        if not isinstance(zone_floor_areas_m2, Mapping):
            raise TypeError("zone_floor_areas_m2 must be a mapping")
        missing = [zone for zone in zone_names if zone not in zone_floor_areas_m2]
        extra = [zone for zone in zone_floor_areas_m2 if zone not in zone_names]
        if missing or extra:
            raise ValueError(
                "zone_floor_areas_m2 must match the controlled zones exactly: " f"missing={missing}, extra={extra}"
            )
        self.zone_floor_areas_m2 = {zone: zone_floor_areas_m2[zone] for zone in zone_names}
        self.reward_config = OccupiedEnergyComfortV2Config(
            energy_weight=energy_weight,
            comfort_weight=comfort_weight,
            energy_scale_kwh_per_m2=energy_scale_kwh_per_m2,
            comfort_scale_person_celsius_squared_hours_per_m2=(comfort_scale_person_celsius_squared_hours_per_m2),
            comfort_min_c=config.comfort_temp_range[0],
            comfort_max_c=config.comfort_temp_range[1],
        )

    def _components(
        self,
        observations: Dict[str, float],
        energy_ledger: CarrierEnergyLedger | None,
    ) -> OccupiedEnergyComfortV2Components:
        if energy_ledger is None:
            raise ValueError("OccupiedEnergyComfortRewardV2 requires an energy ledger")
        samples = []
        for zone in self.zone_names:
            temperature_key = f"{zone}_temperature"
            occupant_key = f"{zone}_occupant_count"
            missing = [key for key in (temperature_key, occupant_key) if key not in observations]
            if missing:
                raise ValueError("Missing required observation(s) for occupied reward v2: " + ", ".join(missing))
            samples.append(
                ZoneComfortSample(
                    zone_id=zone,
                    floor_area_m2=self.zone_floor_areas_m2[zone],
                    temperature_c=observations[temperature_key],
                    occupant_count=observations[occupant_key],
                )
            )
        return calculate_hot_occupied_energy_comfort_v2(
            interval_controlled_hvac_site_energy_kwh=(energy_ledger.total_reported_kwh),
            timestep_hours=1.0 / self.config.timesteps_per_hour,
            zones=samples,
            config=self.reward_config,
        )

    def calculate_reward(
        self,
        observations: Dict[str, float],
        actions: Dict[str, tuple],
        energy_ledger: CarrierEnergyLedger | None = None,
    ) -> float:
        del actions
        return self._components(observations, energy_ledger).reward

    def calculate_reward_with_components(
        self,
        observations: Dict[str, float],
        actions: Dict[str, tuple],
        energy_ledger: CarrierEnergyLedger | None = None,
    ) -> tuple[float, dict[str, Any]]:
        del actions
        components = self._components(observations, energy_ledger)
        return components.reward, asdict(components)


class ComfortOnlyReward(RewardFunction):
    """Reward function focused only on comfort"""

    def calculate_reward(
        self,
        observations: Dict[str, float],
        actions: Dict[str, tuple],
        energy_ledger: CarrierEnergyLedger | None = None,
    ) -> float:
        """Calculate reward based only on comfort"""
        comfort_penalty = 0.0
        comfort_min, comfort_max = self.config.comfort_temp_range

        for zone_name in self.zone_names:
            zone_temp_key = f"{zone_name}_temperature"
            if zone_temp_key not in observations:
                raise ValueError(f"Missing required observation for comfort penalty: {zone_temp_key}")

            zone_temp = observations[zone_temp_key]

            # Calculate distance from comfort zone
            if zone_temp < comfort_min:
                violation = comfort_min - zone_temp
            elif zone_temp > comfort_max:
                violation = zone_temp - comfort_max
            else:
                violation = 0.0

            comfort_penalty += violation**2

        return -comfort_penalty


class CustomReward(RewardFunction):
    """Customizable reward function with configurable weights"""

    def __init__(
        self,
        config: EnergyPlusConfig,
        zone_names: List[str],
        energy_weight: float = 0.1,
        comfort_weight: float = 1.0,
        stability_weight: float = 0.01,
        setpoint_penalty_weight: float = 0.001,
    ):
        super().__init__(config, zone_names)
        self.energy_weight = energy_weight
        self.comfort_weight = comfort_weight
        self.stability_weight = stability_weight
        self.setpoint_penalty_weight = setpoint_penalty_weight
        self.previous_setpoints = {}

    def reset(self) -> None:
        self.previous_setpoints = {}

    def calculate_reward(
        self,
        observations: Dict[str, float],
        actions: Dict[str, tuple],
        energy_ledger: CarrierEnergyLedger | None = None,
    ) -> float:
        """Calculate customizable weighted reward"""

        # DEBUG: Log observations and actions
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"Reward calculation - Observations: {observations}")
        logger.debug(f"Reward calculation - Actions: {actions}")

        # Energy component
        energy_penalty = self._calculate_energy_consumption(energy_ledger)
        logger.debug(f"Energy penalty: {energy_penalty}")

        # Comfort component
        comfort_penalty = self._calculate_comfort_violations(observations)
        logger.debug(f"Comfort penalty: {comfort_penalty}")

        # Setpoint efficiency (penalize extreme setpoints)
        setpoint_penalty = self._calculate_setpoint_penalty(actions)
        logger.debug(f"Setpoint penalty: {setpoint_penalty}")

        # Stability (penalize rapid setpoint changes)
        stability_penalty = self._calculate_setpoint_changes(actions)
        logger.debug(f"Stability penalty: {stability_penalty}")

        # Update previous setpoints for next step
        self.previous_setpoints = actions.copy()

        # Combined reward
        reward = -(
            self.energy_weight * energy_penalty
            + self.comfort_weight * comfort_penalty
            + self.setpoint_penalty_weight * setpoint_penalty
            + self.stability_weight * stability_penalty
        )

        logger.debug(f"Total reward: {reward}")
        logger.debug(
            f"Weights - energy: {self.energy_weight}, comfort: {self.comfort_weight}, setpoint: {self.setpoint_penalty_weight}, stability: {self.stability_weight}"
        )

        return reward

    def _calculate_energy_consumption(self, energy_ledger: CarrierEnergyLedger | None) -> float:
        """Return interval site-energy intensity in kWh/m2."""

        if energy_ledger is None:
            raise ValueError("CustomReward requires an energy ledger")
        return energy_intensity_kwh_per_m2(energy_ledger, self.config.building_floor_area)

    def _calculate_comfort_violations(self, observations: Dict[str, float]) -> float:
        """Comfort violation penalty"""
        comfort_penalty = 0.0
        comfort_min, comfort_max = self.config.comfort_temp_range

        for zone_name in self.zone_names:
            zone_temp_key = f"{zone_name}_temperature"
            if zone_temp_key not in observations:
                raise ValueError(f"Missing required observation for comfort penalty: {zone_temp_key}")

            zone_temp = observations[zone_temp_key]

            if zone_temp < comfort_min:
                violation = comfort_min - zone_temp
            elif zone_temp > comfort_max:
                violation = zone_temp - comfort_max
            else:
                violation = 0.0

            comfort_penalty += violation**2

        return comfort_penalty

    def _calculate_setpoint_penalty(self, actions: Dict[str, tuple]) -> float:
        """Penalty for extreme setpoints"""
        penalty = 0.0

        for zone_name, (heating_sp, cooling_sp) in actions.items():
            # Penalize setpoints far from reasonable values
            heating_penalty = max(0, 18.0 - heating_sp) + max(0, heating_sp - 23.0)
            cooling_penalty = max(0, 23.0 - cooling_sp) + max(0, cooling_sp - 28.0)

            # Penalize small deadband
            deadband = cooling_sp - heating_sp
            deadband_penalty = max(0, 2.0 - deadband) ** 2

            penalty += heating_penalty + cooling_penalty + deadband_penalty

        return penalty

    def _calculate_setpoint_changes(self, actions: Dict[str, tuple]) -> float:
        """Penalty for rapid setpoint changes"""
        if not self.previous_setpoints:
            return 0.0

        change_penalty = 0.0

        for zone_name, (heating_sp, cooling_sp) in actions.items():
            if zone_name in self.previous_setpoints:
                prev_heating, prev_cooling = self.previous_setpoints[zone_name]

                heating_change = abs(heating_sp - prev_heating)
                cooling_change = abs(cooling_sp - prev_cooling)

                change_penalty += heating_change**2 + cooling_change**2

        return change_penalty


def get_reward_function(reward_type: str, config: EnergyPlusConfig, zone_names: List[str], **kwargs) -> RewardFunction:
    """Factory function to create reward functions"""

    from building_gen.scheduled_reward import (
        HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3, ScheduledOccupiedRewardConfig)
    if reward_type == HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3:
        reward_config = ScheduledOccupiedRewardConfig.from_mapping({key: kwargs[key] for key in (
            "energy_weight", "comfort_weight", "energy_scale_kwh_per_m2",
            "comfort_scale_person_celsius_squared_hours_per_m2", "comfort_schedule")})
        return ScheduledOccupiedReward(config, zone_names,
            zone_floor_areas_m2=kwargs["zone_floor_areas_m2"], reward_config=reward_config)
    if reward_type == ENERGY_COMFORT_LEGACY_ALIAS:
        return EnergyComfortReward(config, zone_names)
    if reward_type == HOT_LEGACY_ENERGY_COMFORT_V1:
        return EnergyComfortReward(config, zone_names)
    elif reward_type == HOT_OCCUPIED_ENERGY_COMFORT_V2:
        return OccupiedEnergyComfortRewardV2(
            config,
            zone_names,
            zone_floor_areas_m2=kwargs["zone_floor_areas_m2"],
            energy_weight=kwargs["energy_weight"],
            comfort_weight=kwargs["comfort_weight"],
            energy_scale_kwh_per_m2=kwargs["energy_scale_kwh_per_m2"],
            comfort_scale_person_celsius_squared_hours_per_m2=kwargs[
                "comfort_scale_person_celsius_squared_hours_per_m2"
            ],
        )
    elif reward_type == "energy_only":
        return EnergyOnlyReward(config, zone_names)
    elif reward_type == "comfort_only":
        return ComfortOnlyReward(config, zone_names)
    elif reward_type == "custom":
        return CustomReward(config, zone_names, **kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


class ScheduledOccupiedReward(RewardFunction):
    """Scheduled objective evaluated only from completed-interval evidence."""

    def __init__(self, config, zone_names, *, zone_floor_areas_m2, reward_config):
        from building_gen.scheduled_reward import ScheduledOccupiedRewardConfig
        super().__init__(config, zone_names)
        if not isinstance(reward_config, ScheduledOccupiedRewardConfig):
            raise TypeError('reward_config must be ScheduledOccupiedRewardConfig')
        if set(zone_floor_areas_m2) != set(zone_names) or len(set(zone_names)) != len(zone_names):
            raise ValueError('zone areas must match distinct controlled zones exactly')
        reward_config.comfort_schedule.validate_zones(tuple(zone_names))
        self.zone_floor_areas_m2 = dict(zone_floor_areas_m2)
        for zone, area in self.zone_floor_areas_m2.items():
            ZoneComfortSample(zone, area, 20.0, 0.0)
        self.reward_config = reward_config

    def calculate_reward(self, observations, actions, energy_ledger=None):
        raise ValueError('scheduled reward requires the completed interval clock')

    def calculate_interval_reward(self, observations, actions, energy_ledger, clock):
        from building_gen.scheduled_reward import calculate_scheduled_occupied_reward
        from building_gen.wrapper.simulation_clock import SimulationClock
        del actions
        interval_clock = SimulationClock(**clock)
        if interval_clock.timesteps_per_hour != self.config.timesteps_per_hour:
            raise ValueError('reward clock timestep frequency differs from simulator configuration')
        if energy_ledger is None:
            raise ValueError('scheduled reward requires an energy ledger')
        samples = [ZoneComfortSample(zone, self.zone_floor_areas_m2[zone],
                   observations[f'{zone}_temperature'], observations[f'{zone}_occupant_count'])
                   for zone in self.zone_names]
        components = calculate_scheduled_occupied_reward(
            interval_controlled_hvac_site_energy_kwh=energy_ledger.total_reported_kwh,
            timestep_hours=1.0/self.config.timesteps_per_hour, zones=samples,
            config=self.reward_config, weekday=interval_clock.day_of_week, month=interval_clock.month,
            start_minute=60*(interval_clock.hour+(interval_clock.timestep_index-1)/interval_clock.timesteps_per_hour))
        return components['reward'], components
