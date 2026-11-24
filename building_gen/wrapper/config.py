"""Configuration settings for the EnergyPlus Gym environment."""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from building_gen.wrapper.simulation_clock import (
    CalendarYearSemantics,
    validate_calendar_year_semantics,
)


@dataclass
class EnergyPlusConfig:
    """Configuration for EnergyPlus environment"""

    # File paths - container compatible
    energyplus_dir: str = "/usr/local/EnergyPlus-24-1-0"
    building_file: Optional[str] = None
    weather_file: Optional[str] = None
    output_dir: str = "ep_outputs"

    # Simulation settings
    timesteps_per_hour: int = 4
    episode_length_hours: int = 24
    warmup_days: int = 15
    run_period_days: int = 1
    max_episodes: int = 1000

    # FIXED: Realistic temperature bounds for action space
    heating_setpoint_range: Tuple[float, float] = (18.0, 23.0)  # Realistic heating range
    offset_range: Tuple[float, float] = (2.0, 4.0)  # Realistic deadband range
    cooling_setpoint_range: Tuple[float, float] = (20.0, 27.0)  # Envelope: (18+2, 23+4)
    action_channel_names: Tuple[str, ...] = (
        "heating_setpoint_c",
        "cooling_offset_c",
    )
    action_transform_id: str = "linear_quantized_0_1c_legacy"

    # Legacy settings (kept for backward compatibility)
    energy_penalty_weight: float = 1.0
    comfort_penalty_weight: float = 1.0
    stability_penalty_weight: float = 0.0

    # Comfort settings
    comfort_temp_range: Tuple[float, float] = (20.0, 24.0)  # °C

    # Observation settings
    include_outdoor_conditions: bool = True
    include_zone_conditions: bool = True
    include_energy_consumption: bool = True

    # Zone settings (will be populated from building file)
    zone_names: List[str] = None

    # Add building_floor_area for energy normalization
    building_floor_area: float = 1000.0
    # Appended to preserve the positional layout used by archived callers.
    calendar_year_semantics: Optional[CalendarYearSemantics] = None

    def __post_init__(self):
        """Initialize derived settings and validate temperature bounds"""
        if self.zone_names is None:
            self.zone_names = []

        if self.calendar_year_semantics is not None:
            self.calendar_year_semantics = validate_calendar_year_semantics(self.calendar_year_semantics)

        self.action_channel_names = tuple(self.action_channel_names)
        expected_channels = ("heating_setpoint_c", "cooling_offset_c")
        manifest_transform = self.action_transform_id == "hot_manifest_direct_normalized_v1"
        if manifest_transform:
            if (
                not self.action_channel_names
                or any(type(item) is not str or not item for item in self.action_channel_names)
                or len({item.casefold() for item in self.action_channel_names}) != len(self.action_channel_names)
            ):
                raise ValueError("manifest action channel names must be nonempty exact unique strings")
        elif self.action_channel_names != expected_channels:
            raise ValueError(
                "the thermostat action transform requires channels ordered as "
                f"{expected_channels!r}; received {self.action_channel_names!r}"
            )
        supported_transforms = {
            "hot_manifest_direct_normalized_v1",
            "linear_quantized_0_1c_legacy",
            "linear_unquantized_v1",
            "bounded_thermostat_15_30_v1",
            "bounded_thermostat_12_32_v1",
        }
        if self.action_transform_id not in supported_transforms:
            raise ValueError(
                "unsupported action_transform_id "
                f"{self.action_transform_id!r}; expected one of "
                f"{sorted(supported_transforms)!r}"
            )

        for name, value in (
            ("energy_penalty_weight", self.energy_penalty_weight),
            ("comfort_penalty_weight", self.comfort_penalty_weight),
            ("stability_penalty_weight", self.stability_penalty_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.comfort_temp_range[0] >= self.comfort_temp_range[1]:
            raise ValueError("comfort_temp_range is invalid")

        # Validate temperature bounds consistency
        self._validate_temperature_bounds()

        # Detect container environment and adjust paths
        self._detect_container_environment()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Find and validate EnergyPlus installation
        self._find_energyplus_installation()

    def _validate_temperature_bounds(self):
        """Validate that temperature bounds are consistent and realistic"""
        if self.action_transform_id == "linear_quantized_0_1c_legacy":
            bounds = (*self.heating_setpoint_range, *self.offset_range)
            if any(abs(value * 10.0 - round(value * 10.0)) > 1e-9 for value in bounds):
                raise ValueError("legacy tenth-degree action bounds must be aligned to 0.1 C")
        # Ensure heating range is valid
        if self.heating_setpoint_range[0] >= self.heating_setpoint_range[1]:
            raise ValueError("Heating setpoint range is invalid")

        # Ensure offset range is valid
        if self.offset_range[0] >= self.offset_range[1]:
            raise ValueError("Offset range is invalid")

        # Ensure cooling range is valid
        if self.cooling_setpoint_range[0] >= self.cooling_setpoint_range[1]:
            raise ValueError("Cooling setpoint range is invalid")

        # Calculate expected cooling range from heating + offset
        min_possible_cooling = self.heating_setpoint_range[0] + self.offset_range[0]
        max_possible_cooling = self.heating_setpoint_range[1] + self.offset_range[1]

        if self.action_transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1"):
            from building_gen.bounded_thermostat import validate_ranges
            validate_ranges(self.heating_setpoint_range,self.offset_range,self.action_transform_id)
            max_possible_cooling=32.0 if self.action_transform_id == "bounded_thermostat_12_32_v1" else 30.0
        # Check if cooling range is consistent
        if self.cooling_setpoint_range[0] < min_possible_cooling:
            print(
                f"Warning: Cooling setpoint minimum ({self.cooling_setpoint_range[0]}) is less than "
                f"heating minimum + offset minimum ({min_possible_cooling})"
            )

        if self.cooling_setpoint_range[1] < max_possible_cooling:
            print(
                f"Warning: Cooling setpoint maximum ({self.cooling_setpoint_range[1]}) is less than "
                f"heating maximum + offset maximum ({max_possible_cooling})"
            )

        # Ensure minimum deadband
        if self.offset_range[0] < 1.0:
            raise ValueError(f"Minimum offset ({self.offset_range[0]}) must be at least 1.0°C for proper deadband")

        # Ensure ranges are realistic
        if self.heating_setpoint_range[0] < 15.0 or self.heating_setpoint_range[1] > 25.0:
            print(
                f"Warning: Heating setpoint range {self.heating_setpoint_range} may be outside typical comfort range [15, 25]°C"
            )

        if self.cooling_setpoint_range[0] < 20.0 or self.cooling_setpoint_range[1] > 30.0:
            print(
                f"Warning: Cooling setpoint range {self.cooling_setpoint_range} may be outside typical comfort range [20, 30]°C"
            )

    def _detect_container_environment(self):
        """Detect if running in container and adjust settings"""
        if os.path.exists("/mnt/repo"):
            # Running in container - adjust paths
            if not os.path.isabs(self.output_dir):
                self.output_dir = os.path.join("/mnt/repo", self.output_dir)

            # Check for EnergyPlus environment variable
            energyplus_env = os.environ.get("ENERGYPLUS_PATH")
            if energyplus_env and os.path.exists(energyplus_env):
                self.energyplus_dir = energyplus_env

    def _find_energyplus_installation(self):
        """Find EnergyPlus installation with multiple fallback options"""
        energyplus_env = os.environ.get("ENERGYPLUS_PATH")
        if energyplus_env:
            if not os.path.isdir(energyplus_env):
                raise ValueError(f"ENERGYPLUS_PATH is not an existing directory: {energyplus_env}")
            environment_executable = os.path.join(energyplus_env, "energyplus")
            if not os.path.isfile(environment_executable) or not os.access(environment_executable, os.X_OK):
                raise ValueError(
                    "ENERGYPLUS_PATH does not contain an executable energyplus binary: " f"{environment_executable}"
                )
            self.energyplus_dir = energyplus_env
            self.energyplus_exe = environment_executable
            print(f"Found EnergyPlus at: {energyplus_env}")
            return

        possible_paths = [
            self.energyplus_dir,  # Specified path
            "/usr/local/EnergyPlus-24-1-0",  # Standard container path
            "/usr/local/EnergyPlus-24-2-0",  # Newer version
            "/usr/local/EnergyPlus-23-2-0",  # Older version
            "/opt/EnergyPlus-24-1-0",  # Alternative location
        ]

        for path in possible_paths:
            if os.path.exists(path):
                exe_path = os.path.join(path, "energyplus")
                if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
                    self.energyplus_dir = path
                    self.energyplus_exe = exe_path
                    print(f"Found EnergyPlus at: {path}")
                    return

        # If not found, raise error with helpful message
        raise ValueError(
            f"EnergyPlus installation not found. Tried paths: {possible_paths}\n"
            f"Please ensure EnergyPlus is installed or set ENERGYPLUS_PATH environment variable"
        )

    def validate_building_file(self, building_file: str) -> bool:
        """Validate that building file exists and is correct format"""
        if not os.path.exists(building_file):
            return False
        return building_file.endswith((".epJSON", ".idf"))

    def get_episode_timesteps(self) -> int:
        """Calculate total timesteps per episode"""
        return self.episode_length_hours * self.timesteps_per_hour


# 2. Quick diagnostic script to test your setup
def diagnose_energyplus_setup():
    """Diagnostic function to check EnergyPlus setup"""
    print("=== EnergyPlus Setup Diagnosis ===")

    # Check container environment
    if os.path.exists("/mnt/repo"):
        print("✓ Running in container environment")
    else:
        print("! Not in container - running on host")

    # Check EnergyPlus installations
    possible_paths = [
        "/usr/local/EnergyPlus-24-1-0",
        "/usr/local/EnergyPlus-24-2-0",
        "/usr/local/EnergyPlus-23-2-0",
        "/opt/EnergyPlus-24-1-0",
    ]

    found_installations = []
    for path in possible_paths:
        exe_path = os.path.join(path, "energyplus")
        if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
            found_installations.append(path)
            print(f"✓ Found EnergyPlus at: {path}")

    if not found_installations:
        print("✗ No EnergyPlus installations found!")
        return False

    # Check building files
    building_dirs = ["/mnt/repo/data/base", "/mnt/repo/data/processed_buildings", "/mnt/repo/data"]

    building_files = []
    for dir_path in building_dirs:
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".epJSON"):
                        building_files.append(os.path.join(root, file))
                        if len(building_files) >= 3:  # Show first 3
                            break

    if building_files:
        print(f"✓ Found {len(building_files)} building files:")
        for bf in building_files[:3]:
            print(f"  - {bf}")
    else:
        print("✗ No building files found!")
        return False

    # Check weather files
    weather_dirs = ["/mnt/repo/data/weather", "/mnt/repo/data"]

    weather_files = []
    for dir_path in weather_dirs:
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".epw"):
                        weather_files.append(os.path.join(root, file))
                        if len(weather_files) >= 3:
                            break

    if weather_files:
        print(f"✓ Found {len(weather_files)} weather files:")
        for wf in weather_files[:3]:
            print(f"  - {wf}")
    else:
        print("! No weather files found - will create dummy file")

    print("=== Diagnosis Complete ===")
    return True


# 3. Simple test function
def test_energyplus_interface():
    """Test the fixed EnergyPlus interface"""
    from building_gen.wrapper.config import EnergyPlusConfig
    from building_gen.wrapper.eplus_interface import EnergyPlusInterface

    try:
        # Create config
        config = EnergyPlusConfig(episode_length_hours=1, timesteps_per_hour=4)  # Short test

        # Create interface
        interface = EnergyPlusInterface(config)

        # Find a building file
        building_file = None
        building_dirs = ["/mnt/repo/data/base", "/mnt/repo/data"]

        for dir_path in building_dirs:
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith(".epJSON"):
                            building_file = os.path.join(root, file)
                            break
                    if building_file:
                        break
            if building_file:
                break

        if not building_file:
            print("No building file found for testing")
            return False

        print(f"Testing with building file: {building_file}")

        # Extract zones
        zones = interface._extract_zone_names(building_file)
        print(f"Found zones: {zones[:5]}...")  # Show first 5

        print("✓ EnergyPlus interface basic test passed")
        return True

    except Exception as e:
        print(f"✗ EnergyPlus interface test failed: {e}")
        return False


# Usage:
# Run this in your container to diagnose issues:
# diagnose_energyplus_setup()
# test_energyplus_interface()
