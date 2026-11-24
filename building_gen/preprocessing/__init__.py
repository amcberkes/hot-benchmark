"""Preprocessing APIs with lightweight imports for pure epJSON transforms."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .model_repairs import (
    OFFICESMALL_AUDITED_DEFROST_CURVE_NAME,
    OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_ID,
    OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_VERSION,
    OFFICESMALL_DEFROST_HEATING_COIL_NAMES,
    OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME,
    BiquadraticCurveSnapshot,
    HeatingCoilDefrostCurveReference,
    MixedWaterHeaterUseSideRepairError,
    MixedWaterHeaterUseSideRepairReport,
    OfficeSmallDefrostEIRCurveRepairError,
    OfficeSmallDefrostEIRCurveRepairReport,
    PlantConnectedMixedWaterHeaterRepairError,
    PlantConnectedMixedWaterHeaterRepairReport,
    repair_mixed_water_heater_use_side_nodes,
    repair_officesmall_defrost_eir_curve,
    repair_plant_connected_mixed_water_heater,
)
from .output_requests import (
    HotOutputRequestPlan,
    ModelOutputCapabilities,
    OutputRequestDiff,
    OutputRequestReport,
    OutputRequestValidationError,
    apply_hot_output_requests,
    configure_hot_output_requests_file,
    copy_with_hot_output_requests,
    detect_output_capabilities,
    plan_hot_output_requests,
)

_PROCESSING_EXPORTS = (
    "transition_idf",
    "convert_to_epjson",
    "add_hvac_meters_to_epjson",
    "add_outdoor_air_meters_to_epjson",
    "modify_timestep",
    "add_controllable_setpoint_control_to_epjson",
    "verify_controllable_setpoints",
    "add_outdoor_air_nodes_if_missing",
    "process_idf",
    "add_comprehensive_meters_and_variables",
)


def __getattr__(name: str) -> Any:
    """Load legacy preprocessing dependencies only when explicitly requested."""

    if name in _PROCESSING_EXPORTS:
        return getattr(import_module(".processing", __name__), name)
    if name == "validate_building":
        return getattr(import_module(".validation", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *_PROCESSING_EXPORTS,
    "validate_building",
    "BiquadraticCurveSnapshot",
    "HeatingCoilDefrostCurveReference",
    "MixedWaterHeaterUseSideRepairError",
    "MixedWaterHeaterUseSideRepairReport",
    "OFFICESMALL_AUDITED_DEFROST_CURVE_NAME",
    "OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_ID",
    "OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_VERSION",
    "OFFICESMALL_DEFROST_HEATING_COIL_NAMES",
    "OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME",
    "OfficeSmallDefrostEIRCurveRepairError",
    "OfficeSmallDefrostEIRCurveRepairReport",
    "PlantConnectedMixedWaterHeaterRepairError",
    "PlantConnectedMixedWaterHeaterRepairReport",
    "repair_mixed_water_heater_use_side_nodes",
    "repair_officesmall_defrost_eir_curve",
    "repair_plant_connected_mixed_water_heater",
    "HotOutputRequestPlan",
    "ModelOutputCapabilities",
    "OutputRequestDiff",
    "OutputRequestReport",
    "OutputRequestValidationError",
    "apply_hot_output_requests",
    "configure_hot_output_requests_file",
    "copy_with_hot_output_requests",
    "detect_output_capabilities",
    "plan_hot_output_requests",
]
