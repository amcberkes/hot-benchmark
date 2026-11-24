"""Parse and validate EnergyPlus run diagnostics without importing EnergyPlus.

EnergyPlus writes one copy of a recurring warning and reports subsequent
occurrences in a recurring-error summary.  Counting only lines containing
``** Warning **`` therefore substantially undercounts noisy simulations.  The
parser in this module retains both the marked warning signatures and the
reported recurring occurrence counts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

DIAGNOSTICS_SCHEMA_VERSION = "hot-energyplus-diagnostics/2.0.0"
LEGACY_DIAGNOSTICS_SCHEMA_VERSION = "hot-energyplus-diagnostics/1.0.0"
DIAGNOSTICS_ALLOWLIST_SCHEMA_VERSION = "hot-energyplus-diagnostics-policy/2.0.0"
DIAGNOSTICS_FILENAME = "energyplus_diagnostics.json"
DIAGNOSTICS_MODES = frozenset({"engineering", "scientific"})

_VERSION_RE = re.compile(r"Program Version,\s*EnergyPlus,\s*Version\s+([^,\r\n]+)", re.IGNORECASE)
_MARKED_MESSAGE_RE = re.compile(
    r"^\s*\*\*\s*(Warning|Severe|Fatal)\s*\*\*\s*(.*?)\s*$",
    re.IGNORECASE,
)
_RECURRING_WARNING_RE = re.compile(
    r"^\s*\*{5,}.*?\*\*\s*Warning\s*\*\*\s*(.*?)\s*$",
    re.IGNORECASE,
)
_RECURRING_COUNT_RE = re.compile(r"This error occurred\s+(\d+)\s+total times", re.IGNORECASE)
_ANY_MARKED_DIAGNOSTIC_RE = re.compile(
    r"^\s*(?:\*{5,}\s*)?\*\*\s*(?:Warning|Severe|Fatal|~~~)\s*\*\*",
    re.IGNORECASE,
)
_UNUSED_SCHEDULES_RE = re.compile(r"^\s*\*{5,}\s+There\s+are\s+([1-9]\d*)\s+unused\s+schedules\s+in\s+input\.\s*$")
_UNUSED_WEEK_SCHEDULES_RE = re.compile(
    r"^\s*\*{5,}\s+There\s+are\s+([1-9]\d*)\s+unused\s+week\s+schedules\s+in\s+input\.\s*$"
)
_UNUSED_DAY_SCHEDULES_RE = re.compile(
    r"^\s*\*{5,}\s+There\s+are\s+([1-9]\d*)\s+unused\s+day\s+schedules\s+in\s+input\.\s*$"
)
_UNUSED_SCHEDULES_ADVISORY_RE = re.compile(
    r"^\s*\*{5,}\s+Use\s+Output:Diagnostics,DisplayUnusedSchedules;\s+to\s+see\s+them\.\s*$"
)
_COMPLETION_RE = re.compile(
    r"EnergyPlus\s+(Completed Successfully|Terminated--Fatal Error Detected)\s*(?:--|\.)\s*"
    r"(\d+)\s+Warnings?\s*;\s*(\d+)\s+Severe Errors?",
    re.IGNORECASE,
)


def _signature(value: str) -> str:
    return " ".join(value.split())


def _sorted_counts(values: dict[str, int]) -> tuple["DiagnosticOccurrence", ...]:
    return tuple(DiagnosticOccurrence(signature=signature, count=count) for signature, count in sorted(values.items()))


@dataclass(frozen=True, slots=True)
class DiagnosticOccurrence:
    """One normalized EnergyPlus message and its occurrence count."""

    signature: str
    count: int

    def __post_init__(self) -> None:
        if type(self.signature) is not str or not self.signature or self.signature != _signature(self.signature):
            raise ValueError("diagnostic signature must be an exact, non-empty, whitespace-normalized string")
        if type(self.count) is not int or self.count < 0:
            raise ValueError("diagnostic count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class EnergyPlusDiagnostics:
    """Structured diagnostics parsed from one ``eplusout.err``/``.end`` pair."""

    schema_version: str
    energyplus_version: str | None
    completed_successfully: bool
    completion_records_agree: bool
    err_completion_present: bool
    end_completion_present: bool
    total_warning_count: int | None
    severe_error_count: int
    fatal_error_count: int
    warning_occurrences: tuple[DiagnosticOccurrence, ...]
    # Counts after the visible first occurrence, retained for compatibility
    # with exact signature policies authored against schema 1.
    recurring_warning_occurrences: tuple[DiagnosticOccurrence, ...]
    severe_occurrences: tuple[DiagnosticOccurrence, ...]
    fatal_occurrences: tuple[DiagnosticOccurrence, ...]
    accounted_warning_count: int
    unattributed_warning_count: int | None
    # Exact non-warning diagnostics emitted via ShowMessage.  They must not be
    # added to EnergyPlus's warning counter, but scientific evaluation still
    # requires an explicit signature allowance for every one.
    structural_diagnostic_occurrences: tuple[DiagnosticOccurrence, ...] = ()
    # Number of marked warning lines emitted by EnergyPlus's display-only
    # warning API.  EnergyPlus does not identify those lines in eplusout.err,
    # so only the aggregate is recoverable from its authoritative final count.
    display_only_warning_count: int = 0
    # Exact, unadjusted counts printed by EnergyPlus's recurring-error
    # summary.  Together with display_only_warning_count these authenticate
    # the final warning-counter reconciliation without changing the policy
    # budgets carried by recurring_warning_occurrences.
    recurring_summary_occurrences: tuple[DiagnosticOccurrence, ...] = ()

    def __post_init__(self) -> None:
        # Reject mutable/container subclasses and own fresh exact occurrence
        # records before checking reconciliation.  The helper is late-bound;
        # it is defined below this data model before any instance is created.
        values = _canonical_diagnostics_values(self)
        for field_name in _DIAGNOSTICS_OCCURRENCE_FIELDS:
            object.__setattr__(self, field_name, values[field_name])
        violations = _warning_occurrence_violations(self)
        if violations:
            raise ValueError("; ".join(violations))

    def to_dict(self) -> dict[str, Any]:
        snapshot = canonical_energyplus_diagnostics_snapshot(self)
        return json.loads(snapshot.canonical_json.decode("utf-8"))


class CanonicalDiagnosticsSnapshot(NamedTuple):
    """Immutable canonical JSON identity of one exact diagnostics snapshot."""

    canonical_json: bytes
    sha256: str


_DIAGNOSTICS_BOOL_FIELDS = (
    "completed_successfully",
    "completion_records_agree",
    "err_completion_present",
    "end_completion_present",
)
_DIAGNOSTICS_INTEGER_FIELDS = (
    "severe_error_count",
    "fatal_error_count",
    "accounted_warning_count",
    "display_only_warning_count",
)
_DIAGNOSTICS_OPTIONAL_INTEGER_FIELDS = (
    "total_warning_count",
    "unattributed_warning_count",
)
_DIAGNOSTICS_OCCURRENCE_FIELDS = (
    "warning_occurrences",
    "recurring_warning_occurrences",
    "severe_occurrences",
    "fatal_occurrences",
    "structural_diagnostic_occurrences",
    "recurring_summary_occurrences",
)


def _exact_occurrence_tuple(value: Any, *, field_name: str) -> tuple[DiagnosticOccurrence, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    owned: list[DiagnosticOccurrence] = []
    for index, occurrence in enumerate(value):
        if type(occurrence) is not DiagnosticOccurrence:
            raise TypeError(f"{field_name}[{index}] must be an exact DiagnosticOccurrence")
        signature = occurrence.signature
        count = occurrence.count
        if type(signature) is not str or not signature or signature != _signature(signature):
            raise ValueError(
                f"{field_name}[{index}].signature must be an exact, non-empty, whitespace-normalized string"
            )
        if type(count) is not int or count < 0:
            raise ValueError(f"{field_name}[{index}].count must be a non-negative integer")
        # Never retain the caller's record: frozen dataclasses can still be
        # changed through object.__setattr__ by code holding an alias.
        owned.append(DiagnosticOccurrence(signature=signature, count=count))
    return tuple(owned)


def _canonical_diagnostics_values(diagnostics: EnergyPlusDiagnostics) -> dict[str, Any]:
    if type(diagnostics) is not EnergyPlusDiagnostics:
        raise TypeError("diagnostics must be an exact EnergyPlusDiagnostics instance")

    schema_version = diagnostics.schema_version
    if type(schema_version) is not str or schema_version not in {
        DIAGNOSTICS_SCHEMA_VERSION,
        LEGACY_DIAGNOSTICS_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported diagnostics schema {schema_version!r}")
    energyplus_version = diagnostics.energyplus_version
    if energyplus_version is not None and type(energyplus_version) is not str:
        raise ValueError("energyplus_version must be null or an exact string")

    values: dict[str, Any] = {
        "schema_version": schema_version,
        "energyplus_version": energyplus_version,
    }
    for field_name in _DIAGNOSTICS_BOOL_FIELDS:
        value = getattr(diagnostics, field_name)
        if type(value) is not bool:
            raise TypeError(f"{field_name} must be a boolean")
        values[field_name] = value
    for field_name in _DIAGNOSTICS_INTEGER_FIELDS:
        value = getattr(diagnostics, field_name)
        if type(value) is not int or value < 0:
            raise TypeError(f"{field_name} must be a non-negative integer")
        values[field_name] = value
    for field_name in _DIAGNOSTICS_OPTIONAL_INTEGER_FIELDS:
        value = getattr(diagnostics, field_name)
        if value is not None and (type(value) is not int or value < 0):
            raise TypeError(f"{field_name} must be null or a non-negative integer")
        values[field_name] = value
    for field_name in _DIAGNOSTICS_OCCURRENCE_FIELDS:
        values[field_name] = _exact_occurrence_tuple(
            getattr(diagnostics, field_name),
            field_name=field_name,
        )
    return values


def _occurrence_payload(occurrences: tuple[DiagnosticOccurrence, ...]) -> list[dict[str, Any]]:
    return [{"signature": occurrence.signature, "count": occurrence.count} for occurrence in occurrences]


def _diagnostics_payload(diagnostics: EnergyPlusDiagnostics) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": diagnostics.schema_version,
        "energyplus_version": diagnostics.energyplus_version,
        "completed_successfully": diagnostics.completed_successfully,
        "completion_records_agree": diagnostics.completion_records_agree,
        "err_completion_present": diagnostics.err_completion_present,
        "end_completion_present": diagnostics.end_completion_present,
        "total_warning_count": diagnostics.total_warning_count,
        "severe_error_count": diagnostics.severe_error_count,
        "fatal_error_count": diagnostics.fatal_error_count,
        "warning_occurrences": _occurrence_payload(diagnostics.warning_occurrences),
        "recurring_warning_occurrences": _occurrence_payload(diagnostics.recurring_warning_occurrences),
        "severe_occurrences": _occurrence_payload(diagnostics.severe_occurrences),
        "fatal_occurrences": _occurrence_payload(diagnostics.fatal_occurrences),
        "accounted_warning_count": diagnostics.accounted_warning_count,
        "unattributed_warning_count": diagnostics.unattributed_warning_count,
        "structural_diagnostic_occurrences": _occurrence_payload(diagnostics.structural_diagnostic_occurrences),
        "display_only_warning_count": diagnostics.display_only_warning_count,
        "recurring_summary_occurrences": _occurrence_payload(diagnostics.recurring_summary_occurrences),
    }
    if diagnostics.schema_version == LEGACY_DIAGNOSTICS_SCHEMA_VERSION:
        payload.pop("structural_diagnostic_occurrences")
        payload.pop("display_only_warning_count")
        payload.pop("recurring_summary_occurrences")
    return payload


def _canonical_snapshot_components(
    diagnostics: EnergyPlusDiagnostics,
) -> tuple[EnergyPlusDiagnostics, CanonicalDiagnosticsSnapshot]:
    # Capture every field once into fresh exact base records.  The caller's
    # object is never consulted again after this point.
    owned = EnergyPlusDiagnostics(**_canonical_diagnostics_values(diagnostics))
    canonical_json = json.dumps(
        _diagnostics_payload(owned),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return owned, CanonicalDiagnosticsSnapshot(
        canonical_json=canonical_json,
        sha256=hashlib.sha256(canonical_json).hexdigest(),
    )


def canonical_energyplus_diagnostics_snapshot(
    diagnostics: EnergyPlusDiagnostics,
) -> CanonicalDiagnosticsSnapshot:
    """Return immutable canonical bytes for the fields present at call entry."""

    _owned, snapshot = _canonical_snapshot_components(diagnostics)
    return snapshot


@dataclass(frozen=True, slots=True)
class WarningAllowance:
    """Maximum counts permitted for one exact warning/structural signature."""

    signature: str
    max_occurrences: int
    max_recurring_occurrences: int = 0

    def __post_init__(self) -> None:
        if not self.signature or self.signature != _signature(self.signature):
            raise ValueError("allowlisted signatures must be non-empty and whitespace-normalized")
        for name, value in (
            ("max_occurrences", self.max_occurrences),
            ("max_recurring_occurrences", self.max_recurring_occurrences),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class DiagnosticsAllowlist:
    """Warning policy scoped to exact building bytes, weather bytes, and EnergyPlus version."""

    asset_sha256: str
    weather_sha256: str
    energyplus_version: str
    warning_allowances: tuple[WarningAllowance, ...] = ()
    schema_version: str = DIAGNOSTICS_ALLOWLIST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DIAGNOSTICS_ALLOWLIST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported diagnostics policy schema {self.schema_version!r}; "
                f"expected {DIAGNOSTICS_ALLOWLIST_SCHEMA_VERSION!r}"
            )
        for field_name in ("asset_sha256", "weather_sha256"):
            digest = getattr(self, field_name).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{field_name} must contain exactly 64 hexadecimal characters")
            object.__setattr__(self, field_name, digest)
        if not self.energyplus_version.strip():
            raise ValueError("energyplus_version cannot be empty")
        signatures = [allowance.signature for allowance in self.warning_allowances]
        if len(signatures) != len(set(signatures)):
            raise ValueError("warning_allowances cannot contain duplicate signatures")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-safe policy used for one acceptance decision."""

        return {
            "schema_version": self.schema_version,
            "asset_sha256": self.asset_sha256,
            "weather_sha256": self.weather_sha256,
            "energyplus_version": self.energyplus_version,
            "warning_allowances": [asdict(item) for item in self.warning_allowances],
        }

    def canonical_json(self) -> str:
        """Serialize the immutable policy deterministically for hashing."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def sha256(self) -> str:
        """Content identity of the exact canonical policy."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DiagnosticsGate:
    """Mode and exact model/weather/version policies used by benchmark evaluation.

    Engineering evaluation records diagnostics when they are available but
    never rejects an otherwise valid rollout because diagnostics are absent or
    malformed. Scientific evaluation fails closed and requires exactly one
    allowlist scoped to the task's immutable building and weather bytes and
    the recorded EnergyPlus version.
    """

    mode: str = "engineering"
    allowlists: tuple[DiagnosticsAllowlist, ...] = ()

    def __post_init__(self) -> None:
        mode = self.mode.casefold().strip()
        if mode not in DIAGNOSTICS_MODES:
            raise ValueError(f"diagnostics mode must be one of {sorted(DIAGNOSTICS_MODES)!r}")
        policies = tuple(self.allowlists)
        scopes = [(policy.asset_sha256, policy.weather_sha256, policy.energyplus_version) for policy in policies]
        if len(scopes) != len(set(scopes)):
            raise ValueError("diagnostics allowlists cannot contain duplicate asset/weather/version scopes")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "allowlists", policies)

    @property
    def scientific(self) -> bool:
        return self.mode == "scientific"

    def policy_for(
        self,
        *,
        asset_sha256: str | None,
        weather_sha256: str | None,
        energyplus_version: str,
    ) -> DiagnosticsAllowlist | None:
        """Return the one exact scientific policy, or fail closed when absent."""

        if not self.scientific:
            return None
        if asset_sha256 is None:
            raise ScientificDiagnosticsError("scientific diagnostics require a checksum-pinned building asset")
        if weather_sha256 is None:
            raise ScientificDiagnosticsError("scientific diagnostics require a checksum-pinned weather asset")
        matches = [
            policy
            for policy in self.allowlists
            if policy.asset_sha256 == asset_sha256.lower()
            and policy.weather_sha256 == weather_sha256.lower()
            and policy.energyplus_version == energyplus_version
        ]
        if len(matches) != 1:
            raise ScientificDiagnosticsError(
                "scientific diagnostics require exactly one policy for "
                f"asset SHA256 {asset_sha256.lower()!r}, weather SHA256 "
                f"{weather_sha256.lower()!r}, and EnergyPlus {energyplus_version!r}; "
                f"found {len(matches)}"
            )
        return matches[0]


class ScientificDiagnosticsResult(NamedTuple):
    """Policy decision bound to the exact canonical diagnostics bytes evaluated."""

    accepted: bool
    violations: tuple[str, ...]
    canonical_diagnostics_json: bytes | None = None
    canonical_diagnostics_sha256: str | None = None


class ScientificDiagnosticsError(RuntimeError):
    """Raised when a run fails the scientific diagnostics policy."""


@dataclass(frozen=True, slots=True)
class _CompletionRecord:
    succeeded: bool
    warnings: int
    severe_errors: int


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _exact_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an exact JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{field} keys must be exact strings")
    return dict(value)


def _integer(value: Any, *, field: str, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _boolean(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _occurrences(value: Any, *, field: str) -> tuple[DiagnosticOccurrence, ...]:
    if type(value) is not list:
        raise ValueError(f"{field} must be an exact JSON list")
    result = []
    seen_signatures: set[str] = set()
    for index, item in enumerate(value):
        occurrence = _exact_mapping(item, field=f"{field}[{index}]")
        if set(occurrence) != {"signature", "count"}:
            raise ValueError(f"{field}[{index}] must contain exactly signature and count")
        signature = occurrence.get("signature")
        if type(signature) is not str:
            raise ValueError(f"{field}[{index}].signature must be an exact string")
        count = _integer(occurrence.get("count"), field=f"{field}[{index}].count")
        assert count is not None
        normalized_signature = _signature(signature)
        if normalized_signature in seen_signatures:
            raise ValueError(f"{field} contains duplicate signature {normalized_signature!r}")
        if signature != normalized_signature or not normalized_signature:
            raise ValueError(f"{field}[{index}].signature must be non-empty and whitespace-normalized")
        seen_signatures.add(normalized_signature)
        result.append(DiagnosticOccurrence(signature=normalized_signature, count=count))
    signatures = [item.signature for item in result]
    if signatures != sorted(signatures):
        raise ValueError(f"{field} must be sorted by signature")
    return tuple(result)


def _severity_occurrence_violations(
    diagnostics: EnergyPlusDiagnostics,
) -> tuple[str, ...]:
    """Reconcile explicit severe/fatal occurrences with their recorded totals."""

    violations: list[str] = []
    for field, occurrences in (
        ("severe_occurrences", diagnostics.severe_occurrences),
        ("fatal_occurrences", diagnostics.fatal_occurrences),
    ):
        seen_signatures: set[str] = set()
        signatures = [occurrence.signature for occurrence in occurrences]
        if signatures != sorted(signatures):
            violations.append(f"{field} must be sorted by signature")
        for occurrence in occurrences:
            signature = _signature(occurrence.signature)
            if not signature or signature != occurrence.signature:
                violations.append(f"{field} contains a non-canonical signature {occurrence.signature!r}")
            if signature in seen_signatures:
                violations.append(f"{field} contains duplicate signature {signature!r}")
            seen_signatures.add(signature)
            if occurrence.count <= 0:
                violations.append(f"{field} occurrence {signature!r} must have a positive count")

    severe_total = sum(item.count for item in diagnostics.severe_occurrences)
    if diagnostics.severe_error_count != severe_total:
        violations.append(
            "severe_error_count does not equal the severe_occurrences total: "
            f"recorded={diagnostics.severe_error_count}, computed={severe_total}"
        )

    # EnergyPlus's completion record names a fatal termination but does not
    # provide a numeric fatal total.  Preserve the existing one-count fallback
    # only when both completion records agree that the run terminated.
    completion_fatal = int(
        diagnostics.err_completion_present
        and diagnostics.end_completion_present
        and diagnostics.completion_records_agree
        and not diagnostics.completed_successfully
    )
    fatal_total = max(
        sum(item.count for item in diagnostics.fatal_occurrences),
        completion_fatal,
    )
    if diagnostics.fatal_error_count != fatal_total:
        violations.append(
            "fatal_error_count does not equal the fatal_occurrences/termination total: "
            f"recorded={diagnostics.fatal_error_count}, computed={fatal_total}"
        )
    return tuple(violations)


def _warning_occurrence_violations(
    diagnostics: EnergyPlusDiagnostics,
) -> tuple[str, ...]:
    """Validate exact warning structures and their authoritative counter.

    EnergyPlus has two APIs that print the same ``** Warning **`` prefix:
    ``ShowWarningError`` increments the final warning counter, while
    ``ShowWarningMessage`` deliberately does not and is commonly paired with
    a recurring warning.  The error log does not identify which API produced
    a marked line.  Schema 2 therefore preserves every marked/recurring line
    exactly and records only the aggregate number of display-only marked
    lines needed to reconcile with EnergyPlus's final counter.
    """

    violations: list[str] = []
    for field, occurrences in (
        ("warning_occurrences", diagnostics.warning_occurrences),
        ("recurring_warning_occurrences", diagnostics.recurring_warning_occurrences),
        ("recurring_summary_occurrences", diagnostics.recurring_summary_occurrences),
        ("structural_diagnostic_occurrences", diagnostics.structural_diagnostic_occurrences),
    ):
        seen_signatures: set[str] = set()
        signatures = [occurrence.signature for occurrence in occurrences]
        if signatures != sorted(signatures):
            violations.append(f"{field} must be sorted by signature")
        for occurrence in occurrences:
            signature = _signature(occurrence.signature)
            if not signature or signature != occurrence.signature:
                violations.append(f"{field} contains a non-canonical signature {occurrence.signature!r}")
            if signature in seen_signatures:
                violations.append(f"{field} contains duplicate signature {signature!r}")
            seen_signatures.add(signature)
            minimum = 0 if field == "recurring_warning_occurrences" else 1
            if occurrence.count < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                violations.append(f"{field} occurrence {signature!r} must have a {qualifier} count")

    display_only = diagnostics.display_only_warning_count
    if isinstance(display_only, bool) or not isinstance(display_only, int) or display_only < 0:
        violations.append("display_only_warning_count must be a non-negative integer")
        display_only = 0
    marked_total = sum(item.count for item in diagnostics.warning_occurrences)
    policy_recurring_total = sum(item.count for item in diagnostics.recurring_warning_occurrences)
    if diagnostics.schema_version == LEGACY_DIAGNOSTICS_SCHEMA_VERSION:
        if diagnostics.structural_diagnostic_occurrences:
            violations.append("legacy diagnostics cannot contain structural_diagnostic_occurrences")
        if display_only:
            violations.append("legacy diagnostics cannot contain display_only_warning_count")
        if diagnostics.recurring_summary_occurrences:
            violations.append("legacy diagnostics cannot contain recurring_summary_occurrences")
        computed_accounted = marked_total + policy_recurring_total
    elif diagnostics.schema_version == DIAGNOSTICS_SCHEMA_VERSION:
        policy_recurring = {item.signature: item.count for item in diagnostics.recurring_warning_occurrences}
        summary_recurring = {item.signature: item.count for item in diagnostics.recurring_summary_occurrences}
        if policy_recurring.keys() != summary_recurring.keys():
            violations.append(
                "recurring warning and exact summary signatures do not match: "
                f"policy={sorted(policy_recurring)!r}, summary={sorted(summary_recurring)!r}"
            )
        for signature in policy_recurring.keys() & summary_recurring.keys():
            expected_summary = policy_recurring[signature] + 1
            if summary_recurring[signature] != expected_summary:
                violations.append(
                    f"recurring summary {signature!r} does not equal the policy count plus its visible first occurrence: "
                    f"summary={summary_recurring[signature]}, policy={policy_recurring[signature]}"
                )
        exact_recurring_total = sum(summary_recurring.values())
        if display_only > marked_total:
            violations.append(
                "display_only_warning_count exceeds the marked warning occurrence total: "
                f"recorded={display_only}, marked={marked_total}"
            )
        computed_accounted = marked_total + exact_recurring_total - display_only
    else:
        violations.append(f"unsupported diagnostics schema {diagnostics.schema_version!r}")
        computed_accounted = marked_total + policy_recurring_total

    if diagnostics.accounted_warning_count != computed_accounted:
        violations.append(
            "accounted_warning_count does not equal the reconciled warning occurrence total: "
            f"recorded={diagnostics.accounted_warning_count}, computed={computed_accounted}"
        )
    if diagnostics.total_warning_count is None:
        if diagnostics.unattributed_warning_count is not None:
            violations.append("unattributed_warning_count must be null when total_warning_count is null")
    else:
        computed_unattributed = diagnostics.total_warning_count - computed_accounted
        if computed_unattributed < 0:
            violations.append(
                "reconciled warning occurrence counts exceed total_warning_count: "
                f"summary={diagnostics.total_warning_count}, accounted={computed_accounted}"
            )
        elif diagnostics.unattributed_warning_count != computed_unattributed:
            violations.append(
                "unattributed_warning_count does not reconcile with total_warning_count: "
                f"recorded={diagnostics.unattributed_warning_count}, computed={computed_unattributed}"
            )
    return tuple(violations)


def energyplus_diagnostics_from_mapping(value: Mapping[str, Any]) -> EnergyPlusDiagnostics:
    """Validate and reconstruct a persisted diagnostics payload."""

    payload = _exact_mapping(value, field="diagnostics")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not str:
        raise ValueError("schema_version must be an exact string")
    if schema_version not in {DIAGNOSTICS_SCHEMA_VERSION, LEGACY_DIAGNOSTICS_SCHEMA_VERSION}:
        raise ValueError(
            f"unsupported diagnostics schema {schema_version!r}; expected "
            f"{DIAGNOSTICS_SCHEMA_VERSION!r} or archived {LEGACY_DIAGNOSTICS_SCHEMA_VERSION!r}"
        )
    common_fields = {
        "schema_version",
        "energyplus_version",
        "completed_successfully",
        "completion_records_agree",
        "err_completion_present",
        "end_completion_present",
        "total_warning_count",
        "severe_error_count",
        "fatal_error_count",
        "warning_occurrences",
        "recurring_warning_occurrences",
        "severe_occurrences",
        "fatal_occurrences",
        "accounted_warning_count",
        "unattributed_warning_count",
        # Persisted run artifacts add this identity outside the diagnostics
        # dataclass; its value is validated by their enclosing verifier.
        "executed_model_sha256",
    }
    versioned_fields = (
        {
            "structural_diagnostic_occurrences",
            "display_only_warning_count",
            "recurring_summary_occurrences",
        }
        if schema_version == DIAGNOSTICS_SCHEMA_VERSION
        else set()
    )
    unknown = set(payload) - common_fields - versioned_fields
    if unknown:
        raise ValueError(f"diagnostics contain unknown field(s): {', '.join(sorted(unknown))}")
    version = payload.get("energyplus_version")
    if version is not None and type(version) is not str:
        raise ValueError("energyplus_version must be null or an exact string")
    completed_successfully = _boolean(
        payload.get("completed_successfully"),
        field="completed_successfully",
    )
    completion_records_agree = _boolean(
        payload.get("completion_records_agree"),
        field="completion_records_agree",
    )
    err_completion_present = _boolean(
        payload.get("err_completion_present"),
        field="err_completion_present",
    )
    end_completion_present = _boolean(
        payload.get("end_completion_present"),
        field="end_completion_present",
    )
    total_warning_count = _integer(
        payload.get("total_warning_count"),
        field="total_warning_count",
        optional=True,
    )
    warning_occurrences = _occurrences(
        payload.get("warning_occurrences"),
        field="warning_occurrences",
    )
    recurring_warning_occurrences = _occurrences(
        payload.get("recurring_warning_occurrences"),
        field="recurring_warning_occurrences",
    )
    structural_diagnostic_occurrences = (
        _occurrences(
            payload.get("structural_diagnostic_occurrences"),
            field="structural_diagnostic_occurrences",
        )
        if schema_version == DIAGNOSTICS_SCHEMA_VERSION
        else ()
    )
    recurring_summary_occurrences = (
        _occurrences(
            payload.get("recurring_summary_occurrences"),
            field="recurring_summary_occurrences",
        )
        if schema_version == DIAGNOSTICS_SCHEMA_VERSION
        else ()
    )
    display_only_warning_count = (
        _integer(
            payload.get("display_only_warning_count"),
            field="display_only_warning_count",
        )
        if schema_version == DIAGNOSTICS_SCHEMA_VERSION
        else 0
    )
    assert display_only_warning_count is not None
    accounted_warning_count = _integer(
        payload.get("accounted_warning_count"),
        field="accounted_warning_count",
    )
    assert accounted_warning_count is not None
    unattributed_warning_count = _integer(
        payload.get("unattributed_warning_count"),
        field="unattributed_warning_count",
        optional=True,
    )

    severe_error_count = _integer(
        payload.get("severe_error_count"),
        field="severe_error_count",
    )
    fatal_error_count = _integer(
        payload.get("fatal_error_count"),
        field="fatal_error_count",
    )
    assert severe_error_count is not None and fatal_error_count is not None
    diagnostics = EnergyPlusDiagnostics(
        schema_version=schema_version,
        energyplus_version=version,
        completed_successfully=completed_successfully,
        completion_records_agree=completion_records_agree,
        err_completion_present=err_completion_present,
        end_completion_present=end_completion_present,
        total_warning_count=total_warning_count,
        severe_error_count=severe_error_count,
        fatal_error_count=fatal_error_count,
        warning_occurrences=warning_occurrences,
        recurring_warning_occurrences=recurring_warning_occurrences,
        severe_occurrences=_occurrences(payload.get("severe_occurrences"), field="severe_occurrences"),
        fatal_occurrences=_occurrences(payload.get("fatal_occurrences"), field="fatal_occurrences"),
        accounted_warning_count=accounted_warning_count,
        unattributed_warning_count=unattributed_warning_count,
        structural_diagnostic_occurrences=structural_diagnostic_occurrences,
        display_only_warning_count=display_only_warning_count,
        recurring_summary_occurrences=recurring_summary_occurrences,
    )
    violations = _warning_occurrence_violations(diagnostics) + _severity_occurrence_violations(diagnostics)
    if violations:
        raise ValueError("; ".join(violations))
    return diagnostics


def diagnostics_allowlist_from_mapping(value: Mapping[str, Any]) -> DiagnosticsAllowlist:
    """Construct one exact scientific diagnostics policy from JSON data."""

    payload = _mapping(value, field="diagnostics allowlist")
    unknown = set(payload) - {
        "schema_version",
        "asset_sha256",
        "weather_sha256",
        "energyplus_version",
        "warning_allowances",
    }
    if unknown:
        raise ValueError(f"diagnostics allowlist contains unknown field(s): {', '.join(sorted(unknown))}")
    schema_version = payload.get("schema_version")
    if schema_version != DIAGNOSTICS_ALLOWLIST_SCHEMA_VERSION:
        raise ValueError("diagnostics allowlist requires schema_version " f"{DIAGNOSTICS_ALLOWLIST_SCHEMA_VERSION!r}")
    asset_sha256 = payload.get("asset_sha256")
    weather_sha256 = payload.get("weather_sha256")
    energyplus_version = payload.get("energyplus_version")
    if (
        not isinstance(asset_sha256, str)
        or not isinstance(weather_sha256, str)
        or not isinstance(energyplus_version, str)
    ):
        raise ValueError(
            "diagnostics allowlist requires string asset_sha256, weather_sha256, " "and energyplus_version"
        )
    raw_allowances = payload.get("warning_allowances", [])
    if not isinstance(raw_allowances, list):
        raise ValueError("warning_allowances must be a JSON list")
    allowances = []
    for index, item in enumerate(raw_allowances):
        allowance = _mapping(item, field=f"warning_allowances[{index}]")
        unknown_allowance = set(allowance) - {
            "signature",
            "max_occurrences",
            "max_recurring_occurrences",
        }
        if unknown_allowance:
            raise ValueError(
                f"warning_allowances[{index}] contains unknown field(s): " f"{', '.join(sorted(unknown_allowance))}"
            )
        signature = allowance.get("signature")
        if not isinstance(signature, str):
            raise ValueError(f"warning_allowances[{index}].signature must be a string")
        maximum = _integer(allowance.get("max_occurrences"), field=f"warning_allowances[{index}].max_occurrences")
        recurring = _integer(
            allowance.get("max_recurring_occurrences", 0),
            field=f"warning_allowances[{index}].max_recurring_occurrences",
        )
        assert maximum is not None and recurring is not None
        allowances.append(WarningAllowance(signature, maximum, recurring))
    return DiagnosticsAllowlist(
        asset_sha256,
        weather_sha256,
        energyplus_version,
        tuple(allowances),
        schema_version,
    )


def load_diagnostics_allowlist(path: str | Path) -> DiagnosticsAllowlist:
    """Load one exact scientific diagnostics policy from a JSON file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read diagnostics allowlist {source}: {error}") from error
    return diagnostics_allowlist_from_mapping(_mapping(payload, field=str(source)))


def _completion_record(text: str) -> _CompletionRecord | None:
    matches = list(_COMPLETION_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return _CompletionRecord(
        succeeded=match.group(1).casefold() == "completed successfully",
        warnings=int(match.group(2)),
        severe_errors=int(match.group(3)),
    )


def _marked_occurrences(text: str) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    warnings: dict[str, int] = {}
    severe: dict[str, int] = {}
    fatal: dict[str, int] = {}
    destinations = {"warning": warnings, "severe": severe, "fatal": fatal}
    for line in text.splitlines():
        match = _MARKED_MESSAGE_RE.match(line)
        if match is None:
            continue
        destination = destinations[match.group(1).casefold()]
        signature = _signature(match.group(2))
        if signature:
            destination[signature] = destination.get(signature, 0) + 1
    return warnings, severe, fatal


def _recurring_warning_occurrences(text: str) -> dict[str, int]:
    """Return the exact counts printed for each recurring warning summary.

    The marked first line cannot safely be subtracted here.  EnergyPlus can
    print it with ``ShowWarningMessage`` (which deliberately does not advance
    the warning counter) or ``ShowWarningError`` (which does).  Reconciliation
    against the final authoritative count is therefore performed once, after
    all exact marked and recurring structures have been retained.
    """

    lines = text.splitlines()
    result: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = _RECURRING_WARNING_RE.match(line)
        if match is None:
            continue
        signature = _signature(match.group(1))
        if not signature:
            raise ValueError("EnergyPlus diagnostics contain an empty recurring warning signature")
        if signature in result:
            raise ValueError(f"EnergyPlus diagnostics contain duplicate recurring warning {signature!r}")
        for following in lines[index + 1 : index + 8]:
            count_match = _RECURRING_COUNT_RE.search(following)
            if count_match is not None:
                total_occurrences = int(count_match.group(1))
                if total_occurrences <= 0:
                    raise ValueError(f"EnergyPlus recurring warning {signature!r} has a non-positive occurrence count")
                result[signature] = total_occurrences
                break
        else:
            raise ValueError(f"EnergyPlus recurring warning {signature!r} has no exact occurrence count")
    return result



def recurring_warning_phase_counts(text: str) -> dict[str, dict[str, int | None]]:
    """Extract reported phase counters without weakening diagnostic acceptance.

    Missing phase lines remain unknown, not zero. Warmup and sizing counters
    are reported separately; no disjointness assumption is made. A non-warmup
    count is available only when EnergyPlus explicitly reports zero sizing
    occurrences. Counts refer to warning occurrences, not distinct timesteps.
    """
    totals = _recurring_warning_occurrences(text)
    lines = text.splitlines()
    result = {}
    starts = [(index, _signature(match.group(1))) for index, line in enumerate(lines)
              if (match := _RECURRING_WARNING_RE.match(line)) is not None]
    for position, (start, signature) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        phases = {"warmup": None, "sizing": None}
        for line in lines[start + 1:end]:
            match = re.search(r"during\s+(Warmup|Sizing)\s+(\d+)\s+times", line, re.IGNORECASE)
            if match is None:
                continue
            phase, count = match.group(1).lower(), int(match.group(2))
            if phases[phase] is not None:
                raise ValueError(f"duplicate {phase} counter for recurring warning {signature!r}")
            if count > totals[signature]:
                raise ValueError(f"{phase} counter exceeds total for recurring warning {signature!r}")
            phases[phase] = count
        result[signature] = {
            "total": totals[signature], **phases,
            "outside_warmup_with_no_sizing": (
                totals[signature] - phases["warmup"]
                if phases["warmup"] is not None and phases["sizing"] == 0 else None
            ),
        }
    return result


def _is_unused_schedules_candidate(line: str) -> bool:
    if _ANY_MARKED_DIAGNOSTIC_RE.match(line):
        return False
    folded = line.casefold()
    return any(
        token in folded
        for token in (
            "unused schedule",
            "unused week schedule",
            "unused day schedule",
            "displayunusedschedules",
        )
    )


def _unmarked_unused_schedules_occurrences(text: str) -> dict[str, int]:
    """Parse one exact display-only unused-schedules diagnostic block.

    EnergyPlus 24.1 emits either the historical two-line form or a four-line
    form that inserts exact week- and day-schedule subcounts, in that order,
    before the advisory.  These ``ShowMessage`` records do not advance the
    authoritative warning counter, so their exact normalized block remains a
    structural diagnostic rather than a warning occurrence.
    """

    lines = text.splitlines()
    result: dict[str, int] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _UNUSED_SCHEDULES_RE.match(line)
        if match is not None:
            if result:
                raise ValueError("EnergyPlus diagnostics contain duplicate unused-schedules warnings")

            count = int(match.group(1))
            signature_lines = [f"There are {count} unused schedules in input."]
            cursor = index + 1
            if cursor >= len(lines):
                raise ValueError("EnergyPlus diagnostics contain a malformed unused-schedules warning")

            if _UNUSED_SCHEDULES_ADVISORY_RE.match(lines[cursor]) is None:
                week_match = _UNUSED_WEEK_SCHEDULES_RE.match(lines[cursor])
                if week_match is None or cursor + 2 >= len(lines):
                    raise ValueError("EnergyPlus diagnostics contain a malformed unused-schedules warning")
                day_match = _UNUSED_DAY_SCHEDULES_RE.match(lines[cursor + 1])
                if day_match is None:
                    raise ValueError("EnergyPlus diagnostics contain a malformed unused-schedules warning")
                signature_lines.extend(
                    (
                        f"There are {int(week_match.group(1))} unused week schedules in input.",
                        f"There are {int(day_match.group(1))} unused day schedules in input.",
                    )
                )
                cursor += 2

            if _UNUSED_SCHEDULES_ADVISORY_RE.match(lines[cursor]) is None:
                raise ValueError("EnergyPlus diagnostics contain a malformed unused-schedules warning")
            signature_lines.append("Use Output:Diagnostics,DisplayUnusedSchedules; to see them.")
            result[" ".join(signature_lines)] = 1
            index = cursor + 1
            continue
        if _UNUSED_SCHEDULES_ADVISORY_RE.match(line) is not None or _is_unused_schedules_candidate(line):
            raise ValueError("EnergyPlus diagnostics contain a malformed unused-schedules warning")
        index += 1
    return result


def parse_energyplus_diagnostics(err_text: str, end_text: str) -> EnergyPlusDiagnostics:
    """Parse a complete EnergyPlus error log and end record.

    The function performs no file-system access.  A missing or malformed end
    record is represented as a failed completion rather than being silently
    accepted.
    """

    if type(err_text) is not str or type(end_text) is not str:
        raise TypeError("err_text and end_text must be exact strings")
    version_match = _VERSION_RE.search(err_text)
    energyplus_version = _signature(version_match.group(1)) if version_match else None
    err_completion = _completion_record(err_text)
    end_completion = _completion_record(end_text)
    records_agree = err_completion is not None and end_completion is not None and err_completion == end_completion
    completed = records_agree and err_completion is not None and err_completion.succeeded

    warnings, severe, fatal = _marked_occurrences(err_text)
    structural_diagnostics = _unmarked_unused_schedules_occurrences(err_text)
    recurring_summary = _recurring_warning_occurrences(err_text)
    recurring = {signature: count - 1 for signature, count in recurring_summary.items()}
    final_severe = max(
        sum(severe.values()),
        err_completion.severe_errors if err_completion is not None else 0,
        end_completion.severe_errors if end_completion is not None else 0,
    )
    final_fatal = max(
        sum(fatal.values()),
        int(err_completion is not None and not err_completion.succeeded),
        int(end_completion is not None and not end_completion.succeeded),
    )
    total_warnings = err_completion.warnings if err_completion is not None else None
    raw_warning_total = sum(warnings.values()) + sum(recurring_summary.values())
    display_only_warning_count = 0
    if total_warnings is not None and raw_warning_total > total_warnings:
        display_only_warning_count = raw_warning_total - total_warnings
        # EnergyPlus can emit any number of marked ``ShowWarningMessage``
        # records without advancing its authoritative warning counter. They
        # are not limited to the first display line of recurring summaries;
        # sizing, for example, can emit one such record per zone.
        if display_only_warning_count > sum(warnings.values()):
            raise ValueError(
                "EnergyPlus warning structures exceed the final warning count and cannot be "
                "reconciled as display-only marked messages: "
                f"marked={sum(warnings.values())}, recurring={sum(recurring_summary.values())}, "
                f"recurring_summaries={len(recurring_summary)}, summary={total_warnings}"
            )
    accounted_warnings = raw_warning_total - display_only_warning_count
    unattributed = None if total_warnings is None else total_warnings - accounted_warnings
    assert unattributed is None or unattributed >= 0
    return EnergyPlusDiagnostics(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        energyplus_version=energyplus_version,
        completed_successfully=completed,
        completion_records_agree=records_agree,
        err_completion_present=err_completion is not None,
        end_completion_present=end_completion is not None,
        total_warning_count=total_warnings,
        severe_error_count=final_severe,
        fatal_error_count=final_fatal,
        warning_occurrences=_sorted_counts(warnings),
        recurring_warning_occurrences=_sorted_counts(recurring),
        severe_occurrences=_sorted_counts(severe),
        fatal_occurrences=_sorted_counts(fatal),
        accounted_warning_count=accounted_warnings,
        unattributed_warning_count=unattributed,
        structural_diagnostic_occurrences=_sorted_counts(structural_diagnostics),
        display_only_warning_count=display_only_warning_count,
        recurring_summary_occurrences=_sorted_counts(recurring_summary),
    )


def evaluate_scientific_diagnostics(
    diagnostics: EnergyPlusDiagnostics,
    *,
    asset_sha256: str,
    weather_sha256: str,
    allowlist: DiagnosticsAllowlist,
) -> ScientificDiagnosticsResult:
    """Fail closed on incomplete, inconsistent, unknown, or excessive diagnostics."""

    try:
        diagnostics, snapshot = _canonical_snapshot_components(diagnostics)
    except (AttributeError, TypeError, ValueError) as error:
        return ScientificDiagnosticsResult(
            accepted=False,
            violations=(f"invalid or modified diagnostics object: {error}",),
        )

    violations: list[str] = []
    if diagnostics.schema_version != DIAGNOSTICS_SCHEMA_VERSION:
        violations.append(
            f"diagnostics schema {diagnostics.schema_version!r} is not supported; "
            f"expected {DIAGNOSTICS_SCHEMA_VERSION!r}"
        )
    digest = asset_sha256.lower()
    if digest != allowlist.asset_sha256:
        violations.append(f"asset SHA256 {digest!r} does not match allowlist scope {allowlist.asset_sha256!r}")
    weather_digest = weather_sha256.lower()
    if weather_digest != allowlist.weather_sha256:
        violations.append(
            f"weather SHA256 {weather_digest!r} does not match allowlist scope " f"{allowlist.weather_sha256!r}"
        )
    if diagnostics.energyplus_version != allowlist.energyplus_version:
        violations.append(
            "EnergyPlus version "
            f"{diagnostics.energyplus_version!r} does not match allowlist scope {allowlist.energyplus_version!r}"
        )
    if not diagnostics.err_completion_present:
        violations.append("EnergyPlus err log has no completion record")
    if not diagnostics.end_completion_present:
        violations.append("EnergyPlus end file has no completion record")
    if not diagnostics.completion_records_agree:
        violations.append("EnergyPlus err/end completion records do not agree")
    if not diagnostics.completed_successfully:
        violations.append("EnergyPlus did not complete successfully")
    if diagnostics.severe_error_count:
        violations.append(f"EnergyPlus reported {diagnostics.severe_error_count} severe error(s)")
    if diagnostics.fatal_error_count:
        violations.append(f"EnergyPlus reported {diagnostics.fatal_error_count} fatal error(s)")
    violations.extend(_severity_occurrence_violations(diagnostics))
    violations.extend(_warning_occurrence_violations(diagnostics))
    if diagnostics.total_warning_count is None:
        violations.append("EnergyPlus warning summary is missing")
    if diagnostics.unattributed_warning_count != 0:
        violations.append(
            "warning summary could not be fully attributed: "
            f"summary={diagnostics.total_warning_count}, accounted={diagnostics.accounted_warning_count}"
        )

    allowances = {allowance.signature: allowance for allowance in allowlist.warning_allowances}
    for occurrence in diagnostics.warning_occurrences:
        allowance = allowances.get(occurrence.signature)
        if allowance is None:
            violations.append(f"unknown warning signature: {occurrence.signature}")
        elif occurrence.count > allowance.max_occurrences:
            violations.append(
                f"warning {occurrence.signature!r} occurred {occurrence.count} time(s); "
                f"allowlist maximum is {allowance.max_occurrences}"
            )
    for occurrence in diagnostics.recurring_warning_occurrences:
        allowance = allowances.get(occurrence.signature)
        if allowance is None:
            violations.append(f"unknown recurring warning signature: {occurrence.signature}")
        elif occurrence.count > allowance.max_recurring_occurrences:
            violations.append(
                f"recurring warning {occurrence.signature!r} occurred {occurrence.count} time(s); "
                f"allowlist maximum is {allowance.max_recurring_occurrences}"
            )
    for occurrence in diagnostics.structural_diagnostic_occurrences:
        allowance = allowances.get(occurrence.signature)
        if allowance is None:
            violations.append(f"unknown structural diagnostic signature: {occurrence.signature}")
        elif occurrence.count > allowance.max_occurrences:
            violations.append(
                f"structural diagnostic {occurrence.signature!r} occurred {occurrence.count} time(s); "
                f"allowlist maximum is {allowance.max_occurrences}"
            )
    return ScientificDiagnosticsResult(
        accepted=not violations,
        violations=tuple(violations),
        canonical_diagnostics_json=snapshot.canonical_json,
        canonical_diagnostics_sha256=snapshot.sha256,
    )


def require_scientific_diagnostics(
    diagnostics: EnergyPlusDiagnostics,
    *,
    asset_sha256: str,
    weather_sha256: str,
    allowlist: DiagnosticsAllowlist,
) -> ScientificDiagnosticsResult:
    """Return the accepted immutable snapshot, or raise on policy failure."""

    result = evaluate_scientific_diagnostics(
        diagnostics,
        asset_sha256=asset_sha256,
        weather_sha256=weather_sha256,
        allowlist=allowlist,
    )
    if not result.accepted:
        raise ScientificDiagnosticsError("; ".join(result.violations))
    assert result.canonical_diagnostics_json is not None
    assert result.canonical_diagnostics_sha256 is not None
    return result


def sha256_path(path: str | Path) -> str:
    """Return a lowercase SHA256 digest without loading the file at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def persist_energyplus_diagnostics(
    output_dir: str | Path,
    *,
    model_path: str | Path,
) -> Path:
    """Parse one completed run and atomically write its diagnostics JSON.

    ``model_path`` is the exact, potentially RunPeriod-materialized model that
    EnergyPlus executed.  Scientific allowlists remain scoped separately to
    the immutable source-asset checksum supplied to the policy evaluator.
    """

    directory = Path(output_dir)
    err_path = directory / "eplusout.err"
    end_path = directory / "eplusout.end"
    diagnostics = parse_energyplus_diagnostics(
        err_path.read_text(encoding="utf-8", errors="replace"),
        end_path.read_text(encoding="utf-8", errors="replace"),
    )
    payload = diagnostics.to_dict()
    payload["executed_model_sha256"] = sha256_path(model_path)
    destination = directory / DIAGNOSTICS_FILENAME
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = [
    "DIAGNOSTICS_ALLOWLIST_SCHEMA_VERSION",
    "DIAGNOSTICS_MODES",
    "DIAGNOSTICS_FILENAME",
    "DIAGNOSTICS_SCHEMA_VERSION",
    "LEGACY_DIAGNOSTICS_SCHEMA_VERSION",
    "CanonicalDiagnosticsSnapshot",
    "DiagnosticOccurrence",
    "DiagnosticsAllowlist",
    "DiagnosticsGate",
    "EnergyPlusDiagnostics",
    "ScientificDiagnosticsError",
    "ScientificDiagnosticsResult",
    "WarningAllowance",
    "canonical_energyplus_diagnostics_snapshot",
    "diagnostics_allowlist_from_mapping",
    "energyplus_diagnostics_from_mapping",
    "evaluate_scientific_diagnostics",
    "load_diagnostics_allowlist",
    "parse_energyplus_diagnostics",
    "persist_energyplus_diagnostics",
    "require_scientific_diagnostics",
    "sha256_path",
]
