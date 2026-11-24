"""Explicit, versioned component meters for audited HVAC energy boundaries.

Component membership must come from a reviewed EnergyPlus meter dictionary.
This module does not infer plant ownership or alter historical meter semantics.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Mapping

_RESOURCE_TYPES = {
    'electricity': 'Electricity', 'natural_gas': 'NaturalGas',
    'district_heating_water': 'DistrictHeatingWater',
    'district_heating_steam': 'DistrictHeatingSteam', 'district_cooling': 'DistrictCooling',
}


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip() or any(c in value for c in '\n\r\x00'):
        raise ValueError(f'{name} must be a nonempty stripped single-line string')
    return value


def prepare_component_meters(source_bytes: bytes, *, expected_source_sha256: str,
                             components: Mapping[str, list[dict[str, str]]],
                             boundary_description: str,
                             dictionary_sha256: str) -> tuple[dict, dict]:
    """Copy an exact source model and add non-overlapping explicit custom meters.

    Every component is a concrete key/output-variable pair. Wildcards and nested
    meter references are forbidden. A caller must establish physical membership
    and verify requested variables exist in the simulator; structural validation
    here cannot establish that energy coverage is complete.
    """
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != expected_source_sha256:
        raise ValueError('source model digest mismatch')
    if not isinstance(dictionary_sha256, str) or len(dictionary_sha256) != 64 or any(c not in '0123456789abcdef' for c in dictionary_sha256):
        raise ValueError('dictionary_sha256 must identify the reviewed meter dictionary')
    _text(boundary_description, 'boundary_description')
    if not isinstance(components, Mapping) or not components:
        raise ValueError('at least one carrier must have explicit components')
    normalized = {}; seen = set()
    for carrier, entries in sorted(components.items()):
        if carrier not in _RESOURCE_TYPES or not isinstance(entries, (list, tuple)) or not entries:
            raise ValueError('unknown carrier or empty component list')
        pairs = []
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {'key_name', 'output_variable_name'}:
                raise ValueError('components require exactly key_name and output_variable_name')
            key = _text(entry['key_name'], 'key_name')
            variable = _text(entry['output_variable_name'], 'output_variable_name')
            if '*' in key or '*' in variable or ':' in variable:
                raise ValueError('wildcards and meter references are not component variables')
            identity = (key.casefold(), variable.casefold())
            if identity in seen:
                raise ValueError('overlapping component membership would double-count energy')
            seen.add(identity)
            pairs.append({'key_name': key, 'output_variable_name': variable})
        normalized[carrier] = sorted(pairs, key=lambda x: (x['key_name'].casefold(), x['output_variable_name'].casefold()))
    contract = {'schema_version':'hot-component-energy-boundary/1.0.0',
                'source_model_sha256':source_sha, 'meter_dictionary_sha256':dictionary_sha256,
                'boundary_description':boundary_description, 'components':normalized}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    model = json.loads(source_bytes)
    if not isinstance(model, dict):
        raise ValueError('source model must be an epJSON object')
    model = deepcopy(model)
    meters = model.setdefault('Meter:Custom', {})
    outputs = model.setdefault('Output:Meter', {})
    if not isinstance(meters, dict) or not isinstance(outputs, dict):
        raise ValueError('invalid meter sections in source model')
    meter_names = {}
    for carrier, entries in normalized.items():
        # EnergyPlus canonicalizes custom meter names to uppercase in MTR output.
        name = f'HOT:COMPONENTHVAC:{carrier}:{digest[:16]}'.upper()
        if any(name.casefold() == existing.casefold() for section in ('Meter:Custom', 'Meter:CustomDecrement') for existing in model.get(section, {})):
            raise ValueError('custom meter already exists')
        meters[name] = {'resource_type':_RESOURCE_TYPES[carrier], 'variable_details':[
            {'key_name':e['key_name'], 'output_variable_or_meter_name':e['output_variable_name']} for e in entries]}
        output_name = f'HOT_component_{carrier}_{digest[:16]}'
        if any(output_name.casefold() == existing.casefold() for existing in outputs):
            raise ValueError('output meter request already exists')
        outputs[output_name] = {'key_name':name, 'reporting_frequency':'Timestep'}
        meter_names[carrier] = name
    return model, {**contract, 'boundary_sha256':digest, 'meter_names':meter_names,
                   'physical_coverage_verified':False}


def main(argv=None):
    """Prepare a model and receipt from a reviewed boundary JSON file."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--boundary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        specification = json.loads(args.boundary.read_text())
        if not isinstance(specification, dict):
            raise ValueError('boundary must be a JSON object')
        if 'schema_version' in specification:
            required = {'schema_version', 'source_model_sha256', 'meter_dictionary_sha256',
                        'boundary_description', 'components', 'boundary_sha256',
                        'meter_names', 'physical_coverage_verified'}
            if not required <= specification.keys() or specification.keys() - required - {'excluded_components'}:
                raise ValueError('invalid boundary contract fields')
            model, receipt = prepare_component_meters(
                args.source.read_bytes(),
                expected_source_sha256=specification['source_model_sha256'],
                dictionary_sha256=specification['meter_dictionary_sha256'],
                components=specification['components'],
                boundary_description=specification['boundary_description'])
            for key in required - {'physical_coverage_verified'}:
                if specification[key] != receipt[key]:
                    raise ValueError(f'boundary contract mismatch: {key}')
            if specification['physical_coverage_verified'] is not False:
                raise ValueError('preparation cannot certify physical energy coverage')
            if 'excluded_components' in specification:
                receipt['excluded_components'] = specification['excluded_components']
        else:
            model, receipt = prepare_component_meters(args.source.read_bytes(), **specification)
        encoded = (json.dumps(model, sort_keys=True, indent=2) + '\n').encode()
        receipt['prepared_model_sha256'] = hashlib.sha256(encoded).hexdigest()
        args.output_dir.mkdir(exist_ok=False)
        (args.output_dir / 'model.epJSON').write_bytes(encoded)
        (args.output_dir / 'energy_boundary.json').write_text(json.dumps(receipt, indent=2) + '\n')
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    print(args.output_dir / 'model.epJSON')
    return 0


def component_meter_sources(model: Mapping) -> dict[str, str] | None:
    """Resolve reserved custom meters from a content-pinned prepared model.

    This establishes declared sources, not physical completeness. All declared
    custom carriers must be available at runtime; none is optional.
    """
    import re
    sources = {}; boundary_ids = set(); components_seen = set()
    custom = model.get('Meter:Custom', {})
    if not isinstance(custom, Mapping):
        raise ValueError('invalid custom meter section')
    for name, fields in custom.items():
        _text(name, 'custom meter name')
        if not name.upper().startswith('HOT:COMPONENTHVAC:'):
            continue
        match = re.fullmatch(r'HOT:COMPONENTHVAC:([A-Z_]+):([0-9A-F]{16})', name)
        if not match:
            raise ValueError('component meter names must use their canonical uppercase identity')
        carrier = match[1].lower()
        if not isinstance(fields, Mapping):
            raise ValueError('invalid component meter fields')
        if carrier not in _RESOURCE_TYPES or fields.get('resource_type') != _RESOURCE_TYPES[carrier]:
            raise ValueError('component meter carrier/resource mismatch')
        if carrier in sources:
            raise ValueError('multiple component meters for one carrier')
        details = fields.get('variable_details')
        if not isinstance(details, list) or not details:
            raise ValueError('empty or invalid component meter')
        for entry in details:
            if not isinstance(entry, Mapping) or set(entry) != {'key_name', 'output_variable_or_meter_name'}:
                raise ValueError('invalid component variable declaration')
            key = _text(entry['key_name'], 'component key')
            variable = _text(entry['output_variable_or_meter_name'], 'component variable')
            if '*' in key or '*' in variable or ':' in variable:
                raise ValueError('component meters require explicit variables, not wildcard or meter references')
            identity = (key.casefold(), variable.casefold())
            if identity in components_seen:
                raise ValueError('overlapping component membership')
            components_seen.add(identity)
        sources[carrier] = name; boundary_ids.add(match[2])
    if not sources:
        return None
    if len(boundary_ids) != 1 or 'electricity' not in sources:
        raise ValueError('runtime component meters require one boundary identity and electricity')
    if set(sources) - {'electricity', 'natural_gas'}:
        raise ValueError('runtime component meters currently support electricity and natural gas only')
    return sources


if __name__ == '__main__':
    raise SystemExit(main())
