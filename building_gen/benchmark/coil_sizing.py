"""Fingerprint explicit cooling-coil component-sizing records in EnergyPlus EIO.

This diagnostic schema is separate from the legacy warmup contract. It does not
change any model, simulation, or legacy SingleSpeed fingerprint.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path


def cooling_coil_sizing_fingerprint(eio_text: str, *, coil_types: list[str]) -> dict:
    if not coil_types or any(not isinstance(t, str) or not t.startswith('Coil:Cooling:') or t != t.strip() for t in coil_types):
        raise ValueError('declare nonempty exact cooling-coil component types')
    if len(set(t.casefold() for t in coil_types)) != len(coil_types):
        raise ValueError('duplicate cooling-coil component type')
    expected = set(coil_types)
    counts = dict.fromkeys(sorted(expected), 0)
    selected = []
    identities = set()
    for line in eio_text.splitlines():
        row = next(csv.reader([line]))
        if not row or row[0].strip() != 'Component Sizing Information':
            continue
        if len(row) < 2 or row[1].strip() not in expected:
            continue
        if len(row) != 5 or any(not value.strip() for value in row[1:]):
            raise ValueError('malformed selected component-sizing record')
        identity = tuple(value.strip() for value in row[1:4])
        if identity in identities:
            raise ValueError('duplicate component-sizing quantity')
        identities.add(identity)
        counts[row[1].strip()] += 1
        selected.append(line)
    missing = [t for t, n in counts.items() if n == 0]
    if missing:
        raise ValueError(f'missing component-sizing records: {missing}')
    payload = ('\n'.join(selected) + '\n').encode('utf-8')
    return {'schema_version': 'hot-cooling-coil-sizing/1.0.0',
            'coil_types': sorted(expected), 'line_count': len(selected),
            'counts_by_type': counts, 'sha256': hashlib.sha256(payload).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eio', type=Path)
    parser.add_argument('--coil-type', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = args.eio.read_bytes()
    result = cooling_coil_sizing_fingerprint(payload.decode('utf-8'), coil_types=args.coil_type)
    result['source_eio_sha256'] = hashlib.sha256(payload).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
