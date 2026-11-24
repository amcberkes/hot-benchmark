"""JSON-serializable Python, NumPy and CPU Torch random-generator state.

This state excludes simulator, environment and action-space generators. Restoring
it alone does not establish exact trajectory or training continuation.
"""
from __future__ import annotations

import random


def capture_random_state() -> dict:
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {
        'schema_version': 'hot-cpu-random-state/1.0.0',
        'python': random.getstate(),
        'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        'torch_cpu': torch.get_rng_state().tolist(),
    }


def restore_random_state(state: dict) -> None:
    """Validate every generator before changing any process-global state."""
    import numpy as np
    import torch
    if not isinstance(state, dict) or set(state) != {'schema_version', 'python', 'numpy', 'torch_cpu'}:
        raise ValueError('random state has unexpected fields')
    if state['schema_version'] != 'hot-cpu-random-state/1.0.0':
        raise ValueError('unsupported random state schema')
    try:
        python_state = (state['python'][0], tuple(state['python'][1]), state['python'][2])
        random.Random().setstate(python_state)
        raw_numpy = state['numpy']
        keys = raw_numpy[1]
        if any(type(value) is not int or not 0 <= value <= 2**32-1 for value in keys):
            raise ValueError('invalid NumPy generator keys')
        numpy_state = (raw_numpy[0], np.array(keys, dtype=np.uint32), *raw_numpy[2:])
        np.random.RandomState().set_state(numpy_state)
        raw_torch = state['torch_cpu']
        if any(type(value) is not int or not 0 <= value <= 255 for value in raw_torch):
            raise ValueError('invalid Torch generator bytes')
        torch_state = torch.tensor(raw_torch, dtype=torch.uint8)
        torch.Generator(device='cpu').set_state(torch_state)
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError('invalid saved random state') from error
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
