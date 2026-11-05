"""Lazy, optional CUDA kernels for inference primitive acquisition.

No extension is imported/compiled during config checks or CPU execution. JIT
artifacts live in PyTorch's extension cache, not in the source tree.
"""
from functools import lru_cache
from pathlib import Path
import warnings

import torch


@lru_cache(maxsize=1)
def _load():
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name('csrc')
    return load(
        name='primitiveudf_query_v3',
        sources=[str(source / 'primitive_query.cpp'), str(source / 'primitive_query.cu')],
        extra_cflags=['-O3'],
        # Preserve ordinary float arithmetic; fast-math/FMA changes selection
        # near degeneracy and score boundaries.
        extra_cuda_cflags=['-O3', '--fmad=false'],
        verbose=False,
    )


_failure = None


def extension(device, backend='auto'):
    global _failure
    if backend not in {'auto', 'cuda', 'torch'}:
        raise ValueError('primitive_geometry_backend must be auto, cuda or torch')
    if backend == 'torch' or torch.device(device).type != 'cuda':
        if backend == 'cuda':
            raise RuntimeError('The CUDA primitive backend requires CUDA tensors')
        return None
    if _failure is not None:
        if backend == 'cuda':
            raise RuntimeError('CUDA primitive extension is unavailable') from _failure
        return None
    try:
        return _load()
    except (RuntimeError, OSError, ImportError) as exc:
        _failure = exc
        if backend == 'cuda':
            raise
        warnings.warn(f'Primitive CUDA extension unavailable; using PyTorch: {exc}', RuntimeWarning)
        return None
