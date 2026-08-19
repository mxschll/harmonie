"""CUDA device detection.

The TensorFlow inside ``essentia-tensorflow`` has CUDA statically linked and its
only external CUDA dependency is the driver library, ``libcuda.so.1``. So GPU
support is a property of the host, not of the build: if the driver is loadable
and reports a device, TensorFlow uses it for inference.

This asks the driver directly rather than looking for ``/dev/nvidia*``, which
exists in cases where the runtime still cannot use it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("harmonie.gpu")

CUDA_SUCCESS = 0


def cuda_device_count() -> int:
    """Number of CUDA devices this process can use. 0 when there is no driver."""
    import ctypes

    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return 0
    try:
        if driver.cuInit(0) != CUDA_SUCCESS:
            return 0
        count = ctypes.c_int(0)
        if driver.cuDeviceGetCount(ctypes.byref(count)) != CUDA_SUCCESS:
            return 0
    except (AttributeError, OSError):
        return 0
    return max(0, count.value)
