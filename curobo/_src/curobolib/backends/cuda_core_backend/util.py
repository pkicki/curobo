# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small CUDA backend helpers shared by kernel launchers."""

import torch


def ceil_div(a: int, b: int) -> int:
    """Return the integer ceiling of ``a / b``."""
    return (a + b - 1) // b


def get_device_max_shared_memory(device: torch.device) -> int:
    """Return the opt-in per-block shared-memory limit in bytes.

    PyTorch does not expose the device field consistently across supported
    versions, so query the CUDA driver when neither common property exists.
    """
    properties = torch.cuda.get_device_properties(device)
    value = getattr(properties, "shared_memory_per_block_optin", 0) or getattr(
        properties, "shared_memory_per_block", 0
    )
    if value:
        return int(value)

    try:
        from cuda.bindings import driver as cuda_driver

        cuda_driver.cuInit(0)
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        status, cuda_device = cuda_driver.cuDeviceGet(index)
        if status.name != "CUDA_SUCCESS":
            return 0
        status, value = cuda_driver.cuDeviceGetAttribute(
            cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            cuda_device,
        )
        return int(value) if status.name == "CUDA_SUCCESS" else 0
    except (ImportError, AttributeError, RuntimeError):
        # A zero limit safely disables optional shared-memory-specialized paths.
        return 0
