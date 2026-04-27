"""Provider selection for Snapdragon X NPU inference.

This module intentionally avoids importing ONNX Runtime at import time. The
runtime package and Qualcomm QNN libraries are platform-specific, so the pure
Python selector is testable on development machines while keeping production
provider ordering explicit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class InferenceBackend(str, Enum):
    """Supported inference backends in preference order."""

    QNN_NPU = "qnn-npu"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class ProviderChoice:
    """Resolved inference provider and the reason it was selected."""

    backend: InferenceBackend
    onnx_provider_name: str
    reason: str


PROVIDER_ORDER: tuple[tuple[InferenceBackend, str], ...] = (
    (InferenceBackend.QNN_NPU, "QNNExecutionProvider"),
    (InferenceBackend.DIRECTML, "DmlExecutionProvider"),
    (InferenceBackend.CPU, "CPUExecutionProvider"),
)


def select_provider(
    available_provider_names: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> ProviderChoice:
    """Select the best available ONNX Runtime provider for audio inference.

    `AUDIO_ENHANCER_DISABLE_NPU=1` can be set to force a non-NPU fallback for
    diagnostics or battery comparisons.
    """

    env = environment or os.environ
    available = set(available_provider_names)
    npu_disabled = env.get("AUDIO_ENHANCER_DISABLE_NPU") == "1"

    for backend, provider_name in PROVIDER_ORDER:
        if backend is InferenceBackend.QNN_NPU and npu_disabled:
            continue
        if provider_name in available:
            return ProviderChoice(
                backend=backend,
                onnx_provider_name=provider_name,
                reason=_selection_reason(backend, npu_disabled),
            )

    return ProviderChoice(
        backend=InferenceBackend.CPU,
        onnx_provider_name="CPUExecutionProvider",
        reason="No advertised provider matched; use CPU as the safe baseline.",
    )


def ordered_provider_names(choice: ProviderChoice) -> list[str]:
    """Return ONNX Runtime provider list with the selected provider first."""

    names = [choice.onnx_provider_name]
    for _, provider_name in PROVIDER_ORDER:
        if provider_name not in names:
            names.append(provider_name)
    return names


def _selection_reason(backend: InferenceBackend, npu_disabled: bool) -> str:
    if backend is InferenceBackend.QNN_NPU:
        return "QNN provider is available; route supported models to Snapdragon X NPU."
    if backend is InferenceBackend.DIRECTML and npu_disabled:
        return "NPU disabled by environment; use DirectML fallback."
    if backend is InferenceBackend.DIRECTML:
        return "QNN provider unavailable; use DirectML fallback."
    if npu_disabled:
        return "NPU disabled by environment and DirectML unavailable; use CPU fallback."
    return "Only CPU provider is available."
