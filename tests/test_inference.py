import numpy as np

from sxnpu_audio_enhancer.inference import (
    BackendKind,
    InferenceConfig,
    NullInferenceBackend,
    create_inference_backend,
)


def test_null_backend_is_passthrough() -> None:
    backend = NullInferenceBackend()
    frame = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)

    assert backend.kind is BackendKind.CPU
    np.testing.assert_array_equal(backend.enhance(frame), frame)


def test_auto_backend_falls_back_without_model() -> None:
    backend = create_inference_backend(InferenceConfig(preferred_backend=BackendKind.AUTO))

    assert isinstance(backend, NullInferenceBackend)

