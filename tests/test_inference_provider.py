from audio_enhancer.inference import InferenceBackend, ordered_provider_names, select_provider


def test_selects_qnn_when_snapdragon_npu_is_available() -> None:
    provider = select_provider(
        ["CPUExecutionProvider", "DmlExecutionProvider", "QNNExecutionProvider"]
    )

    assert provider.backend is InferenceBackend.QNN_NPU
    assert provider.onnx_provider_name == "QNNExecutionProvider"
    assert ordered_provider_names(provider)[0] == "QNNExecutionProvider"


def test_directml_is_fallback_when_qnn_is_missing() -> None:
    provider = select_provider(["CPUExecutionProvider", "DmlExecutionProvider"])

    assert provider.backend is InferenceBackend.DIRECTML
    assert provider.onnx_provider_name == "DmlExecutionProvider"


def test_cpu_is_final_safe_fallback() -> None:
    provider = select_provider(["CPUExecutionProvider"])

    assert provider.backend is InferenceBackend.CPU


def test_can_disable_npu_for_battery_or_diagnostics() -> None:
    provider = select_provider(
        ["CPUExecutionProvider", "DmlExecutionProvider", "QNNExecutionProvider"],
        environment={"AUDIO_ENHANCER_DISABLE_NPU": "1"},
    )

    assert provider.backend is InferenceBackend.DIRECTML
