import unittest
from math import isclose

from snapdragon_npu_audio import (
    AudioBuffer,
    BackendKind,
    EnhancementConfig,
    SnapdragonAudioEnhancer,
    select_backend,
)
from snapdragon_npu_audio.dsp import analyze, db_to_linear, loudness_normalize, true_peak_limit
from snapdragon_npu_audio.npu import NpuAssistModel, extract_features


def sine_like_buffer(amplitude: float = 0.25, frames: int = 480) -> AudioBuffer:
    samples = []
    pattern = (0.0, amplitude, 0.0, -amplitude)
    for index in range(frames):
        value = pattern[index % len(pattern)]
        samples.extend((value, value * 0.9))
    return AudioBuffer.from_interleaved(samples, sample_rate=48_000, channels=2)


class AudioPipelineTests(unittest.TestCase):
    def test_loudness_normalize_moves_toward_target_without_excessive_gain(self) -> None:
        source = sine_like_buffer(amplitude=0.05)

        normalized = loudness_normalize(source, target_lufs=-16.0, max_gain_db=6.0)

        self.assertGreater(analyze(normalized).integrated_lufs, analyze(source).integrated_lufs)
        self.assertLessEqual(normalized.peak(), source.peak() * db_to_linear(6.0) + 1.0e-9)


    def test_true_peak_limiter_holds_configured_ceiling(self) -> None:
        source = AudioBuffer.from_interleaved([1.4, -1.3, 0.2, -0.2], sample_rate=48_000, channels=2)

        limited = true_peak_limit(source)

        self.assertLessEqual(limited.peak(), db_to_linear(-1.0) + 1.0e-9)


    def test_backend_selection_prefers_qnn_then_directml_then_cpu(self) -> None:
        self.assertIs(select_backend(available_providers=("QNNExecutionProvider",)), BackendKind.QNN)
        self.assertIs(select_backend(available_providers=("DmlExecutionProvider",)), BackendKind.DIRECTML)
        self.assertIs(select_backend(available_providers=()), BackendKind.CPU)
        self.assertIs(select_backend(preferred="qnn"), BackendKind.QNN)
        self.assertIs(select_backend(env={"SNAPDRAGON_AUDIO_BACKEND": "directml"}), BackendKind.DIRECTML)


    def test_feature_extraction_and_heuristic_controls_are_bounded(self) -> None:
        source = sine_like_buffer(amplitude=0.1)
        features = extract_features(source)
        controls = NpuAssistModel(BackendKind.CPU).predict_controls(analyze(source))

        self.assertLess(-120.0, features.rms_dbfs)
        self.assertLess(features.rms_dbfs, 0.0)
        self.assertGreaterEqual(features.low_band_energy, 0.0)
        self.assertLessEqual(features.low_band_energy, 1.0)
        self.assertGreaterEqual(features.mid_band_energy, 0.0)
        self.assertLessEqual(features.mid_band_energy, 1.0)
        self.assertGreaterEqual(features.high_band_energy, 0.0)
        self.assertLessEqual(features.high_band_energy, 1.0)
        self.assertGreaterEqual(controls["clarity_db"], 0.0)
        self.assertLessEqual(controls["clarity_db"], 1.5)
        self.assertGreaterEqual(controls["stereo_width_delta"], 0.0)
        self.assertLessEqual(controls["stereo_width_delta"], 0.02)


    def test_end_to_end_enhancer_keeps_stereo_and_prevents_clipping(self) -> None:
        source = sine_like_buffer(amplitude=0.8)
        enhancer = SnapdragonAudioEnhancer(
            EnhancementConfig(preferred_backend=BackendKind.CPU, limiter_ceiling_db=-1.0)
        )

        processed, report = enhancer.process(source)

        self.assertEqual(processed.channels, 2)
        self.assertEqual(processed.frame_count, source.frame_count)
        self.assertLessEqual(processed.peak(), db_to_linear(-1.0) + 1.0e-9)
        self.assertIs(report.backend, BackendKind.CPU)
        self.assertTrue(report.predicted_controls)
        self.assertTrue(isclose(report.output_metrics.peak_dbfs, analyze(processed).peak_dbfs))


if __name__ == "__main__":
    unittest.main()
