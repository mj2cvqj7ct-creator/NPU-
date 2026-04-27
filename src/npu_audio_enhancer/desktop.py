from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .profiles import available_profiles
from .realtime import ServiceState, build_realtime_status


class EnhancerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NPU Streaming Music Enhancer")
        self.setMinimumSize(820, 520)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(sorted(available_profiles()))
        self.profile_combo.setCurrentText("holographic-vocal-stage")
        self.latency_combo = QComboBox()
        self.latency_combo.addItems(
            [
                "ASIO XMOS USB DAC - extreme low latency",
                "WASAPI exclusive - low latency",
                "Virtual output - compatibility",
            ]
        )
        self.spotify_check = QCheckBox("Spotify")
        self.apple_music_check = QCheckBox("Apple Music")
        self.youtube_music_check = QCheckBox("YouTube Music")
        for checkbox in (self.spotify_check, self.apple_music_check, self.youtube_music_check):
            checkbox.setChecked(True)

        self.npu_meter = QProgressBar()
        self.npu_meter.setRange(0, 100)
        self.npu_meter.setValue(100)
        self.npu_meter.setFormat("NPU target load: %p%")
        self.state_label = QLabel("STATE: STANDBY / 実機接続待ち")
        self.state_label.setObjectName("stateLabel")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label = QLabel("リアルタイム配信音声の NPU 処理は待機中です。")
        self.status_label.setWordWrap(True)
        self.result_label = QLabel(build_realtime_status(self._service_state(), active=False))
        self.result_label.setWordWrap(True)
        self.result_label.setObjectName("resultLabel")

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(18)

        title = QLabel("NPU Streaming Music Enhancer")
        title.setObjectName("titleLabel")
        subtitle = QLabel(
            "Spotify / Apple Music / YouTube Music の再生音をリアルタイムで受け、"
            "Snapdragon X NPU をフル活用して定位・楽器分離・ボーカル立体感を狙うコントロール"
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitleLabel")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        services = QHBoxLayout()
        services.addWidget(self.spotify_check)
        services.addWidget(self.apple_music_check)
        services.addWidget(self.youtube_music_check)
        services.addStretch(1)
        form.addRow("対象サービス", services)
        form.addRow("プロファイル", self.profile_combo)
        form.addRow("低レイテンシ出力", self.latency_combo)
        form.addRow("NPU 使用率目標", self.npu_meter)
        outer.addLayout(form)

        actions = QHBoxLayout()
        stop_button = QPushButton("停止")
        stop_button.clicked.connect(self._stop_realtime)
        start_button = QPushButton("リアルタイム NPU 処理を開始")
        start_button.setObjectName("primaryButton")
        start_button.clicked.connect(self._start_realtime)
        actions.addWidget(stop_button)
        actions.addStretch(1)
        actions.addWidget(start_button)
        outer.addLayout(actions)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        outer.addWidget(separator)
        outer.addWidget(self.state_label)
        outer.addWidget(self.status_label)
        outer.addWidget(self.result_label)
        outer.addStretch(1)

        root.setStyleSheet(
            """
            QWidget { font-size: 15px; }
            #titleLabel { font-size: 28px; font-weight: 700; }
            #subtitleLabel { color: #4b5563; }
            QLineEdit, QComboBox {
                min-height: 34px;
                padding: 4px 8px;
            }
            QPushButton {
                min-height: 36px;
                padding: 6px 14px;
            }
            #primaryButton {
                background: #2563eb;
                color: white;
                border-radius: 6px;
            }
            #stateLabel {
                background: #dbeafe;
                border: 1px solid #60a5fa;
                border-radius: 8px;
                color: #1e3a8a;
                font-size: 18px;
                font-weight: 700;
                padding: 10px;
            }
            #resultLabel {
                background: #f3f4f6;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                font-family: monospace;
                line-height: 140%;
                padding: 12px;
            }
            """
        )
        self.setCentralWidget(root)

    def _service_state(self) -> ServiceState:
        return ServiceState(
            spotify=self.spotify_check.isChecked(),
            apple_music=self.apple_music_check.isChecked(),
            youtube_music=self.youtube_music_check.isChecked(),
            profile=self.profile_combo.currentText(),
            latency_path=self.latency_combo.currentText(),
        )

    def _start_realtime(self) -> None:
        if not self._service_state().selected_services():
            self.state_label.setText("STATE: NEED SERVICE / 対象サービス未選択")
            self.status_label.setText("対象サービスを少なくとも 1 つ選択してください。")
            self.result_label.setText(build_realtime_status(self._service_state(), active=False))
            return

        self.state_label.setText("STATE: ACTIVE / リアルタイム処理中")
        self.status_label.setText(
            "リアルタイム NPU 処理を開始しました。実機では Windows ARM64 + "
            "Snapdragon X NPU + ONNX Runtime QNN + ASIO XMOS USB DAC 極小バッファへ接続します。"
        )
        self.result_label.setText(build_realtime_status(self._service_state(), active=True))

    def _stop_realtime(self) -> None:
        self.state_label.setText("STATE: STANDBY / 実機接続待ち")
        self.status_label.setText("リアルタイム NPU 処理を停止しました。")
        self.result_label.setText(build_realtime_status(self._service_state(), active=False))


def create_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def main() -> int:
    app = create_app()
    window = EnhancerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
