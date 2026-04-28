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
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .profiles import available_profiles
from .recommender import build_recommendation_status, generate_recommendations
from .realtime import LowLatencyBufferPlan, ServiceState, build_realtime_status


class EnhancerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NPU Streaming Music Enhancer")
        self.setMinimumSize(1120, 760)

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

        self.recommendation_tick = 0
        self.npu_meter = QProgressBar()
        self.npu_meter.setRange(0, 100)
        self.npu_meter.setValue(92)
        self.npu_meter.setFormat("NPU offload target: %p%")
        self.state_label = QLabel("STATE: STANDBY / 待機中")
        self.state_label.setObjectName("stateLabel")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label = QLabel("リアルタイム配信音声の NPU 処理は待機中です。")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("statusLabel")
        self.result_label = QLabel(build_realtime_status(self._service_state(), active=False))
        self.result_label.setWordWrap(True)
        self.result_label.setObjectName("resultLabel")
        self.recommendation_label = QLabel(build_recommendation_status(generate_recommendations()))
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setObjectName("resultLabel")

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(34, 30, 34, 30)
        outer.setSpacing(20)

        title = QLabel("NPU Streaming Music Enhancer")
        title.setObjectName("titleLabel")
        title.setMinimumWidth(620)
        subtitle = QLabel(
            "Spotify / Apple Music / YouTube Music の再生音をリアルタイムで受け、"
            "Snapdragon X NPU / QNN / ASIO XMOS 低レイテンシ経路で定位・楽器分離・ボーカル立体感を狙うコントロール"
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitleLabel")
        hero = QFrame()
        hero.setObjectName("heroCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(8)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        outer.addWidget(hero)

        control_card = QFrame()
        control_card.setObjectName("card")
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(22, 20, 22, 20)
        control_layout.setSpacing(16)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(14)
        form.setHorizontalSpacing(18)
        services = QHBoxLayout()
        services.addWidget(self.spotify_check)
        services.addWidget(self.apple_music_check)
        services.addWidget(self.youtube_music_check)
        services.addStretch(1)
        form.addRow("対象サービス", services)
        form.addRow("プロファイル", self.profile_combo)
        form.addRow("低レイテンシ出力", self.latency_combo)
        form.addRow("NPU オフロード目標", self.npu_meter)
        control_layout.addLayout(form)

        actions = QVBoxLayout()
        actions.setSpacing(12)
        primary_row = QHBoxLayout()
        primary_row.setSpacing(12)
        secondary_row = QHBoxLayout()
        secondary_row.setSpacing(12)
        stop_button = QPushButton("停止")
        stop_button.setObjectName("secondaryButton")
        stop_button.setMinimumWidth(120)
        stop_button.clicked.connect(self._stop_realtime)
        self.start_button = QPushButton("リアルタイム NPU 処理を開始")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setMinimumWidth(360)
        self.start_button.clicked.connect(self._start_realtime)
        self.recommend_button = QPushButton("Deep Learning AI レコメンドをリアルタイム反映")
        self.recommend_button.setObjectName("accentButton")
        self.recommend_button.setMinimumWidth(480)
        self.recommend_button.clicked.connect(self._apply_recommendations)
        primary_row.addWidget(self.start_button)
        primary_row.addWidget(stop_button)
        primary_row.addStretch(1)
        secondary_row.addWidget(self.recommend_button)
        secondary_row.addStretch(1)
        actions.addLayout(primary_row)
        actions.addLayout(secondary_row)
        control_layout.addLayout(actions)
        outer.addWidget(control_card)

        telemetry_card = QFrame()
        telemetry_card.setObjectName("card")
        telemetry_layout = QVBoxLayout(telemetry_card)
        telemetry_layout.setContentsMargins(22, 20, 22, 20)
        telemetry_layout.setSpacing(12)
        telemetry_layout.addWidget(self.state_label)
        telemetry_layout.addWidget(self.status_label)
        telemetry_layout.addWidget(self._scroll_panel(self.result_label, 168))
        telemetry_layout.addWidget(self._scroll_panel(self.recommendation_label, 220))
        outer.addWidget(telemetry_card)
        outer.addStretch(1)

        root.setStyleSheet(
            """
            QWidget {
                background: #0f172a;
                color: #e5e7eb;
                font-size: 15px;
            }
            #heroCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e3a8a, stop:0.55 #312e81, stop:1 #0f172a);
                border: 1px solid rgba(147, 197, 253, 0.35);
                border-radius: 22px;
            }
            #card {
                background: rgba(15, 23, 42, 0.92);
                border: 1px solid #334155;
                border-radius: 18px;
            }
            #titleLabel {
                background: transparent;
                color: #f8fafc;
                font-size: 34px;
                font-weight: 800;
                letter-spacing: 0.4px;
            }
            #subtitleLabel {
                background: transparent;
                color: #bfdbfe;
                font-size: 15px;
            }
            QLabel {
                background: transparent;
            }
            QCheckBox {
                background: transparent;
                color: #e5e7eb;
                spacing: 8px;
            }
            QComboBox {
                background: #111827;
                border: 1px solid #475569;
                border-radius: 10px;
                color: #f8fafc;
                min-height: 38px;
                padding: 4px 10px;
            }
            QProgressBar {
                background: #111827;
                border: 1px solid #475569;
                border-radius: 10px;
                color: #f8fafc;
                min-height: 22px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #22d3ee;
                border-radius: 9px;
            }
            QPushButton {
                background: #1f2937;
                border: 1px solid #475569;
                border-radius: 12px;
                color: #f8fafc;
                min-height: 40px;
                min-width: 120px;
                padding: 7px 12px;
            }
            #primaryButton {
                background: #2563eb;
                border: 1px solid #60a5fa;
                color: white;
                font-weight: 700;
            }
            #accentButton {
                background: #7c3aed;
                border: 1px solid #a78bfa;
                color: white;
                font-weight: 700;
                min-width: 360px;
            }
            #secondaryButton {
                background: #111827;
                color: #cbd5e1;
            }
            #stateLabel {
                background: #172554;
                border: 1px solid #38bdf8;
                border-radius: 14px;
                color: #bfdbfe;
                font-size: 18px;
                font-weight: 800;
                padding: 12px;
            }
            #statusLabel {
                color: #cbd5e1;
            }
            #resultLabel {
                background: #020617;
                border: 1px solid #334155;
                border-radius: 14px;
                color: #dbeafe;
                font-family: monospace;
                line-height: 140%;
                padding: 14px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #0f172a;
                border-radius: 6px;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #475569;
                border-radius: 5px;
            }
            """
        )
        self.setCentralWidget(root)

    def _scroll_panel(self, content: QLabel, height: int) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFixedHeight(height)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(content)
        return area

    def _service_state(self) -> ServiceState:
        return ServiceState(
            spotify=self.spotify_check.isChecked(),
            apple_music=self.apple_music_check.isChecked(),
            youtube_music=self.youtube_music_check.isChecked(),
            profile=self.profile_combo.currentText(),
            latency_path=self.latency_combo.currentText(),
            buffer_plan=LowLatencyBufferPlan(buffer_samples=32, sample_rate=48_000),
        )

    def _start_realtime(self) -> None:
        if not self._service_state().selected_services():
            self.state_label.setText("STATE: NEED SERVICE / 対象サービス未選択")
            self.status_label.setText("対象サービスを少なくとも 1 つ選択してください。")
            self.result_label.setText(build_realtime_status(self._service_state(), active=False))
            return

        self.state_label.setText("STATE: ACTIVE / リアルタイム処理中")
        self.start_button.setText("リアルタイム NPU 処理中")
        self.npu_meter.setValue(96)
        self.status_label.setText(
            "リアルタイム NPU 処理を開始しました。実機では Windows ARM64 + "
            "Snapdragon X NPU + ONNX Runtime QNN + SABAJ A20D(ES) / XMOS USB DAC 極小バッファへ接続します。"
        )
        self.result_label.setText(build_realtime_status(self._service_state(), active=True))

    def _stop_realtime(self) -> None:
        self.state_label.setText("STATE: STANDBY / 待機中")
        self.start_button.setText("リアルタイム NPU 処理を開始")
        self.npu_meter.setValue(92)
        self.status_label.setText("リアルタイム NPU 処理を停止しました。")
        self.result_label.setText(build_realtime_status(self._service_state(), active=False))

    def _apply_recommendations(self) -> None:
        self.recommendation_tick += 1
        recommendations = generate_recommendations(update_id=self.recommendation_tick)
        self.recommendation_label.setText(build_recommendation_status(recommendations))
        self.npu_meter.setValue(98)
        self.status_label.setText(
            f"Deep Learning AI レコメンドを更新しました (realtime tick #{self.recommendation_tick})。実機では Spotify / Apple Music / "
            "YouTube Music のプレイリスト候補と次候補キューへリアルタイム反映します。"
        )


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
