from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .audio import enhance_wav, generate_demo_wav
from .profiles import available_profiles
from .reports import build_status_text


class EnhancerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NPU Audio Enhancer")
        self.setMinimumSize(760, 460)

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(sorted(available_profiles()))
        self.profile_combo.setCurrentText("snapdragon-x-npu")
        self.status_label = QLabel("WAV ファイルを選択して音質向上を開始してください。")
        self.status_label.setWordWrap(True)
        self.result_label = QLabel("結果はここに表示されます。")
        self.result_label.setWordWrap(True)
        self.result_label.setObjectName("resultLabel")

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(18)

        title = QLabel("NPU Audio Enhancer")
        title.setObjectName("titleLabel")
        subtitle = QLabel(
            "Spotify / Apple Music / YouTube Music の再生音を想定した、"
            "Snapdragon X NPU 向け音声後処理プロトタイプ"
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitleLabel")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("入力 WAV", self._path_picker(self.input_edit, self._choose_input))
        form.addRow("出力 WAV", self._path_picker(self.output_edit, self._choose_output))
        form.addRow("プロファイル", self.profile_combo)
        outer.addLayout(form)

        actions = QHBoxLayout()
        demo_button = QPushButton("デモ WAV を生成")
        demo_button.clicked.connect(self._generate_demo)
        enhance_button = QPushButton("音質向上を実行")
        enhance_button.setObjectName("primaryButton")
        enhance_button.clicked.connect(self._enhance)
        actions.addWidget(demo_button)
        actions.addStretch(1)
        actions.addWidget(enhance_button)
        outer.addLayout(actions)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        outer.addWidget(separator)
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
            #resultLabel {
                background: #f3f4f6;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 12px;
            }
            """
        )
        self.setCentralWidget(root)

    def _path_picker(self, edit: QLineEdit, callback) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("参照")
        browse.clicked.connect(callback)
        layout.addWidget(edit)
        layout.addWidget(browse)
        return container

    def _choose_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "入力 WAV を選択", "", "WAV files (*.wav)")
        if path:
            self.input_edit.setText(path)
            if not self.output_edit.text():
                source = Path(path)
                self.output_edit.setText(str(source.with_name(f"{source.stem}_enhanced.wav")))

    def _choose_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "出力 WAV を保存", "", "WAV files (*.wav)")
        if path:
            self.output_edit.setText(path)

    def _generate_demo(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "デモ WAV を保存", "demo_input.wav", "WAV files (*.wav)")
        if not path:
            return
        try:
            generate_demo_wav(path)
        except Exception as exc:  # pragma: no cover - user-facing error path
            self._show_error("デモ生成に失敗しました", exc)
            return

        self.input_edit.setText(path)
        source = Path(path)
        self.output_edit.setText(str(source.with_name(f"{source.stem}_enhanced.wav")))
        self.status_label.setText(f"デモ WAV を生成しました: {path}")
        self.result_label.setText("生成した WAV をそのまま入力として使えます。")

    def _enhance(self) -> None:
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        if not input_path or not output_path:
            QMessageBox.warning(self, "入力不足", "入力 WAV と出力 WAV を指定してください。")
            return

        try:
            report = enhance_wav(input_path, output_path, self.profile_combo.currentText())
        except Exception as exc:  # pragma: no cover - user-facing error path
            self._show_error("音質向上に失敗しました", exc)
            return

        self.status_label.setText(f"処理が完了しました: {output_path}")
        self.result_label.setText(build_status_text(report))

    def _show_error(self, title: str, exc: Exception) -> None:
        QMessageBox.critical(self, title, str(exc))
        self.status_label.setText(f"{title}: {exc}")


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
