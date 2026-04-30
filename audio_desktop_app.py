#!/usr/bin/env python3
"""Desktop GUI for the audio safety assistants."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk
except ModuleNotFoundError:
    tk = None
    ttk = None
    messagebox = None
    scrolledtext = None

if TYPE_CHECKING:
    import tkinter as tk_types

import audio_lossless_assistant as lossless
import npu_audio_enhancement_assistant as npu_enhancement
import snapdragon_streaming_studio as studio
import windows_ldac_assistant as ldac


APP_TITLE = "Audio Codec Assistant"
LOSSLESS_TARGETS = ("flac", "alac", "wav")
HIGH_RES_TARGETS = ("flac-24-96", "flac-24-192", "wav-24-96", "wav-32-192")
STREAMING_SERVICES = ("spotify", "apple-music", "youtube-music")


def build_lossless_assessment_text(codec: str) -> str:
    if not codec.strip():
        raise ValueError("source codec is required")
    return lossless.render_assessment(lossless.assess_codec(codec))


def build_lossless_plan_text(source_codec: str, target_codec: str) -> str:
    if not source_codec.strip():
        raise ValueError("source codec is required")
    return lossless.render_plan(
        lossless.build_preservation_plan(source_codec, target_codec)
    )


def build_ldac_status_text(system: str | None = None) -> str:
    return ldac.render_report(ldac.build_diagnostic(system or platform.system()))


def build_npu_enhancement_text(
    source_codec: str,
    target: str,
    status: npu_enhancement.NpuStatus | None = None,
) -> str:
    if not source_codec.strip():
        raise ValueError("source codec is required")
    plan = npu_enhancement.build_enhancement_plan(
        source_codec,
        target=target,
        npu_status=status,
    )
    return npu_enhancement.render_enhancement_plan(plan)


def build_npu_status_text(status: npu_enhancement.NpuStatus | None = None) -> str:
    return npu_enhancement.render_npu_status(
        status or npu_enhancement.detect_npu_status()
    )


def build_streaming_studio_plan_text(
    service: str,
    user_id: str,
    provider: str | None = None,
    sample_rate_hz: int = 96000,
    frame_size: int = 256,
) -> str:
    if not user_id.strip():
        raise ValueError("user id is required")
    plan = studio.build_studio_plan(
        service=service,
        user_id=user_id,
        sample_rate_hz=sample_rate_hz,
        frame_size=frame_size,
        provider=provider,
    )
    return studio.render_studio_plan(plan)


def build_streaming_recommendation_update_text(
    user_id: str,
    clarity: float,
    depth: float,
    vocal: float,
    bass: float,
) -> str:
    if not user_id.strip():
        raise ValueError("user id is required")
    state = studio.initialize_recommendation_state(user_id)
    updated = studio.update_recommendation_state(
        state,
        {
            "clarity": clarity,
            "depth": depth,
            "vocal_presence": vocal,
            "bass_control": bass,
        },
    )
    bias = studio.recommend_next_track_bias(updated)
    lines = [
        f"Realtime recommendation updated for user: {updated.user_id}",
        f"Updates: {updated.updates}",
        (
            "Embedding: "
            f"clarity={updated.embedding['clarity']}, "
            f"depth={updated.embedding['depth']}, "
            f"vocal_presence={updated.embedding['vocal_presence']}, "
            f"bass_control={updated.embedding['bass_control']}"
        ),
        (
            "Bias: "
            f"acoustic={bias['acoustic']}, "
            f"live_stage={bias['live_stage']}, "
            f"electronic={bias['electronic']}, "
            f"vocal_focus={bias['vocal_focus']}"
        ),
    ]
    return "\n".join(lines)


if ttk is not None:
    BaseFrame = ttk.Frame
else:
    BaseFrame = object


class AudioDesktopApp(BaseFrame):
    def __init__(self, master: "tk_types.Tk") -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is required to run the desktop app")
        super().__init__(master, padding=16)
        self.master = master
        self.source_codec = tk.StringVar(value="mp3")
        self.target_codec = tk.StringVar(value="flac")
        self.npu_source_codec = tk.StringVar(value="ldac")
        self.high_res_target = tk.StringVar(value="flac-24-96")
        self.streaming_service = tk.StringVar(value="spotify")
        self.streaming_user_id = tk.StringVar(value="listener-a")
        self.streaming_provider = tk.StringVar(value="QNNExecutionProvider")
        self.rec_clarity = tk.DoubleVar(value=0.9)
        self.rec_depth = tk.DoubleVar(value=0.82)
        self.rec_vocal = tk.DoubleVar(value=0.91)
        self.rec_bass = tk.DoubleVar(value=0.72)
        self.pack(fill=tk.BOTH, expand=True)
        self._build_widgets()

    def _build_widgets(self) -> None:
        self.master.title(APP_TITLE)
        self.master.minsize(760, 560)

        title = ttk.Label(
            self,
            text="Audio Codec Assistant",
            font=("TkDefaultFont", 18, "bold"),
        )
        title.pack(anchor=tk.W)

        note = ttk.Label(
            self,
            text=(
                "非可逆コーデックで失われた音は復元せず、"
                "安全なロスレス保全計画を作成します。"
            ),
        )
        note.pack(anchor=tk.W, pady=(4, 12))

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        lossless_tab = ttk.Frame(notebook, padding=12)
        npu_tab = ttk.Frame(notebook, padding=12)
        ldac_tab = ttk.Frame(notebook, padding=12)
        streaming_tab = ttk.Frame(notebook, padding=12)
        notebook.add(lossless_tab, text="ロスレス保全")
        notebook.add(npu_tab, text="AI/NPU補完")
        notebook.add(ldac_tab, text="Windows LDAC")
        notebook.add(streaming_tab, text="Snapdragon Streaming")

        self._build_lossless_tab(lossless_tab)
        self._build_npu_tab(npu_tab)
        self._build_ldac_tab(ldac_tab)
        self._build_streaming_tab(streaming_tab)

    def _build_lossless_tab(self, parent: ttk.Frame) -> None:
        form = ttk.Frame(parent)
        form.pack(fill=tk.X)

        ttk.Label(form, text="入力コーデック").grid(row=0, column=0, sticky=tk.W)
        source = ttk.Entry(form, textvariable=self.source_codec, width=24)
        source.grid(row=0, column=1, sticky=tk.W, padx=(8, 16))

        ttk.Label(form, text="保存先").grid(row=0, column=2, sticky=tk.W)
        target = ttk.Combobox(
            form,
            textvariable=self.target_codec,
            values=LOSSLESS_TARGETS,
            width=12,
            state="readonly",
        )
        target.grid(row=0, column=3, sticky=tk.W, padx=(8, 0))

        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, pady=10)
        ttk.Button(buttons, text="コーデック判定", command=self.show_assessment).pack(
            side=tk.LEFT
        )
        ttk.Button(buttons, text="保全計画を作成", command=self.show_plan).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        self.lossless_output = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=20)
        self.lossless_output.pack(fill=tk.BOTH, expand=True)
        self._write_lossless_output(
            "例: mp3 を判定すると、完全復元不可であることを表示します。\n"
            "例: flac から alac への計画は真のロスレス保全として表示します。"
        )

    def _build_npu_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "AI/NPU は劣化部分を推定補完できますが、"
                "真のハイレゾロスレス復元ではありません。"
            ),
        ).pack(anchor=tk.W)

        form = ttk.Frame(parent)
        form.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(form, text="Bluetooth/入力コーデック").grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Entry(form, textvariable=self.npu_source_codec, width=24).grid(
            row=0, column=1, sticky=tk.W, padx=(8, 16)
        )

        ttk.Label(form, text="ハイレゾ保存先").grid(row=0, column=2, sticky=tk.W)
        ttk.Combobox(
            form,
            textvariable=self.high_res_target,
            values=HIGH_RES_TARGETS,
            width=14,
            state="readonly",
        ).grid(row=0, column=3, sticky=tk.W, padx=(8, 0))

        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, pady=10)
        ttk.Button(
            buttons,
            text="NPU状態を表示",
            command=self.show_npu_status,
        ).pack(side=tk.LEFT)
        ttk.Button(
            buttons,
            text="AI補完計画を作成",
            command=self.show_npu_plan,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.npu_output = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=20)
        self.npu_output.pack(fill=tk.BOTH, expand=True)
        self._write_npu_output(build_npu_enhancement_text("ldac", "flac-24-96"))

    def _build_ldac_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Windows 標準 Bluetooth スタックに LDAC エンコーダーは含まれません。"
            ),
        ).pack(anchor=tk.W)
        ttk.Button(
            parent,
            text="Windows LDAC 診断を表示",
            command=self.show_windows_ldac_status,
        ).pack(anchor=tk.W, pady=10)

        self.ldac_output = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=20)
        self.ldac_output.pack(fill=tk.BOTH, expand=True)
        self._write_ldac_output(build_ldac_status_text("Windows"))

    def _build_streaming_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Snapdragon X向けリアルタイム計画: 音場/定位/分離重視 + "
                "XMOS低遅延 + レコメンド更新"
            ),
        ).pack(anchor=tk.W)

        form = ttk.Frame(parent)
        form.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(form, text="サービス").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(
            form,
            textvariable=self.streaming_service,
            values=STREAMING_SERVICES,
            width=16,
            state="readonly",
        ).grid(row=0, column=1, sticky=tk.W, padx=(8, 16))
        ttk.Label(form, text="ユーザーID").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(form, textvariable=self.streaming_user_id, width=18).grid(
            row=0, column=3, sticky=tk.W, padx=(8, 16)
        )
        ttk.Label(form, text="NPU Provider").grid(row=0, column=4, sticky=tk.W)
        ttk.Entry(form, textvariable=self.streaming_provider, width=22).grid(
            row=0, column=5, sticky=tk.W, padx=(8, 0)
        )

        feedback = ttk.Frame(parent)
        feedback.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(feedback, text="Clarity").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(feedback, from_=0.0, to=1.0, variable=self.rec_clarity).grid(
            row=0, column=1, sticky=tk.EW, padx=(8, 16)
        )
        ttk.Label(feedback, text="Depth").grid(row=0, column=2, sticky=tk.W)
        ttk.Scale(feedback, from_=0.0, to=1.0, variable=self.rec_depth).grid(
            row=0, column=3, sticky=tk.EW, padx=(8, 16)
        )
        ttk.Label(feedback, text="Vocal").grid(row=1, column=0, sticky=tk.W)
        ttk.Scale(feedback, from_=0.0, to=1.0, variable=self.rec_vocal).grid(
            row=1, column=1, sticky=tk.EW, padx=(8, 16)
        )
        ttk.Label(feedback, text="Bass").grid(row=1, column=2, sticky=tk.W)
        ttk.Scale(feedback, from_=0.0, to=1.0, variable=self.rec_bass).grid(
            row=1, column=3, sticky=tk.EW, padx=(8, 16)
        )
        feedback.columnconfigure(1, weight=1)
        feedback.columnconfigure(3, weight=1)

        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, pady=10)
        ttk.Button(
            buttons,
            text="スタジオ計画を生成",
            command=self.show_streaming_plan,
        ).pack(side=tk.LEFT)
        ttk.Button(
            buttons,
            text="レコメンドを更新",
            command=self.show_streaming_recommendation_update,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.streaming_output = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=16)
        self.streaming_output.pack(fill=tk.BOTH, expand=True)
        self._write_streaming_output(
            build_streaming_studio_plan_text(
                service=self.streaming_service.get(),
                user_id=self.streaming_user_id.get(),
                provider=self.streaming_provider.get(),
            )
        )

    def _write_lossless_output(self, text: str) -> None:
        self.lossless_output.delete("1.0", tk.END)
        self.lossless_output.insert(tk.END, text)

    def _write_ldac_output(self, text: str) -> None:
        self.ldac_output.delete("1.0", tk.END)
        self.ldac_output.insert(tk.END, text)

    def _write_npu_output(self, text: str) -> None:
        self.npu_output.delete("1.0", tk.END)
        self.npu_output.insert(tk.END, text)

    def _write_streaming_output(self, text: str) -> None:
        self.streaming_output.delete("1.0", tk.END)
        self.streaming_output.insert(tk.END, text)

    def show_assessment(self) -> None:
        try:
            text = build_lossless_assessment_text(self.source_codec.get())
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._write_lossless_output(text)

    def show_plan(self) -> None:
        try:
            text = build_lossless_plan_text(
                self.source_codec.get(), self.target_codec.get()
            )
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._write_lossless_output(text)

    def show_windows_ldac_status(self) -> None:
        self._write_ldac_output(build_ldac_status_text("Windows"))

    def show_npu_status(self) -> None:
        self._write_npu_output(build_npu_status_text())

    def show_npu_plan(self) -> None:
        try:
            text = build_npu_enhancement_text(
                self.npu_source_codec.get(),
                self.high_res_target.get(),
            )
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._write_npu_output(text)

    def show_streaming_plan(self) -> None:
        try:
            text = build_streaming_studio_plan_text(
                service=self.streaming_service.get(),
                user_id=self.streaming_user_id.get(),
                provider=self.streaming_provider.get() or None,
            )
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._write_streaming_output(text)

    def show_streaming_recommendation_update(self) -> None:
        try:
            text = build_streaming_recommendation_update_text(
                user_id=self.streaming_user_id.get(),
                clarity=self.rec_clarity.get(),
                depth=self.rec_depth.get(),
                vocal=self.rec_vocal.get(),
                bass=self.rec_bass.get(),
            )
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._write_streaming_output(text)


def create_app() -> "tk_types.Tk":
    if tk is None:
        raise RuntimeError("Tkinter is required to run the desktop app")
    root = tk.Tk()
    AudioDesktopApp(root)
    return root


def main() -> int:
    create_app().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
