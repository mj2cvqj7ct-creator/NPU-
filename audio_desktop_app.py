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
import windows_ldac_assistant as ldac


APP_TITLE = "Audio Codec Assistant"
LOSSLESS_TARGETS = ("flac", "alac", "wav")


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
        ldac_tab = ttk.Frame(notebook, padding=12)
        notebook.add(lossless_tab, text="ロスレス保全")
        notebook.add(ldac_tab, text="Windows LDAC")

        self._build_lossless_tab(lossless_tab)
        self._build_ldac_tab(ldac_tab)

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

    def _write_lossless_output(self, text: str) -> None:
        self.lossless_output.delete("1.0", tk.END)
        self.lossless_output.insert(tk.END, text)

    def _write_ldac_output(self, text: str) -> None:
        self.ldac_output.delete("1.0", tk.END)
        self.ldac_output.insert(tk.END, text)

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
