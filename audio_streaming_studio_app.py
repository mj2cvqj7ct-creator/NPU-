#!/usr/bin/env python3
"""Modern desktop control panel for Snapdragon streaming audio plans."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk
except ModuleNotFoundError:
    tk = None
    ttk = None
    scrolledtext = None
    messagebox = None

if TYPE_CHECKING:
    import tkinter as tk_types

import snapdragon_streaming_studio as studio


APP_TITLE = "Snapdragon Streaming Studio"
THEME_BG = "#0d1117"
CARD_BG = "#161b22"
TEXT_FG = "#e6edf3"
ACCENT = "#58a6ff"


if ttk is not None:
    BaseFrame = ttk.Frame
else:
    BaseFrame = object


def build_plan_text(
    service: str,
    profile: str,
    target_latency_ms: int,
    sample_rate_hz: int,
    provider: str | None = None,
) -> str:
    plan = studio.build_realtime_audio_plan(
        service=service,
        profile_name=profile,
        target_latency_ms=target_latency_ms,
        preferred_provider=provider,
        sample_rate_hz=sample_rate_hz,
    )
    return studio.render_audio_plan(plan)


def build_exe_text() -> str:
    commands = studio.build_windows_exe_commands(
        script_path="audio_streaming_studio_app.py",
        app_name="SnapdragonStreamingStudio",
    )
    return "\n".join(commands)


class StreamingStudioApp(BaseFrame):
    def __init__(self, master: "tk_types.Tk") -> None:
        if tk is None or ttk is None or scrolledtext is None:
            raise RuntimeError("Tkinter is required to run the desktop app")
        super().__init__(master, padding=16)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)
        self.service = tk.StringVar(value="spotify")
        self.profile = tk.StringVar(value="immersive-reference")
        self.latency = tk.IntVar(value=28)
        self.sample_rate = tk.IntVar(value=48000)
        self.provider = tk.StringVar(value="")
        self.feedback = tk.DoubleVar(value=0.2)
        self.tags = tk.StringVar(value="vocal,detail,spatial")
        self.reco_state = studio.RealtimeRecommendationState()
        self._configure_style()
        self._build_widgets()

    def _configure_style(self) -> None:
        self.master.title(APP_TITLE)
        self.master.minsize(1000, 680)
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.master.configure(background=THEME_BG)
        style.configure(".", background=THEME_BG, foreground=TEXT_FG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("Header.TLabel", background=THEME_BG, foreground=ACCENT, font=("Segoe UI", 18, "bold"))
        style.configure("Info.TLabel", background=THEME_BG, foreground=TEXT_FG, font=("Segoe UI", 10))
        style.configure("TLabel", background=CARD_BG, foreground=TEXT_FG, font=("Segoe UI", 10))
        style.configure("TButton", padding=6)
        style.configure("TEntry", fieldbackground="#0f172a")
        style.configure("TCombobox", fieldbackground="#0f172a")
        style.configure("TNotebook", background=THEME_BG)
        style.configure("TNotebook.Tab", padding=(10, 6))

    def _build_widgets(self) -> None:
        ttk.Label(self, text=APP_TITLE, style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            self,
            text=(
                "Spotify / Apple Music / YouTube Music を前提に、Snapdragon X NPU と "
                "SABAJ A20D(ES) の低遅延安定動作プランを作成します。"
            ),
            style="Info.TLabel",
        ).pack(anchor=tk.W, pady=(4, 12))

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        realtime_tab = ttk.Frame(notebook, style="Card.TFrame", padding=12)
        recommend_tab = ttk.Frame(notebook, style="Card.TFrame", padding=12)
        build_tab = ttk.Frame(notebook, style="Card.TFrame", padding=12)

        notebook.add(realtime_tab, text="リアルタイム最適化")
        notebook.add(recommend_tab, text="リアルタイム学習")
        notebook.add(build_tab, text="Windows EXE")

        self._build_realtime_tab(realtime_tab)
        self._build_recommend_tab(recommend_tab)
        self._build_build_tab(build_tab)

    def _build_realtime_tab(self, parent: ttk.Frame) -> None:
        form = ttk.Frame(parent, style="Card.TFrame")
        form.pack(fill=tk.X)
        ttk.Label(form, text="サービス").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(
            form,
            textvariable=self.service,
            values=["spotify", "apple_music", "youtube_music"],
            state="readonly",
            width=20,
        ).grid(row=0, column=1, padx=(8, 16), sticky=tk.W)

        ttk.Label(form, text="プロファイル").grid(row=0, column=2, sticky=tk.W)
        ttk.Combobox(
            form,
            textvariable=self.profile,
            values=["immersive-reference", "vocal-forward", "wide-stage"],
            state="readonly",
            width=20,
        ).grid(row=0, column=3, padx=(8, 0), sticky=tk.W)

        ttk.Label(form, text="目標遅延(ms)").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        ttk.Entry(form, textvariable=self.latency, width=10).grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        ttk.Label(form, text="サンプルレート(Hz)").grid(row=1, column=2, sticky=tk.W, pady=(10, 0))
        ttk.Combobox(
            form,
            textvariable=self.sample_rate,
            values=[44100, 48000, 96000],
            width=10,
            state="readonly",
        ).grid(row=1, column=3, sticky=tk.W, padx=(8, 0), pady=(10, 0))

        ttk.Label(form, text="優先NPUプロバイダー(任意)").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        ttk.Entry(form, textvariable=self.provider, width=34).grid(row=2, column=1, columnspan=3, sticky=tk.W, pady=(10, 0))

        button_row = ttk.Frame(parent, style="Card.TFrame")
        button_row.pack(fill=tk.X, pady=10)
        ttk.Button(button_row, text="最適化プラン生成", command=self.generate_plan).pack(side=tk.LEFT)
        ttk.Button(button_row, text="サービスを開く", command=self.open_service).pack(side=tk.LEFT, padx=(8, 0))

        self.plan_output = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=24, background="#0b1220", foreground=TEXT_FG, insertbackground=TEXT_FG)
        self.plan_output.pack(fill=tk.BOTH, expand=True)
        self._write_plan("ここにリアルタイム音質最適化プランが表示されます。")

    def _build_recommend_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="タグ単位で再生フィードバックを取り込み、重みをオンライン更新します。",
        ).pack(anchor=tk.W)

        form = ttk.Frame(parent, style="Card.TFrame")
        form.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(form, text="タグ(カンマ区切り)").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(form, textvariable=self.tags, width=60).grid(row=0, column=1, sticky=tk.W, padx=(8, 0))
        ttk.Label(form, text="報酬(-1.0〜1.0)").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        ttk.Scale(form, from_=-1.0, to=1.0, variable=self.feedback, orient=tk.HORIZONTAL, length=340).grid(row=1, column=1, sticky=tk.W, padx=(8, 0), pady=(10, 0))

        button_row = ttk.Frame(parent, style="Card.TFrame")
        button_row.pack(fill=tk.X, pady=10)
        ttk.Button(button_row, text="学習を反映", command=self.learn_feedback).pack(side=tk.LEFT)
        ttk.Button(button_row, text="スコア計算", command=self.compute_score).pack(side=tk.LEFT, padx=(8, 0))

        self.reco_output = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=20, background="#0b1220", foreground=TEXT_FG, insertbackground=TEXT_FG)
        self.reco_output.pack(fill=tk.BOTH, expand=True)
        self._write_reco("学習前: まだ重みがありません。")

    def _build_build_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Windows で PyInstaller を使って EXE を生成し、デスクトップへコピーします。"
            ),
        ).pack(anchor=tk.W)
        button_row = ttk.Frame(parent, style="Card.TFrame")
        button_row.pack(fill=tk.X, pady=10)
        ttk.Button(button_row, text="EXEビルド手順を表示", command=self.show_exe_commands).pack(side=tk.LEFT)
        ttk.Button(button_row, text="JSONで保存", command=self.save_plan_json).pack(side=tk.LEFT, padx=(8, 0))

        self.exe_output = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=22, background="#0b1220", foreground=TEXT_FG, insertbackground=TEXT_FG)
        self.exe_output.pack(fill=tk.BOTH, expand=True)
        self._write_exe(build_exe_text())

    def _write_plan(self, text: str) -> None:
        self.plan_output.delete("1.0", tk.END)
        self.plan_output.insert(tk.END, text)

    def _write_reco(self, text: str) -> None:
        self.reco_output.delete("1.0", tk.END)
        self.reco_output.insert(tk.END, text)

    def _write_exe(self, text: str) -> None:
        self.exe_output.delete("1.0", tk.END)
        self.exe_output.insert(tk.END, text)

    def _parse_tags(self) -> list[str]:
        return [part.strip() for part in self.tags.get().split(",") if part.strip()]

    def generate_plan(self) -> None:
        try:
            text = build_plan_text(
                service=self.service.get(),
                profile=self.profile.get(),
                target_latency_ms=int(self.latency.get()),
                sample_rate_hz=int(self.sample_rate.get()),
                provider=self.provider.get() or None,
            )
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._write_plan(text)

    def open_service(self) -> None:
        try:
            service = self.service.get()
            studio.normalize_service(service)
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        studio.webbrowser.open(studio.STREAMING_SERVICES[studio.normalize_service(service)]["url"])

    def learn_feedback(self) -> None:
        tags = self._parse_tags()
        if not tags:
            messagebox.showerror(APP_TITLE, "タグを入力してください。")
            return
        self.reco_state.learn(tags, float(self.feedback.get()))
        payload = {
            "updates": self.reco_state.updates,
            "tag_weights": dict(sorted(self.reco_state.tag_weights.items())),
        }
        self._write_reco(json_dumps(payload))

    def compute_score(self) -> None:
        tags = self._parse_tags()
        if not tags:
            messagebox.showerror(APP_TITLE, "タグを入力してください。")
            return
        score = self.reco_state.score(tags)
        self._write_reco(
            f"tags={tags}\nscore={score:.4f}\nupdates={self.reco_state.updates}\nweights={json_dumps(self.reco_state.tag_weights)}"
        )

    def show_exe_commands(self) -> None:
        self._write_exe(build_exe_text())

    def save_plan_json(self) -> None:
        try:
            plan = studio.build_realtime_audio_plan(
                service=self.service.get(),
                profile_name=self.profile.get(),
                target_latency_ms=int(self.latency.get()),
                preferred_provider=self.provider.get() or None,
                sample_rate_hz=int(self.sample_rate.get()),
            )
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        output_path = Path.home() / "Desktop" / "snapdragon_streaming_plan.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_dumps(asdict(plan)) + "\n", encoding="utf-8")
        self._write_exe(f"Saved: {output_path}\n\n{build_exe_text()}")


def json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def create_app() -> "tk_types.Tk":
    if tk is None:
        raise RuntimeError("Tkinter is required to run this app")
    root = tk.Tk()
    StreamingStudioApp(root)
    return root


def main() -> int:
    if platform.system() == "Windows":
        create_app().mainloop()
        return 0
    create_app().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
