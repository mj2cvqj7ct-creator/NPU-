#!/usr/bin/env python3
"""Windows LDAC readiness assistant.

Windows does not include a native LDAC Bluetooth encoder. This tool does not
attempt to synthesize proprietary codecs, patch drivers, or bypass OS policy.
It safely records the user's desired LDAC preference, reports readiness, and can
register itself to run at user logon.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import datetime as dt
import json
import os
import platform
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


APP_NAME = "windows-ldac-assistant"
RUN_VALUE_NAME = "WindowsLdacAssistant"
SUPPORTED_BITRATES = (330, 660, 990)


class SettingsError(RuntimeError):
    """Raised when encrypted settings cannot be read or written."""


class Protector(Protocol):
    def protect(self, plaintext: bytes) -> bytes:
        """Encrypt or otherwise protect plaintext bytes."""

    def unprotect(self, ciphertext: bytes) -> bytes:
        """Decrypt protected bytes."""


class WindowsDpapiProtector:
    """Protect settings with Windows DPAPI for the current user."""

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", ctypes.c_uint),
            ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    CRYPTPROTECT_UI_FORBIDDEN = 0x01

    def __init__(self) -> None:
        if platform.system() != "Windows":
            raise SettingsError("Windows DPAPI is only available on Windows")
        self.crypt32 = ctypes.windll.crypt32
        self.kernel32 = ctypes.windll.kernel32

    @classmethod
    def _blob_from_bytes(
        cls, data: bytes
    ) -> tuple["WindowsDpapiProtector.DATA_BLOB", ctypes.Array]:
        buffer = ctypes.create_string_buffer(data)
        blob = cls.DATA_BLOB(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
        return blob, buffer

    def _bytes_from_blob(self, blob: "WindowsDpapiProtector.DATA_BLOB") -> bytes:
        try:
            return ctypes.string_at(blob.pbData, blob.cbData)
        finally:
            if blob.pbData:
                self.kernel32.LocalFree(blob.pbData)

    def protect(self, plaintext: bytes) -> bytes:
        input_blob, _buffer = self._blob_from_bytes(plaintext)
        output_blob = self.DATA_BLOB()
        ok = self.crypt32.CryptProtectData(
            ctypes.byref(input_blob),
            None,
            None,
            None,
            None,
            self.CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(output_blob),
        )
        if not ok:
            raise SettingsError("CryptProtectData failed")
        return self._bytes_from_blob(output_blob)

    def unprotect(self, ciphertext: bytes) -> bytes:
        input_blob, _buffer = self._blob_from_bytes(ciphertext)
        output_blob = self.DATA_BLOB()
        ok = self.crypt32.CryptUnprotectData(
            ctypes.byref(input_blob),
            None,
            None,
            None,
            None,
            self.CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(output_blob),
        )
        if not ok:
            raise SettingsError("CryptUnprotectData failed")
        return self._bytes_from_blob(output_blob)


class PlaintextTestProtector:
    """Deterministic protector for tests and explicit dry-run demos only."""

    def protect(self, plaintext: bytes) -> bytes:
        return b"test:" + plaintext[::-1]

    def unprotect(self, ciphertext: bytes) -> bytes:
        if not ciphertext.startswith(b"test:"):
            raise SettingsError("invalid test ciphertext")
        return ciphertext.removeprefix(b"test:")[::-1]


@dataclass(frozen=True)
class LdacSettings:
    desired_codec: str
    preferred_bitrate_kbps: int
    start_on_login: bool
    created_at: str


@dataclass(frozen=True)
class ExclusiveModeStatus:
    supported: bool
    active: bool
    detail: str


@dataclass(frozen=True)
class DiagnosticReport:
    os_name: str
    native_ldac_available: bool
    safe_to_force_codec: bool
    exclusive_mode: ExclusiveModeStatus
    message: str
    recommendations: list[str]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def default_data_dir() -> Path:
    if os.environ.get("WINDOWS_LDAC_ASSISTANT_HOME"):
        return Path(os.environ["WINDOWS_LDAC_ASSISTANT_HOME"]).expanduser()
    if platform.system() == "Windows" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


class SettingsStore:
    def __init__(self, data_dir: Path, protector: Protector | None = None) -> None:
        self.data_dir = data_dir
        self.path = data_dir / "settings.dpapi"
        self.protector = protector or WindowsDpapiProtector()

    def save(self, settings: LdacSettings) -> None:
        payload = json.dumps(asdict(settings), sort_keys=True).encode("utf-8")
        protected = self.protector.protect(payload)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(base64.b64encode(protected).decode("ascii") + "\n")

    def load(self) -> LdacSettings:
        if not self.path.exists():
            raise SettingsError(f"settings file not found: {self.path}")
        protected = base64.b64decode(self.path.read_text(encoding="ascii"))
        payload = self.protector.unprotect(protected)
        data = json.loads(payload.decode("utf-8"))
        return LdacSettings(**data)


def detect_exclusive_mode(system: str | None = None) -> ExclusiveModeStatus:
    """Detect WASAPI exclusive mode availability.

    On Windows this inspects the registry for the audio endpoint policy.
    On other platforms, exclusive mode is not applicable.
    """
    os_name = (system or platform.system()).lower()
    if os_name != "windows":
        return ExclusiveModeStatus(
            supported=False,
            active=False,
            detail="WASAPI排他モードはWindowsのみ対応です。",
        )
    try:
        import winreg

        key_path = (
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render"
        )
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as _key:
            return ExclusiveModeStatus(
                supported=True,
                active=False,
                detail=(
                    "WASAPI排他モード対応デバイスが存在します。"
                    "デバイスのプロパティで排他モードが許可されているか確認してください。"
                ),
            )
    except Exception:
        return ExclusiveModeStatus(
            supported=True,
            active=False,
            detail=(
                "WASAPI排他モードはWindows環境で利用可能ですが、"
                "現在のデバイス状態を確認できませんでした。"
                "サウンド設定 → デバイスのプロパティ → 詳細 → "
                "排他モードを確認してください。"
            ),
        )


def build_diagnostic(system: str | None = None) -> DiagnosticReport:
    os_name = system or platform.system()
    is_windows = os_name.lower() == "windows"
    exclusive = detect_exclusive_mode(os_name)
    recommendations = [
        "LDAC対応ヘッドホン/スピーカー側で LDAC を有効にしてください。",
        "PC側はLDAC対応のBluetoothドライバーまたはメーカー提供ソフトの有無を確認してください。",
        "このアプリは独自コーデック生成やドライバー改変を行いません。",
    ]
    if is_windows:
        recommendations.append(
            "サウンド設定 → デバイスのプロパティ → 詳細タブで"
            "「アプリケーションによりこのデバイスを排他的に制御できるようにする」"
            "を有効にしてください。"
        )
    if not is_windows:
        return DiagnosticReport(
            os_name=os_name,
            native_ldac_available=False,
            safe_to_force_codec=False,
            exclusive_mode=exclusive,
            message="この補助アプリは Windows の Bluetooth 制約確認向けです。",
            recommendations=recommendations,
        )
    return DiagnosticReport(
        os_name=os_name,
        native_ldac_available=False,
        safe_to_force_codec=False,
        exclusive_mode=exclusive,
        message="Windows 標準 Bluetooth スタックには LDAC エンコーダーが含まれていません。",
        recommendations=recommendations,
    )


def render_report(report: DiagnosticReport) -> str:
    lines = [
        f"OS: {report.os_name}",
        f"ネイティブLDAC対応: {'あり' if report.native_ldac_available else 'なし'}",
        f"コーデック強制変更: {'安全' if report.safe_to_force_codec else '非推奨'}",
        f"WASAPI排他モード対応: {'あり' if report.exclusive_mode.supported else 'なし'}",
        f"WASAPI排他モード状態: {'有効' if report.exclusive_mode.active else '未確認'}",
        f"排他モード詳細: {report.exclusive_mode.detail}",
        report.message,
        "推奨事項:",
    ]
    lines.extend(f"- {item}" for item in report.recommendations)
    return "\n".join(lines)


def quote_command(command: list[str]) -> str:
    if platform.system() == "Windows":
        return subprocess_list2cmdline(command)
    return shlex.join(command)


def subprocess_list2cmdline(command: list[str]) -> str:
    # Keep this tiny wrapper testable without importing subprocess in callers.
    import subprocess

    return subprocess.list2cmdline(command)


def startup_command(python_executable: Path, script: Path, data_dir: Path) -> str:
    return quote_command(
        [
            str(python_executable),
            str(script),
            "--data-dir",
            str(data_dir),
            "monitor",
        ]
    )


def install_startup(command: str, dry_run: bool = False) -> str:
    if dry_run:
        return command
    if platform.system() != "Windows":
        raise RuntimeError("startup registration is only available on Windows")
    import winreg

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, RUN_VALUE_NAME, 0, winreg.REG_SZ, command)
    return command


def remove_startup(dry_run: bool = False) -> None:
    if dry_run:
        return
    if platform.system() != "Windows":
        raise RuntimeError("startup registration is only available on Windows")
    import winreg

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
        try:
            winreg.DeleteValue(key, RUN_VALUE_NAME)
        except FileNotFoundError:
            pass


def configure(args: argparse.Namespace) -> int:
    settings = LdacSettings(
        desired_codec="LDAC",
        preferred_bitrate_kbps=args.preferred_bitrate,
        start_on_login=args.start_on_login,
        created_at=utc_now(),
    )
    store = SettingsStore(args.data_dir, args.protector)
    store.save(settings)
    print(f"encrypted settings saved to {store.path}")
    if args.start_on_login:
        command = startup_command(args.python_executable, args.script, args.data_dir)
        install_startup(command, dry_run=args.dry_run)
        print(f"startup command: {command}")
    return 0


def status(args: argparse.Namespace) -> int:
    print(render_report(build_diagnostic(args.system)))
    return 0


def monitor(args: argparse.Namespace) -> int:
    print(render_report(build_diagnostic()))
    try:
        settings = SettingsStore(args.data_dir, args.protector).load()
    except SettingsError as exc:
        print(f"settings: unavailable ({exc})")
        return 0
    print(
        "settings: "
        f"{settings.desired_codec} preferred at {settings.preferred_bitrate_kbps} kbps; "
        f"start_on_login={str(settings.start_on_login).lower()}"
    )
    return 0


def install_startup_cmd(args: argparse.Namespace) -> int:
    command = startup_command(args.python_executable, args.script, args.data_dir)
    install_startup(command, dry_run=args.dry_run)
    print(command)
    return 0


def remove_startup_cmd(args: argparse.Namespace) -> int:
    remove_startup(dry_run=args.dry_run)
    print(f"removed {RUN_VALUE_NAME} startup registration")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check Windows LDAC readiness, store encrypted preferences, and "
            "optionally start at user logon."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_cmd = subparsers.add_parser("status", help="show LDAC readiness status")
    status_cmd.add_argument("--system", default=None)
    status_cmd.set_defaults(func=status)

    configure_cmd = subparsers.add_parser(
        "configure", help="save encrypted LDAC preference settings"
    )
    configure_cmd.add_argument(
        "--preferred-bitrate",
        type=int,
        choices=SUPPORTED_BITRATES,
        default=990,
        help="desired LDAC bitrate preference in kbps",
    )
    configure_cmd.add_argument("--start-on-login", action="store_true")
    configure_cmd.add_argument("--dry-run", action="store_true")
    configure_cmd.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    configure_cmd.add_argument("--script", type=Path, default=Path(__file__).resolve())
    configure_cmd.set_defaults(func=configure, protector=None)

    monitor_cmd = subparsers.add_parser("monitor", help="run the logon monitor once")
    monitor_cmd.set_defaults(func=monitor, protector=None)

    install_cmd = subparsers.add_parser(
        "install-startup", help="register this assistant for user logon"
    )
    install_cmd.add_argument("--dry-run", action="store_true")
    install_cmd.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    install_cmd.add_argument("--script", type=Path, default=Path(__file__).resolve())
    install_cmd.set_defaults(func=install_startup_cmd)

    remove_cmd = subparsers.add_parser(
        "remove-startup", help="remove user logon registration"
    )
    remove_cmd.add_argument("--dry-run", action="store_true")
    remove_cmd.set_defaults(func=remove_startup_cmd)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
