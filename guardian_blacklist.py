#!/usr/bin/env python3
"""Local defensive blacklist helper.

This tool blocks and documents suspicious hosts on the user's own computer.
It deliberately does not contact providers, police, banks, or any other third
party. Reports are generated for the user to review and submit manually.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import ipaddress
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


APP_NAME = "guardian-blacklist"
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f:.%]+")
PORT_RE = re.compile(
    r"\b(?:dpt|dst_port|destination_port|port)\s*[=: ]\s*(\d{1,5})\b",
    re.IGNORECASE,
)
DEFAULT_SCAN_INTERVAL_SECONDS = 1
DEFAULT_PORT_SCAN_THRESHOLD = 10
DEFAULT_ABUSEIPDB_CATEGORIES = "14,15"


@dataclass(frozen=True)
class BlacklistEntry:
    ip: str
    reason: str
    source: str
    evidence: str
    created_at: str


def default_data_dir() -> Path:
    if os.environ.get("GUARDIAN_BLACKLIST_HOME"):
        return Path(os.environ["GUARDIAN_BLACKLIST_HOME"]).expanduser()
    return Path.home() / ".local" / "share" / APP_NAME


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def validate_blockable_ip(value: str) -> str:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid IP address: {value}") from exc
    if not ip.is_global:
        raise argparse.ArgumentTypeError(
            "refusing to blacklist private, loopback, reserved, or multicast IPs"
        )
    return str(ip)


class BlacklistStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.path = data_dir / "blacklist.json"

    def load(self) -> list[BlacklistEntry]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as file:
            raw_entries = json.load(file)
        return [BlacklistEntry(**entry) for entry in raw_entries]

    def save(self, entries: Iterable[BlacklistEntry]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        serializable = [asdict(entry) for entry in entries]
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(serializable, file, ensure_ascii=False, indent=2)
            file.write("\n")

    def add(self, entry: BlacklistEntry) -> bool:
        entries = self.load()
        if any(existing.ip == entry.ip for existing in entries):
            return False
        entries.append(entry)
        self.save(entries)
        return True


def firewall_commands(ip: str, system: str | None = None) -> list[list[str]]:
    os_name = (system or platform.system()).lower()
    address = ipaddress.ip_address(ip)
    if os_name == "windows":
        return [
            [
                "netsh",
                "advfirewall",
                "firewall",
                "add",
                "rule",
                f"name={APP_NAME}-{ip}",
                "dir=in",
                "action=block",
                f"remoteip={ip}",
            ],
            [
                "netsh",
                "advfirewall",
                "firewall",
                "add",
                "rule",
                f"name={APP_NAME}-{ip}",
                "dir=out",
                "action=block",
                f"remoteip={ip}",
            ],
        ]
    if os_name == "darwin":
        return [["sudo", "pfctl", "-t", APP_NAME, "-T", "add", ip]]
    set_name = APP_NAME if address.version == 4 else f"{APP_NAME}-ipv6"
    return [
        ["sudo", "nft", "add", "element", "inet", "filter", set_name, f"{{ {ip} }}"]
    ]


def format_command(command: list[str]) -> str:
    return " ".join(command)


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def iter_ip_candidates(text: str) -> Iterable[str]:
    for match in IPV4_RE.finditer(text):
        yield match.group(0)
    for match in IPV6_RE.finditer(text):
        candidate = match.group(0).strip("[](),;")
        if "%" in candidate:
            candidate = candidate.split("%", 1)[0]
        yield candidate


def iter_ports(text: str) -> Iterable[int]:
    for match in PORT_RE.finditer(text):
        port = int(match.group(1))
        if 1 <= port <= 65535:
            yield port


def scan_log_file(
    data_dir: Path,
    log_file: Path,
    threshold: int,
    reason: str,
    port_threshold: int = DEFAULT_PORT_SCAN_THRESHOLD,
) -> list[BlacklistEntry]:
    store = BlacklistStore(data_dir)
    text = log_file.read_text(encoding="utf-8", errors="replace")
    counts: dict[str, int] = {}
    ports_by_ip: dict[str, set[int]] = {}
    for line in text.splitlines():
        ports = set(iter_ports(line))
        for candidate in iter_ip_candidates(line):
            try:
                ip = validate_blockable_ip(candidate)
            except argparse.ArgumentTypeError:
                continue
            counts[ip] = counts.get(ip, 0) + 1
            if ports:
                ports_by_ip.setdefault(ip, set()).update(ports)

    added_entries: list[BlacklistEntry] = []
    for ip, count in sorted(counts.items()):
        distinct_ports = ports_by_ip.get(ip, set())
        if count < threshold and len(distinct_ports) < port_threshold:
            continue
        evidence_parts = []
        if count >= threshold:
            evidence_parts.append(
                f"possible IP scan/repeated activity: {log_file} mentions this IP {count} times"
            )
        if len(distinct_ports) >= port_threshold:
            evidence_parts.append(
                "possible port scan: "
                f"{len(distinct_ports)} distinct destination ports observed"
            )
        entry = BlacklistEntry(
            ip=ip,
            reason=reason,
            source=str(log_file),
            evidence="; ".join(evidence_parts),
            created_at=utc_now(),
        )
        if store.add(entry):
            added_entries.append(entry)
    return added_entries


def add_entry(args: argparse.Namespace) -> int:
    store = BlacklistStore(args.data_dir)
    entry = BlacklistEntry(
        ip=args.ip,
        reason=args.reason,
        source=args.source,
        evidence=args.evidence,
        created_at=utc_now(),
    )
    created = store.add(entry)
    if created:
        print(f"added {entry.ip} to local blacklist")
    else:
        print(f"{entry.ip} already exists in local blacklist")

    for command in firewall_commands(entry.ip, args.system):
        print(format_command(command))

    if args.apply:
        for command in firewall_commands(entry.ip, args.system):
            subprocess.run(command, check=True)
    return 0


def list_entries(args: argparse.Namespace) -> int:
    entries = BlacklistStore(args.data_dir).load()
    if not entries:
        print("local blacklist is empty")
        return 0
    for entry in entries:
        print(f"{entry.ip}\t{entry.created_at}\t{entry.reason}\t{entry.source}")
    return 0


def scan_log(args: argparse.Namespace) -> int:
    added_entries = scan_log_file(
        args.data_dir,
        args.log_file,
        args.threshold,
        args.reason,
        args.port_scan_threshold,
    )
    for entry in added_entries:
        print(f"added {entry.ip}: {entry.evidence}")
    print(f"scan complete: {len(added_entries)} new entries")
    return 0


def apply_firewall_for_entries(
    entries: Iterable[BlacklistEntry],
    system: str | None = None,
) -> None:
    for entry in entries:
        for command in firewall_commands(entry.ip, system):
            print(format_command(command))
            subprocess.run(command, check=True)


def watch_log(args: argparse.Namespace) -> int:
    while True:
        if args.log_file.exists():
            added_entries = scan_log_file(
                args.data_dir,
                args.log_file,
                args.threshold,
                args.reason,
                args.port_scan_threshold,
            )
            for entry in added_entries:
                print(f"added {entry.ip}: {entry.evidence}", flush=True)
            if added_entries and getattr(args, "abuseipdb_export", None):
                write_abuseipdb_report(
                    args.data_dir,
                    args.abuseipdb_export,
                    args.abuseipdb_export_format,
                    args.abuseipdb_categories,
                )
                print(
                    f"updated AbuseIPDB manual report at {args.abuseipdb_export}",
                    flush=True,
                )
            if args.apply and added_entries:
                apply_firewall_for_entries(added_entries, args.system)
        else:
            print(f"waiting for log file: {args.log_file}", flush=True)

        if args.once:
            return 0
        time.sleep(args.interval)


def systemd_user_service_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def systemd_system_service_dir() -> Path:
    return Path("/etc/systemd/system")


def build_watch_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-dir",
        str(args.data_dir),
        "watch-log",
        str(args.log_file),
        "--threshold",
        str(args.threshold),
        "--interval",
        str(args.interval),
        "--port-scan-threshold",
        str(args.port_scan_threshold),
        "--reason",
        args.reason,
    ]
    if getattr(args, "abuseipdb_export", None):
        command.extend(
            [
                "--abuseipdb-export",
                str(args.abuseipdb_export),
                "--abuseipdb-export-format",
                args.abuseipdb_export_format,
                "--abuseipdb-categories",
                args.abuseipdb_categories,
            ]
        )
    if args.apply:
        command.append("--apply")
    return command


def write_systemd_service(
    service_path: Path,
    exec_start: str,
    description: str,
    after: str,
    wanted_by: str,
) -> None:
    service_path.write_text(
        "\n".join(
            [
                "[Unit]",
                f"Description={description}",
                f"After={after}",
                "",
                "[Service]",
                "Type=simple",
                f"ExecStart={exec_start}",
                "Restart=on-failure",
                "RestartSec=10",
                "",
                "[Install]",
                f"WantedBy={wanted_by}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def ensure_linux(command_name: str, service_type: str) -> bool:
    if platform.system() != "Linux":
        print(f"{command_name} currently writes a Linux systemd {service_type} only")
        return False
    return True


def install_autostart(args: argparse.Namespace) -> int:
    if not ensure_linux("install-autostart", "user service"):
        return 2

    service_dir = args.service_dir or systemd_user_service_dir()
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / f"{APP_NAME}.service"
    write_systemd_service(
        service_path,
        shell_join(build_watch_command(args)),
        "Guardian Blacklist local log watcher",
        "default.target",
        "default.target",
    )
    print(f"wrote autostart service to {service_path}")
    print("enable with: systemctl --user enable --now guardian-blacklist.service")

    if args.enable:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", f"{APP_NAME}.service"],
            check=True,
        )
    return 0


def install_boot_service(args: argparse.Namespace) -> int:
    if not ensure_linux("install-boot-service", "system service"):
        return 2

    service_dir = args.service_dir or systemd_system_service_dir()
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / f"{APP_NAME}.service"
    write_systemd_service(
        service_path,
        shell_join(build_watch_command(args)),
        "Guardian Blacklist boot-time local log watcher",
        "network-online.target",
        "multi-user.target",
    )
    print(f"wrote boot service to {service_path}")
    print("enable with: sudo systemctl enable --now guardian-blacklist.service")

    if args.enable:
        prefix = [] if os.geteuid() == 0 else ["sudo"]
        subprocess.run([*prefix, "systemctl", "daemon-reload"], check=True)
        subprocess.run(
            [*prefix, "systemctl", "enable", "--now", f"{APP_NAME}.service"],
            check=True,
        )
    return 0


def add_watcher_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument(
        "--port-scan-threshold",
        type=int,
        default=DEFAULT_PORT_SCAN_THRESHOLD,
        help="distinct destination ports from one IP that indicate a port scan",
    )
    parser.add_argument("--reason", default="Repeated suspicious log activity")
    parser.add_argument("--interval", type=int, default=DEFAULT_SCAN_INTERVAL_SECONDS)
    parser.add_argument(
        "--abuseipdb-export",
        type=Path,
        default=None,
        help="update an AbuseIPDB manual submission file when new entries are detected",
    )
    parser.add_argument(
        "--abuseipdb-export-format",
        choices=["json", "csv"],
        default="json",
        help="format for --abuseipdb-export",
    )
    parser.add_argument(
        "--abuseipdb-categories",
        default=DEFAULT_ABUSEIPDB_CATEGORIES,
        help="comma-separated AbuseIPDB category IDs to suggest after review",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="include local firewall application in the watcher",
    )


def add_service_arguments(parser: argparse.ArgumentParser) -> None:
    add_watcher_arguments(parser)
    parser.add_argument(
        "--enable",
        action="store_true",
        help="enable and start the service after writing it",
    )
    parser.add_argument(
        "--service-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )


def evidence_digest(entries: Iterable[BlacklistEntry]) -> str:
    payload = json.dumps([asdict(entry) for entry in entries], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def report_contacts(audience: str) -> list[str]:
    if audience == "japan-international":
        return [
            "- Japanese police cybercrime consultation desk, using its official channel",
            "- JPCERT/CC or the relevant national CERT/CSIRT official reporting channel",
            "- The affected network owner's official abuse or security desk",
            "- International anti-phishing or malware reporting portals, when relevant",
        ]
    if audience == "international":
        return [
            "- Your national CERT/CSIRT official reporting channel",
            "- The affected network owner's official abuse or security desk",
            "- International anti-phishing or malware reporting portals, when relevant",
            "- Local law enforcement first, if personal safety or financial loss is involved",
        ]
    if audience == "public":
        return [
            "- Local police cybercrime consultation desk",
            "- National or regional cyber incident consultation desk",
            "- Your internet provider abuse or security desk",
            "- Your bank's official fraud desk, if account access or payments may be affected",
        ]
    return [
        "- Your internet provider abuse or security desk",
        "- Local police cybercrime consultation desk",
        "- Your bank's official fraud desk, if account access or payments may be affected",
    ]


def report(args: argparse.Namespace) -> int:
    entries = BlacklistStore(args.data_dir).load()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()
    digest = evidence_digest(entries)
    audience = getattr(args, "audience", "general")
    lines = [
        "# Guardian Blacklist Incident Report",
        "",
        f"- Generated at: {generated_at}",
        f"- Entries: {len(entries)}",
        f"- Evidence digest: `{digest}`",
        f"- Intended audience: {audience}",
        "",
        "## Important legal note",
        "",
        "This report is for manual review and submission to legitimate contacts.",
        "The tool does not automatically register, submit, message, or accuse anyone with providers, public agencies, international organizations, or public blacklists.",
        "Verify the evidence and use only official reporting channels before submitting anything.",
        "",
        "## Suggested manual contacts",
        "",
        *report_contacts(audience),
        "",
        "## Entries",
        "",
    ]
    if not entries:
        lines.append("No entries recorded.")
    for index, entry in enumerate(entries, start=1):
        lines.extend(
            [
                f"### {index}. {entry.ip}",
                "",
                f"- Created at: {entry.created_at}",
                f"- Reason: {entry.reason}",
                f"- Source: {entry.source}",
                f"- Evidence: {entry.evidence}",
                "",
            ]
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report to {args.output}")
    return 0


def abuseipdb_comment(entry: BlacklistEntry) -> str:
    return (
        f"{entry.reason}. Evidence: {entry.evidence}. "
        f"Source retained locally: {entry.source}. "
        "Manual review required before submitting to AbuseIPDB."
    )


def write_abuseipdb_report(
    data_dir: Path,
    output: Path,
    output_format: str,
    categories: str,
) -> None:
    entries = BlacklistStore(data_dir).load()
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()

    if output_format == "csv":
        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["ip", "categories", "comment", "timestamp"],
            )
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        "ip": entry.ip,
                        "categories": categories,
                        "comment": abuseipdb_comment(entry),
                        "timestamp": entry.created_at,
                    }
                )
    else:
        payload = {
            "generated_at": generated_at,
            "manual_submission_only": True,
            "does_not_submit_to_abuseipdb": True,
            "legal_note": (
                "Review every entry before submitting through official AbuseIPDB "
                "channels. This tool does not call the AbuseIPDB API."
            ),
            "entries": [
                {
                    "ip": entry.ip,
                    "categories": categories,
                    "comment": abuseipdb_comment(entry),
                    "timestamp": entry.created_at,
                }
                for entry in entries
            ],
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def abuseipdb_report(args: argparse.Namespace) -> int:
    write_abuseipdb_report(args.data_dir, args.output, args.format, args.categories)
    print(f"wrote AbuseIPDB manual report to {args.output}")
    print("manual submission only: no AbuseIPDB API call was made")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Maintain a local defensive blacklist and generate manual incident reports."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="directory for blacklist data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="add one public IPv4 or IPv6 address")
    add.add_argument("ip", type=validate_blockable_ip)
    add.add_argument("--reason", required=True)
    add.add_argument("--source", required=True)
    add.add_argument("--evidence", required=True)
    add.add_argument("--system", choices=["Linux", "Darwin", "Windows"], default=None)
    add.add_argument(
        "--apply",
        action="store_true",
        help="run the local firewall command; requires administrator privileges",
    )
    add.set_defaults(func=add_entry)

    list_cmd = subparsers.add_parser("list", help="list recorded entries")
    list_cmd.set_defaults(func=list_entries)

    scan = subparsers.add_parser("scan-log", help="add public IPs seen repeatedly")
    scan.add_argument("log_file", type=Path)
    scan.add_argument("--threshold", type=int, default=5)
    scan.add_argument(
        "--port-scan-threshold",
        type=int,
        default=DEFAULT_PORT_SCAN_THRESHOLD,
        help="distinct destination ports from one IP that indicate a port scan",
    )
    scan.add_argument("--reason", default="Repeated suspicious log activity")
    scan.set_defaults(func=scan_log)

    watch = subparsers.add_parser(
        "watch-log",
        help="run continuously and add public IPs seen repeatedly in a log file",
    )
    add_watcher_arguments(watch)
    watch.add_argument("--system", choices=["Linux", "Darwin", "Windows"], default=None)
    watch.add_argument(
        "--once",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    watch.set_defaults(func=watch_log)

    autostart = subparsers.add_parser(
        "install-autostart",
        help="write a startup service that runs watch-log when the user logs in",
    )
    add_service_arguments(autostart)
    autostart.set_defaults(func=install_autostart)

    boot_service = subparsers.add_parser(
        "install-boot-service",
        help="write a system service that runs watch-log when the OS boots",
    )
    add_service_arguments(boot_service)
    boot_service.set_defaults(func=install_boot_service)

    report_cmd = subparsers.add_parser("report", help="write a manual report")
    report_cmd.add_argument("output", type=Path)
    report_cmd.add_argument(
        "--audience",
        choices=["general", "public", "international", "japan-international"],
        default="general",
        help="tailor manual contact suggestions without submitting anything",
    )
    report_cmd.set_defaults(func=report)

    abuseipdb = subparsers.add_parser(
        "abuseipdb-report",
        help="write an AbuseIPDB manual submission file without submitting it",
    )
    abuseipdb.add_argument("output", type=Path)
    abuseipdb.add_argument("--format", choices=["json", "csv"], default="json")
    abuseipdb.add_argument(
        "--categories",
        default=DEFAULT_ABUSEIPDB_CATEGORIES,
        help="comma-separated AbuseIPDB category IDs to suggest after review",
    )
    abuseipdb.set_defaults(func=abuseipdb_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
