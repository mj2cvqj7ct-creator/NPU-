#!/usr/bin/env python3
"""Local defensive blacklist helper.

This tool blocks and documents suspicious hosts on the user's own computer.
It deliberately does not contact providers, police, banks, or any other third
party. Reports are generated for the user to review and submit manually.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


APP_NAME = "guardian-blacklist"
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


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
    if ip.version != 4:
        raise argparse.ArgumentTypeError("only IPv4 addresses are supported")
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
    return [
        ["sudo", "nft", "add", "element", "inet", "filter", APP_NAME, f"{{ {ip} }}"]
    ]


def format_command(command: list[str]) -> str:
    return " ".join(command)


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
    store = BlacklistStore(args.data_dir)
    text = args.log_file.read_text(encoding="utf-8", errors="replace")
    counts: dict[str, int] = {}
    for candidate in IP_RE.findall(text):
        try:
            ip = validate_blockable_ip(candidate)
        except argparse.ArgumentTypeError:
            continue
        counts[ip] = counts.get(ip, 0) + 1

    added = 0
    for ip, count in sorted(counts.items()):
        if count < args.threshold:
            continue
        evidence = f"{args.log_file} mentions this IP {count} times"
        entry = BlacklistEntry(
            ip=ip,
            reason=args.reason,
            source=str(args.log_file),
            evidence=evidence,
            created_at=utc_now(),
        )
        if store.add(entry):
            added += 1
            print(f"added {ip}: {evidence}")
    print(f"scan complete: {added} new entries")
    return 0


def evidence_digest(entries: Iterable[BlacklistEntry]) -> str:
    payload = json.dumps([asdict(entry) for entry in entries], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def report(args: argparse.Namespace) -> int:
    entries = BlacklistStore(args.data_dir).load()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()
    digest = evidence_digest(entries)
    lines = [
        "# Guardian Blacklist Incident Report",
        "",
        f"- Generated at: {generated_at}",
        f"- Entries: {len(entries)}",
        f"- Evidence digest: `{digest}`",
        "",
        "## Important legal note",
        "",
        "This report is for manual review and submission to legitimate contacts.",
        "The tool does not automatically register anyone with providers, police, banks, or public blacklists.",
        "",
        "## Suggested manual contacts",
        "",
        "- Your internet provider abuse or security desk",
        "- Local police cybercrime consultation desk",
        "- Your bank's official fraud desk, if account access or payments may be affected",
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

    add = subparsers.add_parser("add", help="add one public IPv4 address")
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
    scan.add_argument("--reason", default="Repeated suspicious log activity")
    scan.set_defaults(func=scan_log)

    report_cmd = subparsers.add_parser("report", help="write a manual report")
    report_cmd.add_argument("output", type=Path)
    report_cmd.set_defaults(func=report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
