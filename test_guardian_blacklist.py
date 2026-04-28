import argparse
import tempfile
import unittest
from pathlib import Path

import guardian_blacklist as gb


class GuardianBlacklistTest(unittest.TestCase):
    def test_rejects_private_addresses(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            gb.validate_blockable_ip("192.168.1.10")
        with self.assertRaises(argparse.ArgumentTypeError):
            gb.validate_blockable_ip("fd00::1")

    def test_accepts_global_ipv4_and_ipv6_addresses(self):
        self.assertEqual(gb.validate_blockable_ip("8.8.8.8"), "8.8.8.8")
        self.assertEqual(gb.validate_blockable_ip("2001:4860:4860::8888"), "2001:4860:4860::8888")

    def test_add_list_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            store = gb.BlacklistStore(data_dir)
            created = store.add(
                gb.BlacklistEntry(
                    ip="8.8.8.8",
                    reason="Suspicious repeated connection attempts",
                    source="firewall.log",
                    evidence="Observed repeated denied inbound attempts",
                    created_at="2026-04-28T02:00:00+00:00",
                )
            )
            self.assertTrue(created)
            self.assertFalse(store.add(store.load()[0]))

            report_path = data_dir / "report.md"
            args = argparse.Namespace(
                data_dir=data_dir,
                output=report_path,
                audience="japan-international",
            )
            self.assertEqual(gb.report(args), 0)
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("8.8.8.8", report_text)
            self.assertIn("Intended audience: japan-international", report_text)
            self.assertIn("JPCERT/CC", report_text)
            self.assertIn("national CERT/CSIRT", report_text)
            self.assertIn("does not automatically register", report_text)
            self.assertIn("international organizations", report_text)

    def test_scan_log_respects_threshold_for_ipv4_and_ipv6(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            log_path = data_dir / "firewall.log"
            log_path.write_text(
                "denied 1.1.1.1\n"
                "allowed 8.8.8.8\n"
                "denied 1.1.1.1\n"
                "denied [2001:4860:4860::8888]\n"
                "denied 2001:4860:4860::8888\n"
                "ignored fd00::1\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                data_dir=data_dir,
                log_file=log_path,
                threshold=2,
                port_scan_threshold=10,
                reason="Repeated suspicious log activity",
            )
            self.assertEqual(gb.scan_log(args), 0)
            entries = gb.BlacklistStore(data_dir).load()
            self.assertEqual(
                [entry.ip for entry in entries],
                ["1.1.1.1", "2001:4860:4860::8888"],
            )

    def test_scan_log_detects_port_scan_by_distinct_ports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            log_path = data_dir / "firewall.log"
            log_path.write_text(
                "\n".join(
                    f"denied src=9.9.9.9 dpt={port}"
                    for port in [22, 23, 25, 53, 80]
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                data_dir=data_dir,
                log_file=log_path,
                threshold=99,
                port_scan_threshold=5,
                reason="Port scan or IP scan detected",
            )

            self.assertEqual(gb.scan_log(args), 0)

            entries = gb.BlacklistStore(data_dir).load()
            self.assertEqual([entry.ip for entry in entries], ["9.9.9.9"])
            self.assertIn("possible port scan", entries[0].evidence)

    def test_abuseipdb_report_is_manual_submission_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            store = gb.BlacklistStore(data_dir)
            self.assertTrue(
                store.add(
                    gb.BlacklistEntry(
                        ip="9.9.9.9",
                        reason="Port scan detected",
                        source="firewall.log",
                        evidence="Observed distinct destination ports",
                        created_at="2026-04-28T05:00:00+00:00",
                    )
                )
            )
            report_path = data_dir / "abuseipdb.json"
            args = argparse.Namespace(
                data_dir=data_dir,
                output=report_path,
                format="json",
                categories="14,15",
            )

            self.assertEqual(gb.abuseipdb_report(args), 0)

            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn('"manual_submission_only": true', report_text)
            self.assertIn('"does_not_submit_to_abuseipdb": true', report_text)
            self.assertIn("Manual review required", report_text)
            self.assertIn("9.9.9.9", report_text)

    def test_watch_log_updates_abuseipdb_manual_export_when_new_entries_arrive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            log_path = data_dir / "firewall.log"
            export_path = data_dir / "abuseipdb.json"
            log_path.write_text("deny 9.9.9.9\ndeny 9.9.9.9\n", encoding="utf-8")
            args = argparse.Namespace(
                data_dir=data_dir,
                log_file=log_path,
                threshold=2,
                reason="Repeated suspicious log activity",
                apply=False,
                system=None,
                once=True,
                interval=1,
                port_scan_threshold=10,
                abuseipdb_export=export_path,
                abuseipdb_export_format="json",
                abuseipdb_categories="14,15",
            )

            self.assertEqual(gb.watch_log(args), 0)

            report_text = export_path.read_text(encoding="utf-8")
            self.assertIn('"manual_submission_only": true', report_text)
            self.assertIn('"does_not_submit_to_abuseipdb": true', report_text)
            self.assertIn("9.9.9.9", report_text)

    def test_watch_log_once_scans_without_looping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            log_path = data_dir / "firewall.log"
            log_path.write_text("deny 9.9.9.9\ndeny 9.9.9.9\n", encoding="utf-8")
            args = argparse.Namespace(
                data_dir=data_dir,
                log_file=log_path,
                threshold=2,
                reason="Repeated suspicious log activity",
                apply=False,
                system=None,
                once=True,
                interval=1,
                port_scan_threshold=10,
                abuseipdb_export=None,
                abuseipdb_export_format="json",
                abuseipdb_categories="14,15",
            )

            self.assertEqual(gb.watch_log(args), 0)

            entries = gb.BlacklistStore(data_dir).load()
            self.assertEqual([entry.ip for entry in entries], ["9.9.9.9"])

    def test_install_autostart_writes_systemd_user_service(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            service_dir = Path(temp_dir) / "systemd"
            log_path = Path(temp_dir) / "firewall log.txt"
            args = argparse.Namespace(
                data_dir=data_dir,
                log_file=log_path,
                threshold=3,
                reason="Repeated suspicious log activity",
                interval=15,
                port_scan_threshold=10,
                abuseipdb_export=None,
                abuseipdb_export_format="json",
                abuseipdb_categories="14,15",
                apply=False,
                enable=False,
                service_dir=service_dir,
            )

            self.assertEqual(gb.install_autostart(args), 0)

            service_text = (service_dir / "guardian-blacklist.service").read_text(
                encoding="utf-8"
            )
            self.assertIn("[Service]", service_text)
            self.assertIn("ExecStart=", service_text)
            self.assertIn("watch-log", service_text)
            self.assertIn("--data-dir", service_text)
            self.assertIn(f"'{log_path}'", service_text)
            self.assertNotIn("police", service_text.lower())
            self.assertNotIn("bank", service_text.lower())

    def test_install_boot_service_writes_systemd_system_service(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            service_dir = Path(temp_dir) / "systemd-system"
            log_path = Path(temp_dir) / "security.log"
            args = argparse.Namespace(
                data_dir=data_dir,
                log_file=log_path,
                threshold=2,
                reason="Repeated suspicious log activity",
                interval=1,
                port_scan_threshold=4,
                abuseipdb_export=Path(temp_dir) / "abuseipdb.json",
                abuseipdb_export_format="json",
                abuseipdb_categories="14,15",
                apply=True,
                enable=False,
                service_dir=service_dir,
            )

            self.assertEqual(gb.install_boot_service(args), 0)

            service_text = (service_dir / "guardian-blacklist.service").read_text(
                encoding="utf-8"
            )
            self.assertIn("After=network-online.target", service_text)
            self.assertIn("WantedBy=multi-user.target", service_text)
            self.assertIn("--interval 1", service_text)
            self.assertIn("--port-scan-threshold 4", service_text)
            self.assertIn("--abuseipdb-export", service_text)
            self.assertIn("--apply", service_text)
            self.assertNotIn("police", service_text.lower())
            self.assertNotIn("bank", service_text.lower())

    def test_firewall_commands_are_local_only(self):
        commands = gb.firewall_commands("8.8.8.8", "Windows")
        rendered = "\n".join(gb.format_command(command) for command in commands)
        self.assertIn("netsh advfirewall firewall add rule", rendered)
        self.assertNotIn("police", rendered.lower())
        self.assertNotIn("bank", rendered.lower())

    def test_linux_firewall_commands_use_ipv6_set_for_ipv6(self):
        commands = gb.firewall_commands("2001:4860:4860::8888", "Linux")
        rendered = "\n".join(gb.format_command(command) for command in commands)
        self.assertIn("guardian-blacklist-ipv6", rendered)
        self.assertIn("2001:4860:4860::8888", rendered)


if __name__ == "__main__":
    unittest.main()
