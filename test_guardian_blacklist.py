import argparse
import tempfile
import unittest
from pathlib import Path

import guardian_blacklist as gb


class GuardianBlacklistTest(unittest.TestCase):
    def test_rejects_private_addresses(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            gb.validate_blockable_ip("192.168.1.10")

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
            args = argparse.Namespace(data_dir=data_dir, output=report_path)
            self.assertEqual(gb.report(args), 0)
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("8.8.8.8", report_text)
            self.assertIn("does not automatically register anyone", report_text)

    def test_scan_log_respects_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            log_path = data_dir / "firewall.log"
            log_path.write_text(
                "denied 1.1.1.1\nallowed 8.8.8.8\ndenied 1.1.1.1\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                data_dir=data_dir,
                log_file=log_path,
                threshold=2,
                reason="Repeated suspicious log activity",
            )
            self.assertEqual(gb.scan_log(args), 0)
            entries = gb.BlacklistStore(data_dir).load()
            self.assertEqual([entry.ip for entry in entries], ["1.1.1.1"])

    def test_firewall_commands_are_local_only(self):
        commands = gb.firewall_commands("8.8.8.8", "Windows")
        rendered = "\n".join(gb.format_command(command) for command in commands)
        self.assertIn("netsh advfirewall firewall add rule", rendered)
        self.assertNotIn("police", rendered.lower())
        self.assertNotIn("bank", rendered.lower())


if __name__ == "__main__":
    unittest.main()
