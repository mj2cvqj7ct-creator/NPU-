import argparse
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import windows_ldac_assistant as wla


class WindowsLdacAssistantTest(unittest.TestCase):
    def test_windows_status_reports_no_native_ldac_and_no_forcing(self):
        report = wla.build_diagnostic("Windows")

        self.assertFalse(report.native_ldac_available)
        self.assertFalse(report.safe_to_force_codec)
        self.assertIn("LDAC", report.message)

    def test_settings_are_protected_at_rest_and_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            store = wla.SettingsStore(data_dir, wla.PlaintextTestProtector())
            settings = wla.LdacSettings(
                desired_codec="LDAC",
                preferred_bitrate_kbps=990,
                start_on_login=True,
                created_at="2026-04-28T06:00:00+00:00",
            )

            store.save(settings)

            raw_file = store.path.read_text(encoding="ascii")
            protected_payload = base64.b64decode(raw_file)
            self.assertTrue(protected_payload.startswith(b"test:"))
            self.assertNotIn(b'"desired_codec"', protected_payload)
            self.assertEqual(store.load(), settings)

    def test_plaintext_test_protector_is_reversible_without_plaintext_payload(self):
        payload = json.dumps({"desired_codec": "LDAC"}).encode("utf-8")
        protected = wla.PlaintextTestProtector().protect(payload)

        self.assertNotIn(payload, protected)
        self.assertEqual(wla.PlaintextTestProtector().unprotect(protected), payload)

    def test_configure_saves_settings_and_registers_startup_dry_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                data_dir=data_dir,
                preferred_bitrate=660,
                start_on_login=True,
                protector=wla.PlaintextTestProtector(),
                dry_run=True,
                python_executable=Path("python.exe"),
                script=Path("windows_ldac_assistant.py"),
            )

            with mock.patch.object(wla, "install_startup") as install_startup:
                self.assertEqual(wla.configure(args), 0)

            saved = wla.SettingsStore(data_dir, wla.PlaintextTestProtector()).load()
            self.assertEqual(saved.preferred_bitrate_kbps, 660)
            self.assertTrue(saved.start_on_login)
            install_startup.assert_called_once()
            command = install_startup.call_args.args[0]
            self.assertIn("windows_ldac_assistant.py", command)
            self.assertIn("monitor", command)

    def test_startup_dry_run_does_not_require_windows(self):
        command = "python windows_ldac_assistant.py monitor"

        self.assertEqual(wla.install_startup(command, dry_run=True), command)

    def test_remove_startup_dry_run_does_not_require_windows(self):
        self.assertIsNone(wla.remove_startup(dry_run=True))


if __name__ == "__main__":
    unittest.main()
