from __future__ import annotations

from .audio import EnhancementReport


def build_status_text(report: EnhancementReport) -> str:
    return "\n".join(
        [
            f"Profile: {report.profile.name}",
            f"Target backend: {report.profile.target_backend}",
            f"Samples: {report.samples}",
            f"Input peak: {report.input_peak:.4f}",
            f"Output peak: {report.output_peak:.4f}",
        ]
    )
