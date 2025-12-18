from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    """
    At the end of the test session, write a small markdown report
    summarising results to a fixed path (reports/test_report.md).
    This avoids needing any extra CLI options so it works with bare `pytest -q`.
    """
    # Always write to this path relative to project root
    report_path = Path("reports") / "test_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    stats = terminalreporter.stats
    total_collected = terminalreporter._numcollected
    num_passed = len(stats.get("passed", []))
    num_failed = len(stats.get("failed", []))
    num_error = len(stats.get("error", []))
    num_skipped = len(stats.get("skipped", []))
    num_xfailed = len(stats.get("xfailed", []))
    num_xpassed = len(stats.get("xpassed", []))

    timestamp = datetime.now().isoformat(timespec="seconds")

    lines = [
        f"# Test Report - {timestamp}",
        "",
        "## Summary",
        "",
        f"- **Total collected**: {total_collected}",
        f"- **Passed**: {num_passed}",
        f"- **Failed**: {num_failed}",
        f"- **Errors**: {num_error}",
        f"- **Skipped**: {num_skipped}",
        f"- **Expected failures (xfailed)**: {num_xfailed}",
        f"- **Unexpected passes (xpassed)**: {num_xpassed}",
        f"- **Exit status**: {exitstatus}",
        "",
        "## By Test",
        "",
    ]

    for outcome in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"):
        reports = stats.get(outcome, [])
        if not reports:
            continue
        lines.append(f"### {outcome.capitalize()} ({len(reports)})")
        lines.append("")
        for rep in reports:
            # nodeid is the test identifier (module::test_name[param]...)
            lines.append(f"- **{rep.nodeid}**")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


