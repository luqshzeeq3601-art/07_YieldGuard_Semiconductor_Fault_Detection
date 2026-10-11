"""Tests for Streamlit dashboard views using streamlit.testing.v1.AppTest."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from yieldguard.artifacts import load_model_bundle


def test_dashboard_apptest_runs_all_views_without_exceptions() -> None:
    dashboard_path = Path("app/dashboard.py").resolve().as_posix()

    at = AppTest.from_file(dashboard_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Dashboard threw unexpected exception on load: {at.exception}"

    # Verify synthetic demo mode banner is present when selecting Synthetic mode
    at.sidebar.selectbox[0].select("Synthetic Demo Data").run()
    assert not at.exception
    warning_texts = [w.value for w in at.warning]
    assert any("SYNTHETIC DEMO MODE" in t for t in warning_texts), (
        "Synthetic mode must be labelled on screen"
    )

    views = [
        "1. Univariate EWMA & Run Rules",
        "2. Multivariate SPC (T² & SPE)",
        "3. Flagged-Run Drill-Down",
        "4. Alarm-Budget Performance",
        "5. Project Overview & Architecture",
        "6. Semiconductor Wafer Etch (v0.2 Benchmark)",
    ]

    for view_name in views:
        at.sidebar.radio[0].set_value(view_name).run()
        assert not at.exception, f"View '{view_name}' threw exception: {at.exception}"


def test_dashboard_chart_values_match_package_outputs() -> None:
    bundle = load_model_bundle("models/v0.1.0")
    sensor = bundle.spc_model.monitored_sensors[0]
    expected_mu0 = bundle.spc_model.sensor_specs[sensor].mu0

    dashboard_path = Path("app/dashboard.py").resolve().as_posix()
    at = AppTest.from_file(dashboard_path, default_timeout=30)
    at.run()
    assert not at.exception

    # Navigate to EWMA view and select the first monitored sensor
    at.sidebar.radio[0].set_value("1. Univariate EWMA & Run Rules").run()
    assert not at.exception

    # Metric for Phase I centre line should match expected_mu0
    metrics = [m.value for m in at.metric]
    mu0_str = f"{expected_mu0:.4f}"
    assert any(mu0_str in m for m in metrics), (
        f"Expected μ₀ ({mu0_str}) to match dashboard metric output"
    )
