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

    # Specifically verify View 6 content under Synthetic Demo Mode
    at.sidebar.radio[0].set_value("6. Semiconductor Wafer Etch (v0.2 Benchmark)").run()
    assert not at.exception
    metric_values = [m.value for m in at.metric]
    assert any("0.9729" in v for v in metric_values), "Expected O1 PR-AUC in Wafer view metrics"
    info_texts = [info.value for info in at.info]
    assert any("synthetic wafer demo traces" in t for t in info_texts), (
        "Synthetic mode must explicitly label synthetic wafer demo traces"
    )


def test_dashboard_wafer_view_real_data_content_and_scorecard() -> None:
    dashboard_path = Path("app/dashboard.py").resolve().as_posix()
    at = AppTest.from_file(dashboard_path, default_timeout=30)
    at.run()
    assert not at.exception

    # Explicitly check if Real Dataset mode option exists and select it
    select_box = at.sidebar.selectbox[0]
    if "Real Dataset (SECOM)" in select_box.options:
        select_box.select("Real Dataset (SECOM)").run()
        assert not at.exception

    # Select Wafer view
    at.sidebar.radio[0].set_value("6. Semiconductor Wafer Etch (v0.2 Benchmark)").run()
    assert not at.exception

    # If real wafer parquet exists, verify that synthetic info warning is NOT rendered
    if Path("data/processed/wafer_runs.parquet").exists() and "Real Dataset (SECOM)" in select_box.options:
        info_texts = [info.value for info in at.info]
        assert not any("synthetic wafer demo traces" in t for t in info_texts), (
            "Real wafer parquet path should not display synthetic wafer banner"
        )

    # Metric labels and values must reflect confirmatory scorecard
    metric_labels = [m.label for m in at.metric]
    metric_values = [m.value for m in at.metric]
    metric_deltas = [m.delta for m in at.metric]

    assert any("O1: PR-AUC" in lbl for lbl in metric_labels)
    assert any("0.9729" in val for val in metric_values)
    # Deltas should be dynamic verdict strings without duplicated "PASS (PASS)"
    assert any(d == "PASS" for d in metric_deltas)
    assert any("PASS (344 TP, 4 FP)" in (d or "") for d in metric_deltas)
    assert any(d == "REPORTED" for d in metric_deltas)


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
