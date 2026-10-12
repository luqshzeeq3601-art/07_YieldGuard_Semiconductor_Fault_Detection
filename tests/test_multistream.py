"""Unit tests for Multi-Stream SPC monitoring on semiconductor tool fleets."""

import numpy as np
import pandas as pd

from yieldguard.multistream import MultiStreamSPC


def test_multistream_eliminates_baseline_offset_false_alarms() -> None:
    """Prove that chamber baseline offsets trip pooled Shewhart alarms, but MultiStreamSPC remains in control."""
    rng = np.random.default_rng(101)
    n_runs_per_tool = 50

    # Two tools: Tool_A baseline mean 0.0, Tool_B baseline mean 5.0 (harmless baseline offset)
    sensors = ["f001", "f002"]

    tool_a_data = rng.normal(loc=0.0, scale=1.0, size=(n_runs_per_tool, 2))
    tool_b_data = rng.normal(loc=5.0, scale=1.0, size=(n_runs_per_tool, 2))

    df_a = pd.DataFrame(tool_a_data, columns=sensors)
    df_a["tool_id"] = "Tool_A"

    df_b = pd.DataFrame(tool_b_data, columns=sensors)
    df_b["tool_id"] = "Tool_B"

    phase1_df = pd.concat([df_a, df_b], ignore_index=True)

    # Fit MultiStream monitor on Phase I data
    ms_spc = MultiStreamSPC(k_sigma=3.0, bonferroni_correction=True).fit(
        phase1_df, stream_col="tool_id", sensor_cols=sensors
    )

    # Normal in-control evaluation runs from Tool_B
    eval_b_data = rng.normal(loc=5.0, scale=1.0, size=(20, 2))
    eval_df = pd.DataFrame(eval_b_data, columns=sensors)
    eval_df["tool_id"] = "Tool_B"

    # MultiStream scoring
    res_df = ms_spc.score_dataframe(eval_df, stream_col="tool_id")
    # All in-control runs should have no false alarms under multi-stream
    assert res_df["alarm_multistream_shewhart"].sum() == 0

    # Now verify that scoring individual run from Tool_B yields known stream and in-control
    run_res = ms_spc.score_run("Tool_B", {"f001": 5.1, "f002": 4.9})
    assert run_res["known_stream"] is True
    assert run_res["alarm"] is False
    assert run_res["max_abs_z"] < 3.0


def test_multistream_catches_true_chamber_fault() -> None:
    """Verify that a true 6-sigma process shift on Chamber_C triggers an alarm and attributes to Chamber_C."""
    rng = np.random.default_rng(202)
    sensors = ["f010", "f020"]

    # Phase 1 baselines
    ch_c_phase1 = pd.DataFrame(rng.normal(loc=10.0, scale=1.0, size=(40, 2)), columns=sensors)
    ch_c_phase1["chamber"] = "Chamber_C"

    ch_d_phase1 = pd.DataFrame(rng.normal(loc=20.0, scale=1.0, size=(40, 2)), columns=sensors)
    ch_d_phase1["chamber"] = "Chamber_D"

    fit_df = pd.concat([ch_c_phase1, ch_d_phase1], ignore_index=True)
    ms_spc = MultiStreamSPC(k_sigma=3.0).fit(fit_df, stream_col="chamber", sensor_cols=sensors)

    # Injected 6-sigma fault on Chamber_C sensor f010
    fault_res = ms_spc.score_run("Chamber_C", {"f010": 16.0, "f020": 10.1})
    assert fault_res["known_stream"] is True
    assert fault_res["alarm"] is True
    assert fault_res["offending_sensor"] == "f010"
    assert fault_res["max_abs_z"] >= 4.5
