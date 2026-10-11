"""YieldGuard: Semiconductor Process Control & Fault Detection Streamlit Dashboard.

Four views per docs/04_TECHNICAL_DESIGN.md section 8:
1. EWMA chart per monitored sensor with limits and Western Electric run-rule markers.
2. T-squared and SPE over run order with theoretical (99%) and budget thresholds.
3. Flagged-run drill-down: MSPC contributions and XGBoost TreeSHAP side by side with group members.
4. Alarm-budget table: alarms, false alarms, and recall at 2, 5, and 10 alarms per 100 runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from yieldguard.artifacts import LoadedBundle, load_model_bundle
from yieldguard.data import SENSOR_COLUMNS
from yieldguard.explain import SMEARING_CAVEAT, rank_sensors_for_run
from yieldguard.predict import score_runs_dataframe
from yieldguard.spc import compute_ewma_series, evaluate_run_rules
from yieldguard.synthetic import generate_synthetic_secom


def check_real_data_available(bundle_dir: str = "models/v0.1.0") -> bool:
    b_path = Path(bundle_dir)
    runs_path = Path("data/processed/runs.parquet")
    return b_path.exists() and (b_path / "metadata.json").exists() and runs_path.exists()


@st.cache_resource
def get_cached_bundle(bundle_dir: str = "models/v0.1.0") -> LoadedBundle | None:
    try:
        p = Path(bundle_dir)
        if p.exists() and (p / "metadata.json").exists():
            return load_model_bundle(p)
    except Exception:
        pass
    return None


@st.cache_data
def load_data_and_scores(
    use_synthetic: bool = False,
    bundle_dir: str = "models/v0.1.0",
) -> tuple[pd.DataFrame, pd.DataFrame | None, LoadedBundle, bool]:
    """Load or generate dataset, bundle, and scored runs dataframe."""
    bundle = get_cached_bundle(bundle_dir)

    if use_synthetic or bundle is None or not Path("data/processed/runs.parquet").exists():
        # Synthetic Demo Data Mode
        matrix, labels_raw, timestamps, _ = generate_synthetic_secom(n_runs=100, seed=42)
        runs_df = pd.DataFrame(matrix, columns=SENSOR_COLUMNS)
        runs_df.insert(0, "run_id", [f"syn_{i:03d}" for i in range(100)])
        runs_df.insert(1, "timestamp", timestamps)
        runs_df["block"] = "synthetic_demo"

        labels_df = pd.DataFrame(
            {
                "run_id": runs_df["run_id"],
                "label_fail": [1 if x == 1 else 0 for x in labels_raw],
            }
        )

        if bundle is None:
            # Fallback to staging bundle if v0.1.0 isn't loaded yet
            bundle = get_cached_bundle("models/staging")

        if bundle is not None:
            scored_df = score_runs_dataframe(bundle, runs_df)
            extra_cols = [c for c in runs_df.columns if c not in scored_df.columns]
            scored_df = pd.concat([scored_df, runs_df[extra_cols]], axis=1)
        else:
            scored_df = runs_df.copy()

        return scored_df, labels_df, bundle, True

    # Real Dataset Mode
    runs_df = pd.read_parquet("data/processed/runs.parquet")
    scored_df = score_runs_dataframe(bundle, runs_df)
    extra_cols = [c for c in runs_df.columns if c not in scored_df.columns]
    scored_df = pd.concat([scored_df, runs_df[extra_cols]], axis=1)

    # Load labels
    labels_dev_p = Path("data/processed/labels_dev.parquet")
    labels_hold_p = Path("data/processed/labels_holdout.parquet")
    holdout_metrics_p = Path("reports/holdout_metrics.json")

    labels_dfs = []
    if labels_dev_p.exists():
        labels_dfs.append(pd.read_parquet(labels_dev_p))
    if holdout_metrics_p.exists() and labels_hold_p.exists():
        # Only show holdout labels once final evaluation exists
        labels_dfs.append(pd.read_parquet(labels_hold_p))

    labels_df = pd.concat(labels_dfs, ignore_index=True) if labels_dfs else None
    return scored_df, labels_df, bundle, False


def render_ewma_view(
    scored_df: pd.DataFrame,
    bundle: LoadedBundle,
) -> None:
    st.subheader("1. Univariate EWMA Chart & Western Electric Run Rules")
    st.markdown(
        "Phase II tracking on monitored sensors chosen by Phase I rank-biserial separation. "
        "Standardized EWMA chart ($\\lambda=0.20, L=3$) with control limits and 4 Western Electric run rules."
    )

    monitored = bundle.spc_model.monitored_sensors
    if not monitored:
        st.info("No monitored sensors available in bundle.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_sensor = st.selectbox("Select Monitored Sensor", monitored, index=0)
        spec = bundle.spc_model.sensor_specs[selected_sensor]
        st.metric("Phase I Centre Line (μ₀)", f"{spec.mu0:.4f}")
        st.metric("Phase I Moving-Range Sigma (σ₀)", f"{spec.sigma0:.4f}")
        st.write(f"**Rank-biserial separation:** {spec.rank_biserial:.4f}")
        st.write(f"**BH adjusted p-value:** {spec.bh_adjusted_p_value:.4e}")
        if spec.group_members:
            st.caption(f"Collinear group members: {', '.join(spec.group_members)}")

    # Extract series and compute EWMA
    sensor_vals = (
        scored_df[selected_sensor].to_numpy(dtype=np.float64)
        if selected_sensor in scored_df.columns
        else np.zeros(len(scored_df))
    )
    ewma_res = compute_ewma_series(
        sensor_vals,
        spec.mu0,
        spec.sigma0,
        lam=bundle.spc_model.ewma_lambda,
        L=bundle.spc_model.ewma_l,
    )
    rules_res = evaluate_run_rules(sensor_vals, spec.mu0, spec.sigma0)

    n_runs = len(sensor_vals)
    run_indices = np.arange(n_runs)

    fig = go.Figure()
    # Centre Line & Limits
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=ewma_res["z"],
            mode="lines",
            name="EWMA (z_t)",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=ewma_res["ucl"],
            mode="lines",
            name="Upper Limit (UCL)",
            line=dict(color="#d62728", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=ewma_res["lcl"],
            mode="lines",
            name="Lower Limit (LCL)",
            line=dict(color="#d62728", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=[spec.mu0] * n_runs,
            mode="lines",
            name="Centre Line (μ₀)",
            line=dict(color="#7f7f7f", dash="dot"),
        )
    )

    # Rule violation markers
    r1_idx = np.flatnonzero(rules_res["rule_1_beyond_3sigma"])
    if len(r1_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=r1_idx,
                y=ewma_res["z"][r1_idx],
                mode="markers",
                name="Rule 1 (> 3σ)",
                marker=dict(color="red", symbol="x", size=10),
            )
        )

    r2_idx = np.flatnonzero(rules_res["rule_2_of_3_beyond_2sigma"])
    if len(r2_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=r2_idx,
                y=ewma_res["z"][r2_idx],
                mode="markers",
                name="Rule 2 (2 of 3 > 2σ)",
                marker=dict(color="orange", symbol="triangle-up", size=8),
            )
        )

    fig.update_layout(
        title=f"EWMA Chart for {selected_sensor}",
        xaxis_title="Run Order",
        yaxis_title="Sensor Value",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    with col2:
        st.plotly_chart(fig, use_container_width=True)


def render_mspc_view(
    scored_df: pd.DataFrame,
    bundle: LoadedBundle,
) -> None:
    st.subheader("2. Multivariate SPC: Hotelling T² & Squared Prediction Error (SPE)")
    st.markdown(
        "Multivariate process monitoring across 50 principal components. "
        "Horizontal reference lines display theoretical 99% limits and empirical 5-alarm-per-100 budget thresholds."
    )

    n_runs = len(scored_df)
    run_indices = np.arange(n_runs)

    t2_vals = scored_df["t2"].to_numpy()
    spe_vals = scored_df["spe"].to_numpy()

    t2_lim_theo = bundle.mspc_model.t2_limit_99
    spe_lim_theo = bundle.mspc_model.spe_limit_99
    t2_thresh_b5 = float(bundle.policy["thresholds"]["E04"]["budget_5"])
    spe_thresh_b5 = float(bundle.policy["thresholds"]["E05"]["budget_5"])

    col1, col2 = st.columns(2)
    with col1:
        fig_t2 = go.Figure()
        fig_t2.add_trace(
            go.Scatter(
                x=run_indices,
                y=t2_vals,
                mode="lines",
                name="T² Statistic",
                line=dict(color="#2ca02c", width=1.5),
            )
        )
        fig_t2.add_trace(
            go.Scatter(
                x=[0, n_runs],
                y=[t2_lim_theo, t2_lim_theo],
                mode="lines",
                name="Theoretical 99% Limit",
                line=dict(color="#d62728", dash="dash"),
            )
        )
        fig_t2.add_trace(
            go.Scatter(
                x=[0, n_runs],
                y=[t2_thresh_b5, t2_thresh_b5],
                mode="lines",
                name="Budget 5 Threshold",
                line=dict(color="#ff7f0e", dash="dot"),
            )
        )
        fig_t2.update_layout(
            title="Hotelling T² over Run Order",
            xaxis_title="Run Order",
            yaxis_title="T²",
            hovermode="x unified",
        )
        st.plotly_chart(fig_t2, use_container_width=True)
        st.metric("T² Budget 5 Alarms", int(np.sum(t2_vals > t2_thresh_b5)))

    with col2:
        fig_spe = go.Figure()
        fig_spe.add_trace(
            go.Scatter(
                x=run_indices,
                y=spe_vals,
                mode="lines",
                name="SPE Statistic",
                line=dict(color="#9467bd", width=1.5),
            )
        )
        fig_spe.add_trace(
            go.Scatter(
                x=[0, n_runs],
                y=[spe_lim_theo, spe_lim_theo],
                mode="lines",
                name="Theoretical 99% Limit",
                line=dict(color="#d62728", dash="dash"),
            )
        )
        fig_spe.add_trace(
            go.Scatter(
                x=[0, n_runs],
                y=[spe_thresh_b5, spe_thresh_b5],
                mode="lines",
                name="Budget 5 Threshold",
                line=dict(color="#ff7f0e", dash="dot"),
            )
        )
        fig_spe.update_layout(
            title="Squared Prediction Error (SPE) over Run Order",
            xaxis_title="Run Order",
            yaxis_title="SPE",
            hovermode="x unified",
        )
        st.plotly_chart(fig_spe, use_container_width=True)
        st.metric("SPE Budget 5 Alarms", int(np.sum(spe_vals > spe_thresh_b5)))


def render_drilldown_view(
    scored_df: pd.DataFrame,
    bundle: LoadedBundle,
) -> None:
    st.subheader("3. Flagged-Run Drill-Down: Sensor Rankings")
    st.markdown(
        "Side-by-side comparison of MSPC contributions and XGBoost TreeSHAP attributions for flagged runs. "
        "Collinear sensor groups are displayed to prevent misinterpreting smeared representative scores."
    )

    # Find flagged runs by primary detector or any budget 5 threshold
    alarm_mask = (
        scored_df["alarm"] if "alarm" in scored_df.columns else np.zeros(len(scored_df), dtype=bool)
    )
    flagged_df = scored_df[alarm_mask]
    if len(flagged_df) == 0:
        flagged_df = scored_df.head(10)
        st.info("No runs flagged by primary alarm threshold; showing first runs for inspection.")

    run_options = list(flagged_df["run_id"].astype(str))
    selected_run_id = st.selectbox(
        "Select Run for Sensor Attribution Drill-Down", run_options, index=0
    )

    run_row = scored_df[scored_df["run_id"].astype(str) == selected_run_id].iloc[0]
    explanation = rank_sensors_for_run(
        run_row,
        quality=bundle.quality,
        mspc_model=bundle.mspc_model,
        xgb_model_or_booster=bundle.xgb_model,
        top_k=5,
    )

    st.caption(f"**Caution on root cause:** {SMEARING_CAVEAT}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top MSPC Contributing Sensors")
        mspc_sensors = [x["sensor"] for x in explanation["mspc_top_sensors"]]
        mspc_vals = [x["contribution"] for x in explanation["mspc_top_sensors"]]
        fig_m = go.Figure(
            go.Bar(x=mspc_vals, y=mspc_sensors, orientation="h", marker=dict(color="#1f77b4"))
        )
        fig_m.update_layout(
            xaxis_title="Combined Contribution",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=20, b=20),
        )
        st.plotly_chart(fig_m, use_container_width=True)
        for item in explanation["mspc_top_sensors"]:
            grp = f" (Group: {', '.join(item['group_members'])})" if item["group_members"] else ""
            st.write(
                f"- **Rank {item['rank']} — {item['sensor']}**: {item['contribution']:.3f}{grp}"
            )

    with col2:
        st.markdown("#### Top XGBoost SHAP Attributions (Raw Margin)")
        shap_sensors = [x["sensor"] for x in explanation["xgb_shap_top_sensors"]]
        shap_vals = [x["shap_value"] for x in explanation["xgb_shap_top_sensors"]]
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in shap_vals]
        fig_s = go.Figure(
            go.Bar(x=shap_vals, y=shap_sensors, orientation="h", marker=dict(color=colors))
        )
        fig_s.update_layout(
            xaxis_title="SHAP Value (log-odds)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=20, b=20),
        )
        st.plotly_chart(fig_s, use_container_width=True)
        for item in explanation["xgb_shap_top_sensors"]:
            grp = f" (Group: {', '.join(item['group_members'])})" if item["group_members"] else ""
            st.write(
                f"- **Rank {item['rank']} — {item['sensor']}**: {item['shap_value']:+.3f} ({item['direction']}){grp}"
            )

    overlap = explanation["overlap"]["top_k_common_sensors"]
    st.write(f"**Direct Overlap Sensors:** {', '.join(overlap) if overlap else 'None'}")


def render_alarm_budget_view(
    scored_df: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    bundle: LoadedBundle,
) -> None:
    st.subheader("4. Alarm-Budget Performance")
    st.markdown(
        "Operating performance at fixed review capacity budgets: 2, 5 (primary), and 10 alarms per 100 runs. "
        "Thresholds were calibrated on development block D5 and applied out-of-sample."
    )

    n_runs = len(scored_df)
    has_labels = labels_df is not None and "label_fail" in labels_df.columns
    if has_labels:
        merged = pd.merge(scored_df, labels_df[["run_id", "label_fail"]], on="run_id", how="inner")
        y_true = merged["label_fail"].to_numpy()
        n_fails = int(np.sum(y_true == 1))
    else:
        merged = scored_df
        y_true = None
        n_fails = None

    rows: list[dict[str, Any]] = []
    detectors = ["E02", "E03", "E04", "E05", "E06", "E07", "E08"]
    thresholds = bundle.policy.get("thresholds", {})

    for det in detectors:
        for b in (2, 5, 10):
            tau = thresholds.get(det, {}).get(f"budget_{b}")
            if tau is None:
                continue
            flag_col = f"alarm_{det.lower()}_b{b}"
            alarms_count = int(scored_df[flag_col].sum()) if flag_col in scored_df.columns else 0
            realised_rate = float(alarms_count / n_runs * 100.0)

            row: dict[str, Any] = {
                "Detector": det,
                "Budget": f"{b} per 100",
                "Threshold": round(float(tau), 4),
                "Total Alarms": alarms_count,
                "Realised Rate (/100)": round(realised_rate, 2),
            }

            if has_labels and y_true is not None:
                flags = merged[flag_col].to_numpy()
                true_alarms = int(np.sum((flags == 1) & (y_true == 1)))
                false_alarms = int(np.sum((flags == 1) & (y_true == 0)))
                recall = float(true_alarms / n_fails) if n_fails > 0 else 0.0
                row["True Alarms"] = true_alarms
                row["False Alarms"] = false_alarms
                row["Recall"] = round(recall, 3)

            rows.append(row)

    df_table = pd.DataFrame(rows)
    st.dataframe(df_table, use_container_width=True)

    # Show holdout metrics summary if available
    holdout_metrics_p = Path("reports/holdout_metrics.json")
    if holdout_metrics_p.exists():
        st.markdown("### Official Frozen Holdout Evaluation Results")
        h_data = json.loads(holdout_metrics_p.read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("Holdout Runs", h_data["n_holdout_runs"])
        cols[1].metric("Holdout Fails", h_data["n_holdout_fails"])
        cols[2].metric("Holdout Prevalence", f"{h_data['holdout_fail_prevalence'] * 100:.2f}%")
        cols[3].metric("Primary Detector", h_data["primary_detector"])

        st.json(h_data.get("objectives", {}))


def main() -> None:
    st.set_page_config(
        page_title="YieldGuard — Semiconductor Process Control",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🔬 YieldGuard")
    st.sidebar.markdown(
        "**Semiconductor Process Control & Fault Detection**\n\n"
        "Decision-support dashboard for yield engineers."
    )

    real_available = check_real_data_available()
    mode_options = (
        ["Real Dataset (SECOM)", "Synthetic Demo Data"]
        if real_available
        else ["Synthetic Demo Data"]
    )
    selected_mode = st.sidebar.selectbox("Data Mode", mode_options, index=0)
    is_synthetic = (selected_mode == "Synthetic Demo Data") or not real_available

    if is_synthetic:
        st.warning("⚠️ Running in SYNTHETIC DEMO MODE (Synthetic SECOM-like generated data).")

    bundle_dir = "models/v0.1.0" if Path("models/v0.1.0").exists() else "models/staging"
    scored_df, labels_df, bundle, synthetic_active = load_data_and_scores(
        use_synthetic=is_synthetic,
        bundle_dir=bundle_dir,
    )

    if bundle is None:
        st.error(
            "No valid model bundle found in models/v0.1.0 or models/staging. Run training first."
        )
        return

    # View navigation
    views = [
        "1. Univariate EWMA & Run Rules",
        "2. Multivariate SPC (T² & SPE)",
        "3. Flagged-Run Drill-Down",
        "4. Alarm-Budget Performance",
    ]
    active_view = st.sidebar.radio("Dashboard View", views, index=0)

    if active_view == "1. Univariate EWMA & Run Rules":
        render_ewma_view(scored_df, bundle)
    elif active_view == "2. Multivariate SPC (T² & SPE)":
        render_mspc_view(scored_df, bundle)
    elif active_view == "3. Flagged-Run Drill-Down":
        render_drilldown_view(scored_df, bundle)
    elif active_view == "4. Alarm-Budget Performance":
        render_alarm_budget_view(scored_df, labels_df, bundle)


if __name__ == "__main__":
    main()
