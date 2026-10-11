"""YieldGuard: Semiconductor Process Control & Fault Detection Streamlit Dashboard.

Five operational and educational views (satisfying docs/04_TECHNICAL_DESIGN.md section 8):
1. Univariate EWMA & Run Rules: Phase II chart with time-varying limits and 4 Western Electric rules.
2. Multivariate SPC (T² & SPE): Hotelling T-squared and Q-residual over run order with budget thresholds.
3. Flagged-Run Drill-Down: Side-by-side MSPC contributions and XGBoost TreeSHAP with collinear group maps.
4. Alarm-Budget Performance: Operating metrics across budgets 2, 5, 10, holdout scorecard, and drift summary.
5. Project Overview & Architecture: End-to-end guide, chronological split protocol, and detector glossary.
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

DETECTOR_CATALOG: dict[str, dict[str, str]] = {
    "E01": {
        "name": "E01 — Chance Baseline",
        "short": "Chance Baseline",
        "family": "Baseline",
        "desc": "Constant zero score reference reflecting base failure prevalence.",
    },
    "E02": {
        "name": "E02 — Univariate SPC (Individuals) ★ Primary",
        "short": "Univariate SPC (Individuals)",
        "family": "Univariate SPC",
        "desc": "Max standardized |z|-score across 10 Phase I screened sensors using moving-range sigma.",
    },
    "E03": {
        "name": "E03 — Univariate SPC (EWMA λ=0.20)",
        "short": "Univariate EWMA",
        "family": "Univariate SPC",
        "desc": "Exponentially Weighted Moving Average tracking persistent step shifts across 10 sensors.",
    },
    "E04": {
        "name": "E04 — Multivariate Hotelling T²",
        "short": "Hotelling T² (PCA)",
        "family": "Multivariate SPC",
        "desc": "Mahalanobis distance inside the 50-component PCA subspace fitted on passing runs.",
    },
    "E05": {
        "name": "E05 — Squared Prediction Error (SPE / Q)",
        "short": "SPE / Q-Residual",
        "family": "Multivariate SPC",
        "desc": "Residual distance orthogonal to the 50-component PCA subspace (correlation breakdown).",
    },
    "E06": {
        "name": "E06 — Combined MSPC Index",
        "short": "Combined MSPC (T² + SPE)",
        "family": "Multivariate SPC",
        "desc": "Maximum normalized excursion across Hotelling T² and SPE statistics.",
    },
    "E07": {
        "name": "E07 — Isolation Forest",
        "short": "Isolation Forest",
        "family": "Unsupervised ML",
        "desc": "200-tree ensemble fitted strictly on passing training runs across 274 retained sensors.",
    },
    "E08": {
        "name": "E08 — Class-Weighted XGBoost",
        "short": "XGBoost Classifier",
        "family": "Supervised ML",
        "desc": "Shallow gradient boosted trees with dynamic scale_pos_weight tuned via inner-fold 1-SE rule.",
    },
}


def inject_custom_css() -> None:
    """Inject clean, accessible operational dashboard styles without generic AI gradients."""
    st.markdown(
        """
        <style>
        .yg-header {
            padding: 1rem 1.25rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-left: 5px solid #0f766e;
            border-radius: 6px;
            margin-bottom: 1rem;
            background-color: rgba(15, 118, 110, 0.04);
        }
        .yg-header h1 {
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            margin: 0 0 0.35rem 0 !important;
            padding: 0 !important;
        }
        .yg-header p {
            margin: 0 !important;
            font-size: 0.95rem;
            opacity: 0.88;
        }
        .yg-badge {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 4px;
            margin-right: 0.4rem;
            font-family: monospace;
            letter-spacing: 0.02em;
        }
        .yg-badge-teal {
            background-color: rgba(15, 118, 110, 0.15);
            color: #0f766e;
            border: 1px solid rgba(15, 118, 110, 0.35);
        }
        .yg-badge-amber {
            background-color: rgba(180, 83, 9, 0.15);
            color: #b45309;
            border: 1px solid rgba(180, 83, 9, 0.35);
        }
        .yg-badge-slate {
            background-color: rgba(100, 116, 139, 0.15);
            color: #475569;
            border: 1px solid rgba(100, 116, 139, 0.35);
        }
        .yg-card {
            padding: 0.9rem 1.1rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 6px;
            margin-bottom: 0.85rem;
            background-color: rgba(128, 128, 128, 0.03);
        }
        .yg-card h4 {
            margin: 0 0 0.4rem 0 !important;
            font-size: 1.0rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-variant-numeric: tabular-nums;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def render_top_header_and_kpis(
    scored_df: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    bundle: LoadedBundle,
    is_synthetic: bool,
) -> None:
    """Render top project banner, plain-English onboarding guide, and executive KPI strip."""
    mode_badge = (
        '<span class="yg-badge yg-badge-amber">SYNTHETIC DEMO DATA (100 RUNS)</span>'
        if is_synthetic
        else '<span class="yg-badge yg-badge-teal">UCI SECOM BENCHMARK (1,567 RUNS)</span>'
    )
    version_badge = f'<span class="yg-badge yg-badge-slate">BUNDLE: {bundle.version}</span>'
    primary_id = bundle.policy.get("primary_detector", "E02")
    primary_badge = (
        f'<span class="yg-badge yg-badge-teal">PRIMARY DETECTOR: {primary_id} (SPC INDIVIDUALS)</span>'
    )

    st.markdown(
        f"""
        <div class="yg-header">
            <div style="margin-bottom: 0.45rem;">{mode_badge}{version_badge}{primary_badge}</div>
            <h1>🔬 YieldGuard — Semiconductor Process Control & Fault Detection</h1>
            <p>
                Decision-support console for wafer fab yield engineers. Monitors 590 metrology & process sensor channels
                under a strict <strong>5 false-alarms per 100 passing runs</strong> review capacity budget, comparing classical
                Statistical Process Control (SPC) against machine learning under a leak-free chronological protocol.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📖 New to YieldGuard? Click here for a Plain-English Project Guide & How to Read This Dashboard"):
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.markdown("#### 1. What Problem Does This Solve?")
            st.markdown(
                "- In semiconductor wafer manufacturing, **590 sensor channels** record chamber conditions after each process step.\n"
                "- Only **~6.6% of wafer runs fail** final electrical test (`label_fail = 1`).\n"
                "- Standard textbook $3\\sigma$ control charts trigger alarms on **>65% of normal runs**, causing severe engineer alarm fatigue.\n"
                "- **YieldGuard** calibrates detectors to a strict **Alarm Budget** (e.g., **5 alarms per 100 passing runs**) so engineers only triage the highest-risk wafers."
            )
        with g_col2:
            st.markdown("#### 2. How Was the Model Selected?")
            st.markdown(
                "- **Zero Data Leakage**: Wafers are split strictly by timestamp into 5 development blocks (`D1`–`D5`, 1,175 runs) and an unseen `Holdout` set (392 runs).\n"
                "- **Train-Only Cleaning**: Out of 590 raw sensors, 24 high-missing, 122 constant, and 170 collinear duplicate channels are pruned on training runs only $\\rightarrow$ **274 retained features**.\n"
                "- **Parsimony Rule (Rule 4)**: Tuned XGBoost (`E08`) beat Univariate SPC (`E02`) by $+0.0638$ PR-AUC in development, but the 95% confidence interval $[-0.090, +0.216]$ spanned zero. Per predeclared protocol, the simpler, transparent **`E02` Univariate SPC** was frozen as the primary detector."
            )
        with g_col3:
            st.markdown("#### 3. How to Navigate the 5 Views")
            st.markdown(
                "1. **Univariate EWMA & Run Rules**: Inspect individual key sensors with smoothed trends and Western Electric alarm markers.\n"
                "2. **Multivariate SPC ($T^2$ & $SPE$)**: Track global 50-component PCA correlation shifts and sensor breakdown spikes.\n"
                "3. **Flagged-Run Drill-Down**: Triage any flagged wafer run with side-by-side MSPC physical contributions and XGBoost SHAP values.\n"
                "4. **Alarm-Budget Performance**: Compare all 8 detectors (`E01`–`E08`) across budgets and inspect the one-shot holdout audit.\n"
                "5. **Project Overview & Architecture**: Review the full system design, data hygiene rules, and drift findings."
            )

    # Top-level KPI Summary Strip
    n_runs = len(scored_df)
    n_flagged = int(scored_df["alarm"].sum()) if "alarm" in scored_df.columns else 0
    flagged_rate = (n_flagged / n_runs * 100.0) if n_runs > 0 else 0.0
    tau_b5 = float(bundle.policy.get("thresholds", {}).get(primary_id, {}).get("budget_5", 6.2822))

    if labels_df is not None and "label_fail" in labels_df.columns:
        merged_kpi = pd.merge(scored_df[["run_id", "alarm"]], labels_df[["run_id", "label_fail"]], on="run_id", how="inner")
        n_fails = int((merged_kpi["label_fail"] == 1).sum())
        fail_prev = (n_fails / len(merged_kpi) * 100.0) if len(merged_kpi) > 0 else 0.0
        fail_str = f"{n_fails} fails ({fail_prev:.2f}%)"
    else:
        fail_str = "Unlabeled cohort"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Evaluated Wafer Runs",
        f"{n_runs:,} runs",
        delta=fail_str,
        delta_color="off",
        help="Total wafer runs loaded in the active cohort and their ground-truth electrical test failure rate.",
    )
    k2.metric(
        "Retained / Monitored Sensors",
        f"{len(bundle.quality.retained_columns)} / {len(bundle.spc_model.monitored_sensors)}",
        delta="from 590 raw channels",
        delta_color="off",
        help="274 sensors retained after train-only cleaning; top 10 sensors monitored by Phase I Univariate SPC.",
    )
    k3.metric(
        "Primary Alarm Budget (E02)",
        "5.0 / 100 runs",
        delta=f"Cutoff τ = {tau_b5:.4f}",
        delta_color="off",
        help="Target maximum false-alarm rate on passing runs, calibrated on validation block D5.",
    )
    k4.metric(
        "Flagged Runs for Triage",
        f"{n_flagged} runs",
        delta=f"{flagged_rate:.2f} alarms / 100 runs",
        delta_color="off",
        help="Wafer runs exceeding the frozen primary detector budget-5 threshold.",
    )
    st.divider()


def render_ewma_view(
    scored_df: pd.DataFrame,
    bundle: LoadedBundle,
) -> None:
    st.subheader("1. Univariate EWMA Chart & Western Electric Run Rules")
    st.markdown(
        "**Purpose:** Track individual high-impact process sensors over time. "
        "During Phase I training (`D1`–`D4`), all 274 cleaned sensors were screened using **rank-biserial correlation** "
        "between passing and failing wafers. The top 10 most discriminative sensors are monitored below using an "
        "**Exponentially Weighted Moving Average ($\\lambda=0.20, L=3.0$)** and **4 Western Electric run rules**."
    )

    monitored = bundle.spc_model.monitored_sensors
    if not monitored:
        st.info("No monitored sensors available in bundle.")
        return

    col1, col2 = st.columns([1, 2.3])
    with col1:
        def _format_sensor(s_name: str) -> str:
            idx = monitored.index(s_name) + 1
            sp = bundle.spc_model.sensor_specs[s_name]
            return f"Rank #{idx} — Sensor {s_name} (|r_rb|={abs(sp.rank_biserial):.3f})"

        selected_sensor = st.selectbox(
            "Select Monitored Sensor (Ranked by Fault Separation)",
            monitored,
            index=0,
            format_func=_format_sensor,
            help="Sensors are ordered by how strongly they separated failing vs passing runs in Phase I training.",
        )
        spec = bundle.spc_model.sensor_specs[selected_sensor]

        show_raw_trace = st.checkbox(
            "Overlay raw unsmoothed sensor readings",
            value=False,
            help="Toggle raw run-by-run values behind the smoothed EWMA trend line.",
        )

        # Required metrics (keep exact format for automated test verification)
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric(
                "Phase I Centre Line (μ₀)",
                f"{spec.mu0:.4f}",
                help="Mean of passing training runs in Phase I after 3-sigma outlier trimming.",
            )
        with m_c2:
            st.metric(
                "Phase I Moving-Range Sigma (σ₀)",
                f"{spec.sigma0:.4f}",
                help="Within-run standard deviation estimated from average two-point moving range (MR / 1.128).",
            )

        st.markdown(
            f"""
            <div class="yg-card">
                <h4>📊 Phase I Screening Diagnostics ({selected_sensor})</h4>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
                    <li><strong>Rank-Biserial Correlation ($r_{{rb}}$):</strong> <code>{spec.rank_biserial:+.4f}</code></li>
                    <li><strong>Benjamini-Hochberg FDR $p$-value:</strong> <code>{spec.bh_adjusted_p_value:.4e}</code></li>
                    <li><strong>Training Missingness:</strong> <code>{spec.training_missing_fraction * 100:.2f}%</code></li>
                    <li><strong>Collinear Cluster Allies:</strong> <code>{', '.join(spec.group_members) if spec.group_members else 'None (unique channel)'}</code></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

    r1_idx = np.flatnonzero(rules_res["rule_1_beyond_3sigma"])
    r2_idx = np.flatnonzero(rules_res["rule_2_of_3_beyond_2sigma"])
    r3_idx = np.flatnonzero(rules_res["rule_4_of_5_beyond_1sigma"])
    r4_idx = np.flatnonzero(rules_res["rule_8_on_one_side"])

    with col1:
        st.markdown("##### Western Electric Run-Rule Violations")
        rules_summary_df = pd.DataFrame(
            [
                {"Rule": "Rule 1 (> 3σ)", "Fault Signature": "Sudden single-run spike", "Violations": len(r1_idx)},
                {"Rule": "Rule 2 (2 of 3 > 2σ)", "Fault Signature": "Moderate step shift", "Violations": len(r2_idx)},
                {"Rule": "Rule 3 (4 of 5 > 1σ)", "Fault Signature": "Sustained process drift", "Violations": len(r3_idx)},
                {"Rule": "Rule 4 (8 one side)", "Fault Signature": "Long-term mean offset", "Violations": len(r4_idx)},
            ]
        )
        st.dataframe(rules_summary_df, width="stretch", hide_index=True)

    fig = go.Figure()
    if show_raw_trace:
        fig.add_trace(
            go.Scatter(
                x=run_indices,
                y=sensor_vals,
                mode="lines",
                name="Raw Sensor Reading (x_t)",
                line=dict(color="rgba(148, 163, 184, 0.45)", width=1),
            )
        )

    # Centre Line & Limits
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=ewma_res["z"],
            mode="lines",
            name="Smoothed EWMA (z_t)",
            line=dict(color="#0284c7", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=ewma_res["ucl"],
            mode="lines",
            name="Upper Control Limit (UCL, +3σ_z)",
            line=dict(color="#dc2626", dash="dash", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=ewma_res["lcl"],
            mode="lines",
            name="Lower Control Limit (LCL, -3σ_z)",
            line=dict(color="#dc2626", dash="dash", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=[spec.mu0] * n_runs,
            mode="lines",
            name="Phase I Centre Line (μ₀)",
            line=dict(color="#64748b", dash="dot", width=1.5),
        )
    )

    # Rule violation markers
    if len(r1_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=r1_idx,
                y=ewma_res["z"][r1_idx],
                mode="markers",
                name=f"Rule 1 (> 3σ) [{len(r1_idx)}]",
                marker=dict(color="#dc2626", symbol="x", size=9),
            )
        )

    if len(r2_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=r2_idx,
                y=ewma_res["z"][r2_idx],
                mode="markers",
                name=f"Rule 2 (2 of 3 > 2σ) [{len(r2_idx)}]",
                marker=dict(color="#ea580c", symbol="triangle-up", size=8),
            )
        )

    if len(r3_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=r3_idx,
                y=ewma_res["z"][r3_idx],
                mode="markers",
                name=f"Rule 3 (4 of 5 > 1σ) [{len(r3_idx)}]",
                marker=dict(color="#ca8a04", symbol="diamond-open", size=7),
            )
        )

    fig.update_layout(
        title=f"Phase II EWMA Control Chart — Sensor {selected_sensor} (λ={bundle.spc_model.ewma_lambda}, L={bundle.spc_model.ewma_l})",
        xaxis_title="Chronological Wafer Run Index (t)",
        yaxis_title=f"Sensor {selected_sensor} Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=65, b=20),
        height=460,
    )
    with col2:
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "💡 **How to read this chart:** The blue line tracks the exponentially weighted moving average ($z_t$) of the sensor. "
            "Missing sensor values hold $z_t$ constant without corrupting the state. Red `×` and orange `▲` markers highlight "
            "runs where the raw sensor violated Western Electric stability rules relative to Phase I baseline parameters."
        )

    with st.expander("📋 View All 10 Monitored Phase I Sensors Summary Table"):
        all_specs_rows = []
        for idx, s_id in enumerate(monitored, start=1):
            sp = bundle.spc_model.sensor_specs[s_id]
            all_specs_rows.append(
                {
                    "Selection Rank": f"#{idx}",
                    "Sensor ID": s_id,
                    "Phase I Mean (μ₀)": round(sp.mu0, 4),
                    "Moving-Range Sigma (σ₀)": round(sp.sigma0, 4),
                    "Rank-Biserial (r_rb)": round(sp.rank_biserial, 4),
                    "BH Adjusted p-value": f"{sp.bh_adjusted_p_value:.3e}",
                    "Train Missing (%)": round(sp.training_missing_fraction * 100, 2),
                    "Collinear Group Members": ", ".join(sp.group_members) if sp.group_members else "—",
                }
            )
        st.dataframe(pd.DataFrame(all_specs_rows), width="stretch", hide_index=True)


def render_mspc_view(
    scored_df: pd.DataFrame,
    bundle: LoadedBundle,
) -> None:
    st.subheader("2. Multivariate SPC: Hotelling T² & Squared Prediction Error (SPE)")
    st.markdown(
        "**Purpose:** Monitor all **274 retained sensors simultaneously** by projecting them onto a **50-component Principal Component Analysis (PCA)** "
        "model fitted strictly on Phase I passing wafer runs (explaining 62.1% of total variance)."
    )

    info_col1, info_col2, info_col3 = st.columns([1.2, 1.2, 1.0])
    with info_col1:
        st.markdown(
            """
            <div class="yg-card">
                <h4>🟢 Hotelling T² Statistic (Detector E04)</h4>
                <p style="margin:0; font-size:0.88rem;">
                    Measures <strong>in-model variation</strong>—how far a wafer run moves <em>along</em> the 50 normal correlation axes.
                    High $T^2$ indicates a global shift across many correlated sensors (e.g., overall chamber temperature or pressure drift).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col2:
        st.markdown(
            """
            <div class="yg-card">
                <h4>🟣 Squared Prediction Error / SPE (Detector E05)</h4>
                <p style="margin:0; font-size:0.88rem;">
                    Measures <strong>residual model breakdown ($Q$-statistic)</strong>—how far a wafer run deviates <em>perpendicular</em> to the PCA plane.
                    High $SPE$ catches localized sensor faults that break normal physical correlation between channels.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col3:
        use_log_scale = st.checkbox(
            "Use Logarithmic Y-Axis Scale (Log₁₀)",
            value=True,
            help="Recommended: Fault excursions on SECOM span plusieurs orders of magnitude; log scaling makes both baseline limits and extreme spikes clearly visible.",
        )
        st.caption(
            "**Why two threshold lines?** Heavy-tailed fab data causes textbook 99% Gaussian limits (red dashed) to over-alarm. "
            "Empirical **Budget-5 thresholds** (orange dotted) control the actual false alarm rate."
        )

    n_runs = len(scored_df)
    run_indices = np.arange(n_runs)

    t2_vals = scored_df["t2"].to_numpy()
    spe_vals = scored_df["spe"].to_numpy()

    t2_lim_theo = bundle.mspc_model.t2_limit_99
    spe_lim_theo = bundle.mspc_model.spe_limit_99
    t2_thresh_b5 = float(bundle.policy["thresholds"]["E04"]["budget_5"])
    spe_thresh_b5 = float(bundle.policy["thresholds"]["E05"]["budget_5"])

    t2_alarm_idx = np.flatnonzero(t2_vals > t2_thresh_b5)
    spe_alarm_idx = np.flatnonzero(spe_vals > spe_thresh_b5)

    col1, col2 = st.columns(2)
    with col1:
        fig_t2 = go.Figure()
        fig_t2.add_trace(
            go.Scatter(
                x=run_indices,
                y=t2_vals,
                mode="lines",
                name="T² Statistic",
                line=dict(color="#16a34a", width=1.5),
            )
        )
        if len(t2_alarm_idx) > 0:
            fig_t2.add_trace(
                go.Scatter(
                    x=t2_alarm_idx,
                    y=t2_vals[t2_alarm_idx],
                    mode="markers",
                    name=f"Budget 5 Alarm ({len(t2_alarm_idx)})",
                    marker=dict(color="#dc2626", size=7, symbol="circle"),
                )
            )
        fig_t2.add_trace(
            go.Scatter(
                x=[0, n_runs - 1],
                y=[t2_lim_theo, t2_lim_theo],
                mode="lines",
                name=f"Theoretical 99% Limit ({t2_lim_theo:.1f})",
                line=dict(color="#dc2626", dash="dash"),
            )
        )
        fig_t2.add_trace(
            go.Scatter(
                x=[0, n_runs - 1],
                y=[t2_thresh_b5, t2_thresh_b5],
                mode="lines",
                name=f"Budget 5 Threshold ({t2_thresh_b5:.1f})",
                line=dict(color="#ea580c", dash="dot", width=2),
            )
        )
        fig_t2.update_layout(
            title="Hotelling T² over Chronological Run Order (E04)",
            xaxis_title="Chronological Wafer Run Index",
            yaxis_title="Hotelling T² (Log Scale)" if use_log_scale else "Hotelling T²",
            yaxis_type="log" if use_log_scale else "linear",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=65, b=20),
            height=420,
        )
        st.plotly_chart(fig_t2, width="stretch")

        m1, m2 = st.columns(2)
        m1.metric("T² Budget 5 Alarms", int(len(t2_alarm_idx)), delta=f"{len(t2_alarm_idx) / n_runs * 100:.2f} / 100 runs", delta_color="off")
        m2.metric("T² Theoretical 99% Alarms", int(np.sum(t2_vals > t2_lim_theo)), delta=f"Limit = {t2_lim_theo:.2f}", delta_color="off")

    with col2:
        fig_spe = go.Figure()
        fig_spe.add_trace(
            go.Scatter(
                x=run_indices,
                y=spe_vals,
                mode="lines",
                name="SPE Statistic",
                line=dict(color="#7c3aed", width=1.5),
            )
        )
        if len(spe_alarm_idx) > 0:
            fig_spe.add_trace(
                go.Scatter(
                    x=spe_alarm_idx,
                    y=spe_vals[spe_alarm_idx],
                    mode="markers",
                    name=f"Budget 5 Alarm ({len(spe_alarm_idx)})",
                    marker=dict(color="#dc2626", size=7, symbol="circle"),
                )
            )
        fig_spe.add_trace(
            go.Scatter(
                x=[0, n_runs - 1],
                y=[spe_lim_theo, spe_lim_theo],
                mode="lines",
                name=f"Theoretical 99% Limit ({spe_lim_theo:.1f})",
                line=dict(color="#dc2626", dash="dash"),
            )
        )
        fig_spe.add_trace(
            go.Scatter(
                x=[0, n_runs - 1],
                y=[spe_thresh_b5, spe_thresh_b5],
                mode="lines",
                name=f"Budget 5 Threshold ({spe_thresh_b5:.1f})",
                line=dict(color="#ea580c", dash="dot", width=2),
            )
        )
        fig_spe.update_layout(
            title="Squared Prediction Error (SPE / Q) over Run Order (E05)",
            xaxis_title="Chronological Wafer Run Index",
            yaxis_title="SPE Statistic (Log Scale)" if use_log_scale else "SPE Statistic",
            yaxis_type="log" if use_log_scale else "linear",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=65, b=20),
            height=420,
        )
        st.plotly_chart(fig_spe, width="stretch")

        s1, s2 = st.columns(2)
        s1.metric("SPE Budget 5 Alarms", int(len(spe_alarm_idx)), delta=f"{len(spe_alarm_idx) / n_runs * 100:.2f} / 100 runs", delta_color="off")
        s2.metric("SPE Theoretical 99% Alarms", int(np.sum(spe_vals > spe_lim_theo)), delta=f"Limit = {spe_lim_theo:.2f}", delta_color="off")


def render_drilldown_view(
    scored_df: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    bundle: LoadedBundle,
) -> None:
    st.subheader("3. Flagged-Run Drill-Down: Root-Cause Sensor Triage")
    st.markdown(
        "**Purpose:** When a wafer run triggers an alarm, yield engineers need to know **which sensors caused the excursion**. "
        "This view compares two complementary attribution methods side by side:\n"
        "1. **MSPC Physical Contributions (Left):** Decomposes $T^2$ and $SPE$ into per-sensor squared deviations from normal passing wafers.\n"
        "2. **XGBoost TreeSHAP Attributions (Right):** Computes exact additive log-odds contributions pushing the ML model toward predicting wafer failure."
    )

    # Merge labels if available for richer context
    label_map: dict[str, int] = {}
    if labels_df is not None and "label_fail" in labels_df.columns:
        label_map = dict(zip(labels_df["run_id"].astype(str), labels_df["label_fail"].astype(int)))

    alarm_mask = (
        scored_df["alarm"] if "alarm" in scored_df.columns else np.zeros(len(scored_df), dtype=bool)
    )
    flagged_df = scored_df[alarm_mask]
    if len(flagged_df) == 0:
        flagged_df = scored_df.head(10)
        st.info("No runs flagged by primary alarm threshold; showing first 10 runs for inspection.")

    run_options = list(flagged_df["run_id"].astype(str))
    run_meta_lookup = flagged_df.set_index(flagged_df["run_id"].astype(str)).to_dict(orient="index")

    def _format_run_option(r_id: str) -> str:
        meta = run_meta_lookup.get(r_id, {})
        blk = meta.get("block", "unknown")
        e02_val = meta.get("e02_score", 0.0)
        lbl_str = ""
        if r_id in label_map:
            lbl_str = " | Ground Truth: FAIL 🔴" if label_map[r_id] == 1 else " | Ground Truth: PASS 🟢"
        return f"Run {r_id} (Block: {blk} | Primary E02 Score: {e02_val:.2f}{lbl_str})"

    selected_run_id = st.selectbox(
        f"Select Flagged Wafer Run to Inspect ({len(run_options)} Flagged Runs Available)",
        run_options,
        index=0,
        format_func=_format_run_option,
    )

    run_row = scored_df[scored_df["run_id"].astype(str) == selected_run_id].iloc[0]
    explanation = rank_sensors_for_run(
        run_row,
        quality=bundle.quality,
        mspc_model=bundle.mspc_model,
        xgb_model_or_booster=bundle.xgb_model,
        top_k=5,
    )

    # Run Diagnostic Status Strip
    tau_b5 = float(bundle.policy.get("thresholds", {}).get("E02", {}).get("budget_5", 6.2822))
    gt_status = (
        "FAIL (Scrap/Defect)"
        if label_map.get(selected_run_id) == 1
        else ("PASS (Normal)" if label_map.get(selected_run_id) == 0 else "Unlabeled")
    )
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric("Selected Run ID", selected_run_id, delta=f"Block: {run_row.get('block', 'N/A')}", delta_color="off")
    rc2.metric("Wafer Electrical Test", gt_status, delta=f"Timestamp: {str(run_row.get('timestamp', 'N/A'))[:16]}", delta_color="off")
    rc3.metric("Primary E02 Score", f"{float(run_row.get('e02_score', 0.0)):.2f}", delta=f"Cutoff τ = {tau_b5:.2f}", delta_color="off")
    rc4.metric("MSPC Index (E06)", f"{float(run_row.get('mspc_index', 0.0)):.2f}", delta=f"SPE = {float(run_row.get('spe', 0.0)):.1f}", delta_color="off")
    rc5.metric("Missing Channels", f"{int(run_row.get('n_missing', 0))} / 590", delta="Imputed via Phase I median", delta_color="off")

    overlap = explanation["overlap"]["top_k_common_sensors"]
    overlap_text = ", ".join(overlap) if overlap else "None (MSPC and XGBoost focused on distinct physical modes)"
    st.info(
        f"🔍 **Cross-Method Consensus (Top-5 Overlap):** `{overlap_text}`  \n"
        f"⚠️ **Attribution Smearing Caveat:** {SMEARING_CAVEAT}"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. Top-5 MSPC Physical Contributing Sensors")
        mspc_sensors = [x["sensor"] for x in explanation["mspc_top_sensors"]]
        mspc_vals = [x["contribution"] for x in explanation["mspc_top_sensors"]]
        fig_m = go.Figure(
            go.Bar(
                x=mspc_vals,
                y=mspc_sensors,
                orientation="h",
                marker=dict(color="#0284c7"),
                text=[f"{v:.2f}" for v in mspc_vals],
                textposition="auto",
            )
        )
        fig_m.update_layout(
            xaxis_title="Combined Standardized MSPC Contribution",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=15, b=20),
            height=280,
        )
        st.plotly_chart(fig_m, width="stretch")

        mspc_table_rows = []
        for item in explanation["mspc_top_sensors"]:
            mspc_table_rows.append(
                {
                    "Rank": f"#{item['rank']}",
                    "Sensor": item["sensor"],
                    "Combined Score": round(float(item["contribution"]), 3),
                    "SPE Share": round(float(item.get("spe_contribution", 0.0)), 2),
                    "T² Share": round(float(item.get("t2_contribution", 0.0)), 2),
                    "Correlated Cluster Members (|r| > 0.98)": (
                        ", ".join(item["group_members"]) if item["group_members"] else "—"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(mspc_table_rows), width="stretch", hide_index=True)

    with col2:
        st.markdown("#### 2. Top-5 XGBoost TreeSHAP Attributions (Log-Odds)")
        shap_sensors = [x["sensor"] for x in explanation["xgb_shap_top_sensors"]]
        shap_vals = [x["shap_value"] for x in explanation["xgb_shap_top_sensors"]]
        colors = ["#dc2626" if v > 0 else "#16a34a" for v in shap_vals]
        fig_s = go.Figure(
            go.Bar(
                x=shap_vals,
                y=shap_sensors,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:+.3f}" for v in shap_vals],
                textposition="auto",
            )
        )
        fig_s.update_layout(
            xaxis_title="SHAP Contribution to Failure Log-Odds (Red = Risk ↑, Green = Risk ↓)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=15, b=20),
            height=280,
        )
        st.plotly_chart(fig_s, width="stretch")

        shap_table_rows = []
        for item in explanation["xgb_shap_top_sensors"]:
            shap_table_rows.append(
                {
                    "Rank": f"#{item['rank']}",
                    "Sensor": item["sensor"],
                    "SHAP Value": f"{float(item['shap_value']):+.4f}",
                    "Risk Direction": "Increases Fail Risk 🔴" if item["shap_value"] > 0 else "Decreases Fail Risk 🟢",
                    "Correlated Cluster Members (|r| > 0.98)": (
                        ", ".join(item["group_members"]) if item["group_members"] else "—"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(shap_table_rows), width="stretch", hide_index=True)


def render_alarm_budget_view(
    scored_df: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    bundle: LoadedBundle,
) -> None:
    st.subheader("4. Alarm-Budget Performance & Official Holdout Benchmark Audit")
    st.markdown(
        "**Purpose:** Compare all candidate detectors (`E02` through `E08`) under fixed engineering review capacities: "
        "**2, 5 (primary), and 10 false alarms per 100 passing runs**. Thresholds ($\\tau$) were calibrated on development "
        "block `D5` (235 runs) and frozen prior to holdout evaluation."
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

    budget_filter = st.radio(
        "Filter Comparison Table & Chart by Review Budget:",
        ["All Budgets (2, 5, 10)", "Budget 5 per 100 (Primary)", "Budget 2 per 100 (Strict)", "Budget 10 per 100 (High Sensitivity)"],
        index=0,
        horizontal=True,
    )

    rows: list[dict[str, Any]] = []
    detectors = ["E02", "E03", "E04", "E05", "E06", "E07", "E08"]
    thresholds = bundle.policy.get("thresholds", {})

    for det in detectors:
        det_info = DETECTOR_CATALOG.get(det, {"name": det, "family": "Detector"})
        for b in (2, 5, 10):
            if budget_filter.startswith("Budget 5") and b != 5:
                continue
            if budget_filter.startswith("Budget 2") and b != 2:
                continue
            if budget_filter.startswith("Budget 10") and b != 10:
                continue

            tau = thresholds.get(det, {}).get(f"budget_{b}")
            if tau is None:
                continue
            flag_col = f"alarm_{det.lower()}_b{b}"
            alarms_count = int(scored_df[flag_col].sum()) if flag_col in scored_df.columns else 0
            realised_rate = float(alarms_count / n_runs * 100.0)

            row: dict[str, Any] = {
                "Detector": det_info["name"],
                "Family": det_info["family"],
                "Budget": f"{b} per 100",
                "Threshold (τ)": round(float(tau), 4),
                "Total Alarms": alarms_count,
                "Realised Rate (/100)": round(realised_rate, 2),
            }

            if has_labels and y_true is not None:
                flags = merged[flag_col].to_numpy()
                true_alarms = int(np.sum((flags == 1) & (y_true == 1)))
                false_alarms = int(np.sum((flags == 1) & (y_true == 0)))
                recall = float(true_alarms / n_fails) if n_fails and n_fails > 0 else 0.0
                precision = float(true_alarms / alarms_count) if alarms_count > 0 else 0.0
                row["True Alarms (Caught Fails)"] = true_alarms
                row["False Alarms"] = false_alarms
                row["Precision (%)"] = round(precision * 100.0, 1)
                row["Recall (%)"] = round(recall * 100.0, 1)

            rows.append(row)

    df_table = pd.DataFrame(rows)
    st.dataframe(df_table, width="stretch", hide_index=True)

    # Visual comparison chart at Budget 5
    if has_labels and y_true is not None and n_fails and n_fails > 0:
        b5_names = []
        b5_recalls = []
        b5_rates = []
        for det in detectors:
            flag_col = f"alarm_{det.lower()}_b5"
            if flag_col in merged.columns:
                flags = merged[flag_col].to_numpy()
                tp = int(np.sum((flags == 1) & (y_true == 1)))
                b5_names.append(DETECTOR_CATALOG[det]["short"])
                b5_recalls.append(tp / n_fails * 100.0)
                b5_rates.append(float(np.sum(flags == 1)) / len(merged) * 100.0)

        fig_cmp = go.Figure()
        fig_cmp.add_trace(
            go.Bar(name="Fault Recall (% of Fails Caught)", x=b5_names, y=b5_recalls, marker_color="#0f766e")
        )
        fig_cmp.add_trace(
            go.Bar(name="Realised Alarm Rate (Alarms per 100 Runs)", x=b5_names, y=b5_rates, marker_color="#f59e0b")
        )
        fig_cmp.update_layout(
            title="Detector Comparison at Primary Budget (5 Alarms / 100 Runs)",
            barmode="group",
            yaxis_title="Percentage (%) / Rate per 100 Runs",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=55, b=20),
            height=340,
        )
        st.plotly_chart(fig_cmp, width="stretch")

    # Show structured holdout metrics & drift summary if available
    holdout_metrics_p = Path("reports/holdout_metrics.json")
    if holdout_metrics_p.exists():
        st.divider()
        st.markdown("### 🏆 Official One-Shot Holdout Evaluation Scorecard (392 Unseen Runs)")
        st.caption(
            "Evaluated once post-freeze (`reports/holdout_metrics.json`). Predictions were written and SHA-256 hashed "
            "before opening `labels_holdout.parquet`. All figures are offline benchmark results, not financial or yield savings."
        )
        h_data = json.loads(holdout_metrics_p.read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("Holdout Runs", h_data["n_holdout_runs"])
        cols[1].metric("Holdout Fails", h_data["n_holdout_fails"])
        cols[2].metric("Holdout Prevalence", f"{h_data['holdout_fail_prevalence'] * 100:.2f}%")
        cols[3].metric("Primary Detector", f"{h_data['primary_detector']} (SPC Individuals)")

        obj = h_data.get("objectives", {})
        o1 = obj.get("O1_ranking_pr_auc", {})
        o2 = obj.get("O2_recall_at_5_far", {})
        o3 = obj.get("O3_alarm_budget_stability", {})
        o4 = obj.get("O4_spc_vs_ml_comparison", {})

        scorecard_rows = [
            {
                "Objective": "O1 — Fault Ranking PR-AUC",
                "Predeclared Target": f"PR-AUC ≥ 2× prevalence ({o1.get('target_min_pr_auc_2x_prevalence', 0.1224):.4f})",
                "Measured Holdout Result": f"{o1.get('actual_pr_auc', 0.0724):.4f} (Prevalence: 0.0612)",
                "95% Wider Bootstrap CI": f"[{o1.get('wider_95_ci', [0.0384, 0.1514])[0]:.4f}, {o1.get('wider_95_ci', [0.0384, 0.1514])[1]:.4f}]",
                "Verdict": f"{o1.get('verdict', 'MISS')} 🔴",
                "Engineering Interpretation": "Above chance (0.0612), but severe temporal sensor drift prevented reaching 2× lift.",
            },
            {
                "Objective": "O2 — Recall @ 5% False Alarm Rate",
                "Predeclared Target": "Recall ≥ 20.0% at 5 FAR / 100 passing runs",
                "Measured Holdout Result": f"{o2.get('actual_recall_at_5_far', 0.0417) * 100:.2f}% (1 / 24 fails; 12.5% @ 10 FAR)",
                "95% Wider Bootstrap CI": f"[{o2.get('wider_95_ci', [0.0, 0.1765])[0] * 100:.1f}%, {o2.get('wider_95_ci', [0.0, 0.1765])[1] * 100:.1f}%]",
                "Verdict": f"{o2.get('verdict', 'MISS')} 🔴",
                "Engineering Interpretation": "High-dimensional sensor noise without spatial wafer coordinates limits low-FAR recall.",
            },
            {
                "Objective": "O3 — Alarm Budget Stability",
                "Predeclared Target": "Realised alarm rate within [2.0, 8.0] per 100 runs",
                "Measured Holdout Result": f"{o3.get('realised_alarms_per_100', 3.06):.2f} per 100 ({o3.get('alarms', 12)} alarms / 392 runs)",
                "95% Wider Bootstrap CI": "Deterministic count on holdout",
                "Verdict": f"{o3.get('verdict', 'PASS')} 🟢",
                "Engineering Interpretation": "D5 quantile threshold successfully bounded operator review workload on unseen future runs.",
            },
            {
                "Objective": "O4 — SPC vs ML Comparison",
                "Predeclared Target": "Report paired difference (E08 XGBoost − E02 SPC) & 95% CI",
                "Measured Holdout Result": f"{o4.get('diff_ml_minus_spc_dev_pair', 0.0153):+.4f} (E08: 0.0877 vs E02: 0.0724)",
                "95% Wider Bootstrap CI": f"[{o4.get('wider_95_ci_dev_pair', [-0.0470, 0.0910])[0]:+.4f}, {o4.get('wider_95_ci_dev_pair', [-0.0470, 0.0910])[1]:+.4f}]",
                "Verdict": f"{o4.get('verdict', 'REPORTED')} 🔵",
                "Engineering Interpretation": "CI spans zero on holdout too—confirming Rule 4 parsimony choice of E02 over XGBoost.",
            },
            {
                "Objective": "O5 — Sensor Attribution Accuracy",
                "Predeclared Target": "Injected 5σ single-sensor fault ranked in Top-3 ≥ 90%",
                "Measured Holdout Result": "100.0% in Top-3 (100% Top-1 MSPC, 96% SPE)",
                "95% Wider Bootstrap CI": "100 synthetic trials",
                "Verdict": "PASS 🟢",
                "Engineering Interpretation": "MSPC contribution decomposition reliably isolates localized sensor excursions.",
            },
            {
                "Objective": "O6 & O7 — Reproducibility & API Parity",
                "Predeclared Target": "Score delta ≤ 1e-6 (O6); CLI/API parity ≤ 1e-9 (O7)",
                "Measured Holdout Result": "0.0 max score difference; warm p95 latency 63.8 ms",
                "95% Wider Bootstrap CI": "100 benchmark trials",
                "Verdict": "PASS 🟢",
                "Engineering Interpretation": "Shared validation & cleaning pipeline guarantees bit-exact training/serving parity.",
            },
        ]
        st.dataframe(pd.DataFrame(scorecard_rows), width="stretch", hide_index=True)

        # Drift explanation card if drift summary exists
        drift_p = Path("reports/drift/drift_summary.json")
        if drift_p.exists():
            d_data = json.loads(drift_p.read_text(encoding="utf-8"))
            st.markdown(
                f"""
                <div class="yg-card">
                    <h4>📉 Why Did Holdout PR-AUC Drop? Evidence from Evidently Data Drift Monitoring</h4>
                    <p style="margin:0; font-size:0.9rem;">
                        Comparing the 940 Phase I–IV reference runs against the 392 unseen holdout runs (`reports/drift/drift_summary.json`) reveals
                        severe non-stationarity across the 2008 manufacturing timeline:
                        <strong>{d_data.get('number_of_drifted_columns', 237)} of {d_data.get('number_of_columns', 274)} retained sensors
                        ({d_data.get('share_of_drifted_columns', 0.865) * 100:.1f}%) drifted significantly</strong> (Two-sample K-S test, $p < 0.05$),
                        and <strong>{d_data.get('missing_rate_summary', {}).get('columns_with_missingness_change_gt_5pct', 48)} sensors</strong> shifted
                        in missingness by more than $\\pm 5\\%$ (up to $+61.0\\%$ on `f247`, `f112`, `f385`). Despite this drift, the calibrated
                        <strong>E02 alarm budget held steady at 3.06 alarms / 100 runs (Target O3 PASS)</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_project_overview_view(bundle: LoadedBundle) -> None:
    """Render comprehensive educational overview of YieldGuard architecture, methodology, and detectors."""
    st.subheader("5. Project Overview, Chronological Protocol & Detector Glossary")
    st.markdown(
        "This reference page explains the engineering architecture, leakage-free data splitting, and statistical "
        "foundations of **YieldGuard** so any reviewer or process engineer can understand the system end-to-end."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1. Chronological Split & Zero-Leakage Guarantee")
        st.markdown(
            "Semiconductor manufacturing tools experience maintenance cycles, recipe adjustments, and sensor drift over time. "
            "**Random row shuffling or standard $k$-fold cross-validation leaks future chamber states into past training runs.** "
            "YieldGuard enforces a strict timestamp-ordered split across all 1,567 SECOM wafer runs:"
        )
        split_rows = [
            {"Block": "D1 (Dev Block 1)", "Runs": 235, "Fails": "32 (13.62%)", "Role in Protocol": "Phase I Training (Fold 1, 2, 3)"},
            {"Block": "D2 (Dev Block 2)", "Runs": 235, "Fails": "26 (11.06%)", "Role in Protocol": "Phase I Training (Fold 1, 2, 3)"},
            {"Block": "D3 (Dev Block 3)", "Runs": 235, "Fails": "8 (3.40%)", "Role in Protocol": "Fold 1 Validation / Fold 2 & 3 Training"},
            {"Block": "D4 (Dev Block 4)", "Runs": 235, "Fails": "10 (4.26%)", "Role in Protocol": "Fold 2 Validation / Fold 3 (Final Staging) Training"},
            {"Block": "D5 (Calibration)", "Runs": 235, "Fails": "4 (1.70%)", "Role in Protocol": "Fold 3 Validation & Alarm-Budget Threshold Calibration"},
            {"Block": "Holdout (Sealed)", "Runs": 392, "Fails": "24 (6.12%)", "Role in Protocol": "One-Shot Final Evaluation (Opened Once Post-Freeze)"},
        ]
        st.dataframe(pd.DataFrame(split_rows), width="stretch", hide_index=True)

    with c2:
        st.markdown("#### 2. Four-Stage Train-Only Feature Cleaning")
        st.markdown(
            "Fitted strictly on the 940 runs of `D1`–`D4` (`models/v0.1.0/quality.joblib`) and applied identically to batch CSVs, "
            "FastAPI requests, and dashboard views:"
        )
        clean_rows = [
            {"Stage": "1. Raw Ingestion", "Rule": "Verify 590 numeric channels (`f000`–`f589`), canonicalize NaNs, reject ±Inf", "Columns Remaining": 590},
            {"Stage": "2. Missingness Pruning", "Rule": "Drop sensors with > 50% missing values on training runs", "Columns Remaining": 566},
            {"Stage": "3. Zero-Variance Pruning", "Rule": "Drop constant flatline sensors (std = 0 on training runs)", "Columns Remaining": 440},
            {"Stage": "4. Collinear Deduplication", "Rule": "Cluster sensors with |r| > 0.98; keep 1 representative, log allies in group_map", "Columns Remaining": 274},
            {"Stage": "5. Imputation & Scaling", "Rule": "Impute missing values with training medians; fit passing-run Z-score scaler", "Columns Remaining": 274},
        ]
        st.dataframe(pd.DataFrame(clean_rows), width="stretch", hide_index=True)

    st.markdown("#### 3. Complete Detector Bank (`E01` – `E08`)")
    det_rows = []
    thresholds = bundle.policy.get("thresholds", {})
    for det_id, info in DETECTOR_CATALOG.items():
        t_dict = thresholds.get(det_id, {})
        det_rows.append(
            {
                "ID": det_id,
                "Detector Name": info["name"],
                "Family": info["family"],
                "Budget 2 Cutoff": round(float(t_dict.get("budget_2", 0.0)), 4),
                "Budget 5 Cutoff (Primary)": round(float(t_dict.get("budget_5", 0.0)), 4),
                "Budget 10 Cutoff": round(float(t_dict.get("budget_10", 0.0)), 4),
                "Methodology & Mechanism": info["desc"],
            }
        )
    st.dataframe(pd.DataFrame(det_rows), width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="YieldGuard — Semiconductor Process Control",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    st.sidebar.title("🔬 YieldGuard")
    st.sidebar.markdown(
        "**Semiconductor Process Control & Fault Detection**\n\n"
        "Decision-support console for wafer fab yield engineers."
    )

    real_available = check_real_data_available()
    mode_options = (
        ["Real Dataset (SECOM)", "Synthetic Demo Data"]
        if real_available
        else ["Synthetic Demo Data"]
    )
    selected_mode = st.sidebar.selectbox(
        "Data Mode",
        mode_options,
        index=0,
        help="Switch between the real UCI SECOM benchmark dataset (1,567 runs) and an on-the-fly synthetic wafer dataset (100 runs).",
    )
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

    # View navigation (preserving exact first 4 view strings for AppTest compatibility)
    views = [
        "1. Univariate EWMA & Run Rules",
        "2. Multivariate SPC (T² & SPE)",
        "3. Flagged-Run Drill-Down",
        "4. Alarm-Budget Performance",
        "5. Project Overview & Architecture",
    ]
    active_view = st.sidebar.radio(
        "Dashboard View",
        views,
        index=0,
        help="Navigate between process control charts, flagged-run root-cause explanations, and benchmark evaluations.",
    )

    st.sidebar.divider()
    with st.sidebar.expander("🗂️ Quick Detector Legend (E01–E08)", expanded=False):
        st.markdown(
            "- **E01**: Chance Baseline\n"
            "- **E02 ★**: Univariate SPC (Individuals)\n"
            "- **E03**: Univariate SPC (EWMA)\n"
            "- **E04**: Hotelling $T^2$ (50-PCA)\n"
            "- **E05**: Squared Prediction Error ($SPE$)\n"
            "- **E06**: Combined MSPC Index\n"
            "- **E07**: Isolation Forest\n"
            "- **E08**: Class-Weighted XGBoost"
        )
    st.sidebar.caption(
        f"**Release Bundle:** `{bundle.version}`  \n"
        "**Claims Boundary:** Offline benchmark results on UCI SECOM (CC BY 4.0); not measured fab yield or financial savings."
    )

    # Render global context header & KPI strip
    render_top_header_and_kpis(scored_df, labels_df, bundle, is_synthetic)

    if active_view == "1. Univariate EWMA & Run Rules":
        render_ewma_view(scored_df, bundle)
    elif active_view == "2. Multivariate SPC (T² & SPE)":
        render_mspc_view(scored_df, bundle)
    elif active_view == "3. Flagged-Run Drill-Down":
        render_drilldown_view(scored_df, labels_df, bundle)
    elif active_view == "4. Alarm-Budget Performance":
        render_alarm_budget_view(scored_df, labels_df, bundle)
    elif active_view == "5. Project Overview & Architecture":
        render_project_overview_view(bundle)


if __name__ == "__main__":
    main()
