"""YieldGuard: Semiconductor Process Control & Fault Detection Streamlit Dashboard.

A plain-language landing page plus the four views of docs/04_TECHNICAL_DESIGN.md section 8:
- Start here (Project Overview): the problem, the method, the frozen findings and a glossary.
- Single-sensor trends (Univariate EWMA & Run Rules): EWMA chart with limits and Western Electric rules.
- All sensors at once (Multivariate SPC): Hotelling T-squared and SPE over run order with budget thresholds.
- Why was a run flagged? (Drill-Down): MSPC contributions and XGBoost TreeSHAP side by side.
- Results (Alarm-Budget Performance): metrics at budgets 2, 5, 10, holdout scorecard and drift summary.

Each view opens with a jargon-free summary and reading guide; method details sit in a collapsed expander.
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
        "desc": "Sum of normalized excursions across Hotelling T² and SPE statistics (T² / τ_T² + SPE / τ_SPE).",
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

# One-sentence, jargon-free description of each detector for newcomers.
DETECTOR_PLAIN: dict[str, str] = {
    "E01": "Random guessing. The bar every other detector has to beat.",
    "E02": "Flags a run when any of 10 key sensors is far outside its normal range. The chosen detector.",
    "E03": "Like E02, but on a smoothed trend, so it reacts to slow drifts rather than single spikes.",
    "E04": "One number for all sensors: how far the run moved from normal in the usual pattern.",
    "E05": "One number for all sensors: how much the run breaks the usual relationships between sensors.",
    "E06": "Combines E04 and E05 by adding their normalized alarm distances together.",
    "E07": "Machine learning that studies good runs only and flags runs that look unlike them.",
    "E08": "Machine learning that studies labelled good and failed runs and predicts failure risk.",
}

VIEW_EWMA = "1. Univariate EWMA & Run Rules"
VIEW_MSPC = "2. Multivariate SPC (T² & SPE)"
VIEW_DRILLDOWN = "3. Flagged-Run Drill-Down"
VIEW_BUDGET = "4. Alarm-Budget Performance"
VIEW_OVERVIEW = "5. Project Overview & Architecture"
VIEW_WAFER = "6. Semiconductor Wafer Etch (v0.2 Benchmark)"

# Sidebar navigation: plain-language label and the technical name shown beneath it.
VIEW_LABELS: dict[str, tuple[str, str]] = {
    VIEW_OVERVIEW: ("Start here", "What the project is and what it found"),
    VIEW_EWMA: ("Single-sensor trends", "EWMA charts and run rules"),
    VIEW_MSPC: ("All sensors at once", "Hotelling T² and SPE"),
    VIEW_DRILLDOWN: ("Why was a run flagged?", "Sensor ranking for a flagged run"),
    VIEW_BUDGET: ("Results", "Detector comparison and final test"),
    VIEW_WAFER: ("Wafer Etch benchmark", "Confirmatory results on second dataset (v0.2)"),
}

GLOSSARY: list[tuple[str, str]] = [
    ("Run", "One pass of a wafer (the silicon disc chips are built on) through production. Each data row is one run."),
    ("Sensor (f000–f589)", "One measured signal. The dataset hides what each sensor physically measures."),
    ("Pass / fail", "The final quality-test result for a run. Used only to check the detectors, never as an input."),
    ("Detector", "A method that gives each run an 'unusualness' score. Eight are compared (E01–E08)."),
    ("Alarm budget", "How many alarms per 100 runs engineers can realistically review. The main budget here is 5."),
    ("Score cutoff (threshold, τ)", "The score above which a run is flagged. Chosen so the alarm budget is respected."),
    ("SPC", "Statistical process control: classic factory control charts that flag readings outside their normal band."),
    ("EWMA", "A smoothed running average that weights recent runs more. Good at spotting slow drifts."),
    ("T² and SPE", "Two summary scores across all sensors: T² is distance from normal; SPE is how much the usual pattern broke."),
    ("Recall", "Of the runs that really failed, the share that was flagged."),
    ("Precision", "Of the runs that were flagged, the share that really failed."),
    ("PR-AUC", "One number from 0 to 1 for how well a detector ranks failed runs above good ones. Random guessing scores about the failure rate."),
    ("Holdout", "The most recent runs, locked away until every choice was frozen, then scored exactly once."),
    ("Drift", "The process and its sensors changing over time, so the old picture of 'normal' no longer fits."),
    ("SHAP", "A way of showing which inputs pushed a machine-learning prediction up or down."),
]


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
            border: 1px solid rgba(15, 118, 110, 0.55);
        }
        .yg-badge-amber {
            background-color: rgba(180, 83, 9, 0.15);
            border: 1px solid rgba(180, 83, 9, 0.55);
        }
        .yg-badge-slate {
            background-color: rgba(100, 116, 139, 0.15);
            border: 1px solid rgba(100, 116, 139, 0.55);
        }
        div[data-testid="stMetricValue"] {
            font-variant-numeric: tabular-nums;
            font-size: 1.6rem;
        }
        div[data-testid="stMetricLabel"] p {
            white-space: normal;
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


def load_json_report(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def render_view_intro(
    title: str,
    summary: str,
    how_to_read: list[str],
    technical: str,
) -> None:
    """Render a view title, a plain-language summary, a reading guide and collapsed technical notes."""
    st.header(title)
    st.markdown(summary)
    with st.container(border=True):
        st.markdown("**How to read this page**\n\n" + "\n".join(f"- {line}" for line in how_to_read))
    with st.expander("Technical details"):
        st.markdown(technical)


def render_top_header_and_kpis(
    scored_df: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    bundle: LoadedBundle,
    is_synthetic: bool,
) -> None:
    """Render the project banner and the KPI strip shared by every view."""
    n_runs = len(scored_df)
    mode_badge = (
        f'<span class="yg-badge yg-badge-amber">SYNTHETIC DEMO DATA ({n_runs:,} RUNS)</span>'
        if is_synthetic
        else f'<span class="yg-badge yg-badge-teal">REAL DATA: UCI SECOM ({n_runs:,} RUNS)</span>'
    )
    version_badge = f'<span class="yg-badge yg-badge-slate">MODEL {bundle.version}</span>'
    primary_id = bundle.policy.get("primary_detector", "E02")
    budget = float(bundle.policy.get("primary_alarm_budget_per_100", 5.0))

    st.markdown(
        f"""
        <div class="yg-header">
            <div style="margin-bottom: 0.45rem;">{mode_badge}{version_badge}</div>
            <h1>YieldGuard: early warning for faulty chip-production runs</h1>
            <p>
                Factory sensors record every production run. YieldGuard scores each run and flags the most unusual
                ones for an engineer to check, while keeping to about <strong>{budget:g} alarms per 100 runs</strong>.
                It also tests whether machine learning does this better than classic control charts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_flagged = int(scored_df["alarm"].sum()) if "alarm" in scored_df.columns else 0
    flagged_rate = (n_flagged / n_runs * 100.0) if n_runs > 0 else 0.0
    tau_primary = bundle.policy.get("thresholds", {}).get(primary_id, {}).get(f"budget_{budget:g}")

    if labels_df is not None and "label_fail" in labels_df.columns:
        merged_kpi = pd.merge(scored_df[["run_id", "alarm"]], labels_df[["run_id", "label_fail"]], on="run_id", how="inner")
        n_fails = int((merged_kpi["label_fail"] == 1).sum())
        fail_prev = (n_fails / len(merged_kpi) * 100.0) if len(merged_kpi) > 0 else 0.0
        fail_str = f"{n_fails} failed final test ({fail_prev:.1f}%)"
    else:
        fail_str = "Test results not available"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Production runs",
        f"{n_runs:,}",
        help="Runs loaded in this session and how many of them failed the final quality test.",
    )
    k2.metric(
        "Sensors used",
        f"{len(bundle.quality.retained_columns)} of {len(bundle.quality.input_columns)}",
        help="Sensors kept after removing ones that were mostly empty, never changed or duplicated another sensor.",
    )
    k3.metric(
        "Alarm budget",
        f"{budget:g} per 100",
        help=(
            "How many alarms per 100 runs engineers can realistically review. The cutoff was set on separate "
            f"calibration runs for detector {primary_id} and then frozen."
        ),
    )
    k4.metric(
        "Runs flagged",
        f"{n_flagged:,}",
        help=f"Runs whose score from the chosen detector ({primary_id}) is above the cutoff.",
    )
    k1.caption(fail_str)
    k2.caption(f"{len(bundle.spc_model.monitored_sensors)} tracked one by one")
    k3.caption(f"Score cutoff {float(tau_primary):.2f}" if tau_primary is not None else "No cutoff stored")
    k4.caption(f"{flagged_rate:.1f} per 100 runs")
    st.divider()


def render_headline_findings(holdout: dict[str, Any]) -> None:
    """Render the frozen holdout verdicts as three plain-language cards."""
    obj = holdout.get("objectives", {})
    o2 = obj.get("O2_recall_at_5_far", {})
    o3 = obj.get("O3_alarm_budget_stability", {})
    o4 = obj.get("O4_spc_vs_ml_comparison", {})
    n_fails = int(holdout.get("n_holdout_fails", 0))
    n_holdout = int(holdout.get("n_holdout_runs", 0))
    primary_id = holdout.get("primary_detector", "E02")

    def _badge(verdict: str) -> None:
        if verdict == "PASS":
            st.badge("Target met", color="green", icon=":material/check:")
        elif verdict == "MISS":
            st.badge("Target missed", color="red", icon=":material/close:")
        else:
            st.badge("Reported, no target", color="blue", icon=":material/info:")

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        _badge(str(o2.get("verdict", "")))
        st.markdown("**Catching failed runs**")
        if o2 and n_fails > 0:
            recall = float(o2["actual_recall_at_5_far"])
            st.markdown(
                f"Allowing 5 false alarms per 100 good runs, the chosen detector flagged "
                f"**{round(recall * n_fails)} of {n_fails}** failed runs ({recall * 100:.1f}%). "
                f"The target was {float(o2['target_min_recall']) * 100:.0f}%; "
                f"random guessing would reach about {float(o2['chance_recall']) * 100:.0f}%."
            )
    with c2, st.container(border=True):
        _badge(str(o3.get("verdict", "")))
        st.markdown("**Keeping alarms manageable**")
        if o3:
            lo, hi = o3["allowed_range_per_100"]
            st.markdown(
                f"With the cutoff fixed in advance, it raised **{o3['alarms']} alarms on {o3['n_runs']} runs** "
                f"({float(o3['realised_alarms_per_100']):.1f} per 100), inside the allowed {lo:g} to {hi:g}. "
                f"{o3['true_alarms']} of those alarms were real failures."
            )
    with c3, st.container(border=True):
        _badge(str(o4.get("verdict", "")))
        st.markdown("**Machine learning vs. classic charts**")
        if o4:
            diff = float(o4["diff_ml_minus_spc_dev_pair"])
            lo, hi = o4["wider_95_ci_dev_pair"]
            spans_zero = lo <= 0.0 <= hi
            st.markdown(
                f"The best ML model ({o4['dev_selected_ml']}) scored {diff:+.3f} PR-AUC against the simple chart "
                f"({o4['dev_selected_spc']}), with a 95% uncertainty range of {lo:+.3f} to {hi:+.3f}. "
                + (
                    "That range includes zero, so there is **no reliable difference**."
                    if spans_zero
                    else "That range excludes zero, so the difference is **statistically reliable**."
                )
            )

    st.caption(
        f"Measured once on the {n_holdout} most recent runs ({n_fails} failed), which were locked away until every "
        f"setting was frozen. Chosen detector: {primary_id}, "
        f"{DETECTOR_CATALOG.get(primary_id, {}).get('short', primary_id)}. "
        "These are offline benchmark results on a public dataset, not measured factory yield or cost savings."
    )


def render_ewma_view(
    scored_df: pd.DataFrame,
    bundle: LoadedBundle,
) -> None:
    monitored = bundle.spc_model.monitored_sensors
    lam = bundle.spc_model.ewma_lambda
    ewma_l = bundle.spc_model.ewma_l
    render_view_intro(
        "Single-sensor trends",
        "Each chart follows **one sensor** across production runs in time order. If its smoothed trend leaves the "
        "band it normally stays in, something about the process may have changed.",
        [
            "**Blue line**: the sensor's smoothed trend, where recent runs count more.",
            "**Red dashed lines**: the normal band, learned from early good runs. Outside the band is unusual.",
            "**Grey dotted line**: the sensor's normal average.",
            "**Markers**: runs that broke a stability rule (× a sudden spike, ▲ a shift, ◇ a slower drift).",
            "Sensor names such as `f059` are anonymised in this dataset, so we cannot say what each one measures.",
        ],
        f"During training (blocks D1–D4), all {len(bundle.quality.retained_columns)} cleaned sensors were screened "
        "by rank-biserial correlation between passing and failing runs, with Benjamini-Hochberg correction. The "
        f"top {len(monitored)} are monitored with an EWMA chart (λ = {lam:g}, L = {ewma_l:g}) and four Western "
        "Electric run rules, using the centre line μ₀ and moving-range sigma σ₀ from passing training runs. "
        "A missing reading leaves the EWMA unchanged. Detectors E02 and E03 are built on these sensors.",
    )

    if not monitored:
        st.info("This model has no individually tracked sensors, so there is nothing to chart here.")
        return

    # Controls on top, chart at full width, supporting numbers underneath.
    ctrl_sel, ctrl_raw = st.columns([2, 1], vertical_alignment="bottom")
    chart_area = st.container()
    detail_left, detail_right = st.columns(2)

    with ctrl_sel:
        def _format_sensor(s_name: str) -> str:
            idx = monitored.index(s_name) + 1
            sp = bundle.spc_model.sensor_specs[s_name]
            return f"#{idx} · {s_name} (separation {abs(sp.rank_biserial):.2f})"

        selected_sensor = st.selectbox(
            "Sensor to inspect (strongest fault signal first)",
            monitored,
            index=0,
            format_func=_format_sensor,
            help=(
                "Sensors are ordered by how well they separated failed from passed runs in training. "
                "Separation runs from 0 (none) to 1 (perfect)."
            ),
        )
        spec = bundle.spc_model.sensor_specs[selected_sensor]

    with ctrl_raw:
        show_raw_trace = st.checkbox(
            "Also show the raw readings",
            value=False,
            help="Draws the unsmoothed run-by-run values behind the trend line.",
        )

    with detail_left:
        # Values keep the .4f format: tests compare them with the package output.
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric(
                "Normal average (μ₀)",
                f"{spec.mu0:.4f}",
                help="Average reading over passing training runs, after trimming extreme outliers.",
            )
        with m_c2:
            st.metric(
                "Normal variation (σ₀)",
                f"{spec.sigma0:.4f}",
                help="Typical run-to-run spread, estimated from the average moving range (MR / 1.128).",
            )

        with st.container(border=True):
            st.markdown(
                f"**Why sensor {selected_sensor} is tracked**\n\n"
                f"- Separation of failed from passed runs (rank-biserial): `{spec.rank_biserial:+.4f}`\n"
                f"- Chance the separation is a fluke (adjusted p-value): `{spec.bh_adjusted_p_value:.2e}`\n"
                f"- Missing readings in training: `{spec.training_missing_fraction * 100:.2f}%`\n"
                f"- Near-duplicate sensors: `{', '.join(spec.group_members) if spec.group_members else 'none'}`"
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
        lam=lam,
        L=ewma_l,
    )
    rules_res = evaluate_run_rules(sensor_vals, spec.mu0, spec.sigma0)

    n_runs = len(sensor_vals)
    run_indices = np.arange(n_runs)

    r1_idx = np.flatnonzero(rules_res["rule_1_beyond_3sigma"])
    r2_idx = np.flatnonzero(rules_res["rule_2_of_3_beyond_2sigma"])
    r3_idx = np.flatnonzero(rules_res["rule_4_of_5_beyond_1sigma"])
    r4_idx = np.flatnonzero(rules_res["rule_8_on_one_side"])

    with detail_right:
        st.markdown("**Stability rules broken** (Western Electric run rules)")
        rules_summary_df = pd.DataFrame(
            [
                {"Rule": "1 reading beyond 3σ", "Suggests": "Sudden spike", "Runs": len(r1_idx)},
                {"Rule": "2 of 3 beyond 2σ", "Suggests": "Step shift", "Runs": len(r2_idx)},
                {"Rule": "4 of 5 beyond 1σ", "Suggests": "Sustained drift", "Runs": len(r3_idx)},
                {"Rule": "8 in a row on one side", "Suggests": "Long-term offset", "Runs": len(r4_idx)},
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
                name="Raw reading",
                line=dict(color="rgba(148, 163, 184, 0.45)", width=1),
            )
        )

    # Centre Line & Limits
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=ewma_res["z"],
            mode="lines",
            name="Smoothed trend (EWMA)",
            line=dict(color="#0284c7", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=ewma_res["ucl"],
            mode="lines",
            name="Upper limit of normal band",
            line=dict(color="#dc2626", dash="dash", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=ewma_res["lcl"],
            mode="lines",
            name="Lower limit of normal band",
            line=dict(color="#dc2626", dash="dash", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=[spec.mu0] * n_runs,
            mode="lines",
            name="Normal average (μ₀)",
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
                name=f"Spike: beyond 3σ ({len(r1_idx)})",
                marker=dict(color="#dc2626", symbol="x", size=9),
            )
        )

    if len(r2_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=r2_idx,
                y=ewma_res["z"][r2_idx],
                mode="markers",
                name=f"Shift: 2 of 3 beyond 2σ ({len(r2_idx)})",
                marker=dict(color="#ea580c", symbol="triangle-up", size=8),
            )
        )

    if len(r3_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=r3_idx,
                y=ewma_res["z"][r3_idx],
                mode="markers",
                name=f"Drift: 4 of 5 beyond 1σ ({len(r3_idx)})",
                marker=dict(color="#ca8a04", symbol="diamond-open", size=7),
            )
        )

    fig.update_layout(
        title=f"Sensor {selected_sensor}: smoothed trend against its normal band",
        xaxis_title="Production run (time order)",
        yaxis_title=f"Sensor {selected_sensor} reading",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=50, b=20),
        height=480,
    )
    with chart_area:
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Hover over the chart for exact values; click a legend entry to hide or show it. "
            "Where a reading is missing, the trend line stays flat rather than guessing."
        )

    with st.expander(f"Compare all {len(monitored)} tracked sensors"):
        all_specs_rows = []
        for idx, s_id in enumerate(monitored, start=1):
            sp = bundle.spc_model.sensor_specs[s_id]
            all_specs_rows.append(
                {
                    "Rank": f"#{idx}",
                    "Sensor": s_id,
                    "Normal average (μ₀)": round(sp.mu0, 4),
                    "Normal variation (σ₀)": round(sp.sigma0, 4),
                    "Separation (rank-biserial)": round(sp.rank_biserial, 4),
                    "Adjusted p-value": f"{sp.bh_adjusted_p_value:.3e}",
                    "Missing in training (%)": round(sp.training_missing_fraction * 100, 2),
                    "Near-duplicate sensors": ", ".join(sp.group_members) if sp.group_members else "—",
                }
            )
        st.dataframe(pd.DataFrame(all_specs_rows), width="stretch", hide_index=True)


def render_mspc_view(
    scored_df: pd.DataFrame,
    bundle: LoadedBundle,
) -> None:
    mspc = bundle.mspc_model
    n_retained = len(bundle.quality.retained_columns)
    render_view_intro(
        "All sensors at once",
        f"Instead of one sensor at a time, these charts boil **all {n_retained} sensors down to a single score per "
        "run**. A tall spike means that run looked very different from normal good runs.",
        [
            "**Left chart, overall shift (T²)**: many sensors moved together in their usual pattern, only further "
            "than normal. Typical of a broad change, such as equipment drifting.",
            "**Right chart, broken pattern (SPE)**: sensors stopped relating to each other the way they normally "
            "do. Typical of one sensor or sub-system misbehaving.",
            "**Orange dotted line**: our alarm line, tuned to raise about 5 alarms per 100 runs.",
            "**Red dashed line**: the textbook limit. It assumes well-behaved data and fires far too often here, "
            "which is why we do not use it.",
            "**Red dots**: runs above our alarm line.",
        ],
        f"A PCA model with {mspc.n_components} components is fitted on passing training runs only and explains "
        f"{mspc.cumulative_variance_explained * 100:.1f}% of their variance. Hotelling T² (detector E04) is the "
        "Mahalanobis distance inside that subspace; SPE, also called the Q statistic (detector E05), is the squared "
        "residual outside it. The textbook limits are the 99% Gaussian-theory limits; our alarm lines are empirical "
        "budget-5 cutoffs calibrated on block D5.",
    )

    use_log_scale = st.checkbox(
        "Log scale (recommended)",
        value=True,
        help="Scores here span several orders of magnitude. A log scale keeps both the everyday values and the extreme spikes visible.",
    )

    n_runs = len(scored_df)
    run_indices = np.arange(n_runs)

    t2_vals = scored_df["t2"].to_numpy()
    spe_vals = scored_df["spe"].to_numpy()

    t2_lim_theo = mspc.t2_limit_99
    spe_lim_theo = mspc.spe_limit_99
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
                name="T² score",
                line=dict(color="#16a34a", width=1.5),
            )
        )
        if len(t2_alarm_idx) > 0:
            fig_t2.add_trace(
                go.Scatter(
                    x=t2_alarm_idx,
                    y=t2_vals[t2_alarm_idx],
                    mode="markers",
                    name=f"Above our alarm line ({len(t2_alarm_idx)})",
                    marker=dict(color="#dc2626", size=7, symbol="circle"),
                )
            )
        fig_t2.add_trace(
            go.Scatter(
                x=[0, n_runs - 1],
                y=[t2_lim_theo, t2_lim_theo],
                mode="lines",
                name=f"Textbook 99% limit ({t2_lim_theo:.1f})",
                line=dict(color="#dc2626", dash="dash"),
            )
        )
        fig_t2.add_trace(
            go.Scatter(
                x=[0, n_runs - 1],
                y=[t2_thresh_b5, t2_thresh_b5],
                mode="lines",
                name=f"Our alarm line ({t2_thresh_b5:.1f})",
                line=dict(color="#ea580c", dash="dot", width=2),
            )
        )
        fig_t2.update_layout(
            title="Overall shift (T²) for each run",
            xaxis_title="Production run (time order)",
            yaxis_title="T² score (log scale)" if use_log_scale else "T² score",
            yaxis_type="log" if use_log_scale else "linear",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=50, b=20),
            height=440,
        )
        st.plotly_chart(fig_t2, width="stretch")

        m1, m2 = st.columns(2)
        m1.metric("Runs above our alarm line", int(len(t2_alarm_idx)), delta=f"{len(t2_alarm_idx) / n_runs * 100:.2f} / 100 runs", delta_color="off", delta_arrow="off")
        m2.metric("Runs above textbook limit", int(np.sum(t2_vals > t2_lim_theo)), delta=f"Limit = {t2_lim_theo:.2f}", delta_color="off", delta_arrow="off")

    with col2:
        fig_spe = go.Figure()
        fig_spe.add_trace(
            go.Scatter(
                x=run_indices,
                y=spe_vals,
                mode="lines",
                name="SPE score",
                line=dict(color="#7c3aed", width=1.5),
            )
        )
        if len(spe_alarm_idx) > 0:
            fig_spe.add_trace(
                go.Scatter(
                    x=spe_alarm_idx,
                    y=spe_vals[spe_alarm_idx],
                    mode="markers",
                    name=f"Above our alarm line ({len(spe_alarm_idx)})",
                    marker=dict(color="#dc2626", size=7, symbol="circle"),
                )
            )
        fig_spe.add_trace(
            go.Scatter(
                x=[0, n_runs - 1],
                y=[spe_lim_theo, spe_lim_theo],
                mode="lines",
                name=f"Textbook 99% limit ({spe_lim_theo:.1f})",
                line=dict(color="#dc2626", dash="dash"),
            )
        )
        fig_spe.add_trace(
            go.Scatter(
                x=[0, n_runs - 1],
                y=[spe_thresh_b5, spe_thresh_b5],
                mode="lines",
                name=f"Our alarm line ({spe_thresh_b5:.1f})",
                line=dict(color="#ea580c", dash="dot", width=2),
            )
        )
        fig_spe.update_layout(
            title="Broken pattern (SPE) for each run",
            xaxis_title="Production run (time order)",
            yaxis_title="SPE score (log scale)" if use_log_scale else "SPE score",
            yaxis_type="log" if use_log_scale else "linear",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=50, b=20),
            height=440,
        )
        st.plotly_chart(fig_spe, width="stretch")

        s1, s2 = st.columns(2)
        s1.metric("Runs above our alarm line", int(len(spe_alarm_idx)), delta=f"{len(spe_alarm_idx) / n_runs * 100:.2f} / 100 runs", delta_color="off", delta_arrow="off")
        s2.metric("Runs above textbook limit", int(np.sum(spe_vals > spe_lim_theo)), delta=f"Limit = {spe_lim_theo:.2f}", delta_color="off", delta_arrow="off")


def render_drilldown_view(
    scored_df: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    bundle: LoadedBundle,
) -> None:
    render_view_intro(
        "Why was a run flagged?",
        "An alarm is only useful if it says where to look. Pick a flagged run to see **which sensors looked most "
        "unusual**. Two independent methods each name their top 5.",
        [
            "**Left, distance from normal**: a longer bar means that sensor did more to make the run look abnormal.",
            "**Right, what the ML model reacted to**: red bars pushed its prediction towards 'fail', green bars "
            "towards 'pass'.",
            "Sensors named by **both** methods are the best place to start investigating.",
            "These rankings point to where the data looked odd. They do not prove what caused a failure.",
        ],
        "Left: per-sensor contributions to Hotelling T² and SPE (MSPC), measured against passing training runs. "
        "Right: TreeSHAP values for the XGBoost model (E08), which are exact additive contributions to the "
        "predicted log-odds of failure. Near-duplicate sensors removed during cleaning are listed beside the "
        "sensor that represents them.",
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
        st.info("No runs are above the alarm cutoff, so the first 10 runs are shown instead.")

    run_options = list(flagged_df["run_id"].astype(str))
    run_meta_lookup = flagged_df.set_index(flagged_df["run_id"].astype(str)).to_dict(orient="index")

    def _format_run_option(r_id: str) -> str:
        meta = run_meta_lookup.get(r_id, {})
        blk = meta.get("block", "unknown")
        e02_val = meta.get("e02_score", 0.0)
        lbl_str = ""
        if r_id in label_map:
            lbl_str = " · failed final test" if label_map[r_id] == 1 else " · passed final test"
        return f"{r_id} · alarm score {e02_val:.2f} · block {blk}{lbl_str}"

    selected_run_id = st.selectbox(
        f"Run to inspect ({len(run_options)} available, in time order)",
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
        "Failed"
        if label_map.get(selected_run_id) == 1
        else ("Passed" if label_map.get(selected_run_id) == 0 else "Unknown")
    )
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric("Run", selected_run_id, delta=f"Block {run_row.get('block', 'N/A')}", delta_color="off", delta_arrow="off")
    rc2.metric("Final test result", gt_status, delta=str(run_row.get('timestamp', 'N/A'))[:16], delta_color="off", delta_arrow="off")
    rc3.metric("Alarm score (E02)", f"{float(run_row.get('e02_score', 0.0)):.2f}", delta=f"Alarm above {tau_b5:.2f}", delta_color="off", delta_arrow="off")
    rc4.metric("All-sensor score (E06)", f"{float(run_row.get('mspc_index', 0.0)):.2f}", delta=f"SPE {float(run_row.get('spe', 0.0)):.1f}", delta_color="off", delta_arrow="off")
    rc5.metric("Missing readings", f"{int(run_row.get('n_missing', 0))} of {len(bundle.quality.input_columns)}", delta="Filled with typical values", delta_color="off", delta_arrow="off")

    overlap = explanation["overlap"]["top_k_common_sensors"]
    overlap_text = ", ".join(overlap) if overlap else "none for this run (the two methods point at different sensors)"
    st.info(
        f"**Sensors named by both methods:** {overlap_text}  \n"
        f"**Caution:** {SMEARING_CAVEAT}"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distance from normal (MSPC)")
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
            xaxis_title="Contribution to the run looking abnormal",
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
                    "Contribution": round(float(item["contribution"]), 3),
                    "From broken pattern (SPE)": round(float(item.get("spe_contribution", 0.0)), 2),
                    "From overall shift (T²)": round(float(item.get("t2_contribution", 0.0)), 2),
                    "Near-duplicate sensors": (
                        ", ".join(item["group_members"]) if item["group_members"] else "—"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(mspc_table_rows), width="stretch", hide_index=True)

    with col2:
        st.subheader("What the ML model reacted to (SHAP)")
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
            xaxis_title="Push on predicted failure risk (red raises, green lowers)",
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
                    "SHAP value": f"{float(item['shap_value']):+.4f}",
                    "Effect": "Raises failure risk" if item["shap_value"] > 0 else "Lowers failure risk",
                    "Near-duplicate sensors": (
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
    render_view_intro(
        "Results: how well does it work?",
        "Every detector gets the same allowance of 2, 5 or 10 alarms per 100 runs, so they can be compared fairly. "
        "Two questions matter: of the runs that really failed, **how many were flagged** (recall)? And of the alarms "
        "raised, **how many were real** (precision)?",
        [
            "**Comparison table and chart**: every loaded run, including the runs the detectors learned from. "
            "Use it to compare detectors, not as a final score.",
            "**Final test scorecard** (further down): the fair test, on real runs no detector had seen.",
            "Higher recall and precision are better. Alarms per 100 runs should sit near the budget.",
            "★ marks the chosen detector. Each detector is explained in the sidebar.",
        ],
        "Cutoffs (τ) for budgets 2, 5 and 10 were calibrated on development block D5 and frozen before the holdout "
        "was opened. An alarm is raised when score > cutoff. Budget 5 is the primary operating point.",
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
        "Alarm budget to show",
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
                "Type": det_info["family"],
                "Alarm budget": f"{b} per 100",
                "Score cutoff": round(float(tau), 4),
                "Alarms raised": alarms_count,
                "Alarms per 100 runs": round(realised_rate, 2),
            }

            if has_labels and y_true is not None:
                flags = merged[flag_col].to_numpy()
                true_alarms = int(np.sum((flags == 1) & (y_true == 1)))
                false_alarms = int(np.sum((flags == 1) & (y_true == 0)))
                recall = float(true_alarms / n_fails) if n_fails and n_fails > 0 else 0.0
                precision = float(true_alarms / alarms_count) if alarms_count > 0 else 0.0
                row["Real failures caught"] = true_alarms
                row["False alarms"] = false_alarms
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
            go.Bar(name="Recall: % of failed runs flagged", x=b5_names, y=b5_recalls, marker_color="#0f766e")
        )
        fig_cmp.add_trace(
            go.Bar(name="Alarms raised per 100 runs", x=b5_names, y=b5_rates, marker_color="#f59e0b")
        )
        fig_cmp.update_layout(
            title="Detectors compared at the main budget (5 alarms per 100 runs)",
            barmode="group",
            yaxis_title="Percent, or alarms per 100 runs",
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=50, b=20),
            height=380,
        )
        st.plotly_chart(fig_cmp, width="stretch")

    # Frozen holdout verdicts and drift summary, shown once the final evaluation exists
    h_data = load_json_report("reports/holdout_metrics.json")
    if h_data is not None:
        n_hold = int(h_data["n_holdout_runs"])
        n_hold_fails = int(h_data["n_holdout_fails"])
        prevalence = float(h_data["holdout_fail_prevalence"])
        st.divider()
        st.subheader(f"Final test: {n_hold} unseen runs from the real SECOM data")
        st.markdown(
            f"The most recent {n_hold} runs ({n_hold_fails} failed, {prevalence * 100:.1f}%) were locked away until "
            "every setting was frozen, then scored exactly once. The targets below were written down before any "
            "detector was built, and misses are reported as measured."
        )
        render_headline_findings(h_data)

        obj = h_data.get("objectives", {})
        o1 = obj.get("O1_ranking_pr_auc", {})
        o2 = obj.get("O2_recall_at_5_far", {})
        o3 = obj.get("O3_alarm_budget_stability", {})
        o4 = obj.get("O4_spc_vs_ml_comparison", {})
        verdict_text = {"PASS": "Met", "MISS": "Missed", "REPORTED": "Reported"}

        o1_ci = o1.get("wider_95_ci", [0.0384, 0.1514])
        o2_ci = o2.get("wider_95_ci", [0.0, 0.1765])
        o4_ci = o4.get("wider_95_ci_dev_pair", [-0.0470, 0.0910])
        scorecard_rows = [
            {
                "Goal": "O1: rank failed runs above good ones (PR-AUC)",
                "Target set in advance": f"At least 2× the failure rate ({o1.get('target_min_pr_auc_2x_prevalence', 0.1224):.4f})",
                "Measured": f"{o1.get('actual_pr_auc', 0.0724):.4f} (random guessing: {o1.get('chance_prevalence', prevalence):.4f})",
                "95% uncertainty range": f"{o1_ci[0]:.4f} to {o1_ci[1]:.4f}",
                "Verdict": verdict_text.get(o1.get("verdict", "MISS"), "Missed"),
                "What it means": "No detector beat chance: the 95% range includes the 6.12% failure rate, well short of the 2× target.",
            },
            {
                "Goal": "O2: failed runs caught at 5 false alarms per 100 good runs (recall)",
                "Target set in advance": f"At least {o2.get('target_min_recall', 0.2) * 100:.0f}%",
                "Measured": (
                    f"{o2.get('actual_recall_at_5_far', 0.0417) * 100:.1f}% "
                    f"({o2.get('recall_at_10_far', 0.125) * 100:.1f}% at 10 per 100)"
                ),
                "95% uncertainty range": f"{o2_ci[0] * 100:.1f}% to {o2_ci[1] * 100:.1f}%",
                "Verdict": verdict_text.get(o2.get("verdict", "MISS"), "Missed"),
                "What it means": f"1 of 24 fails caught — indistinguishable from random guessing ({o2.get('chance_recall', 0.05) * 100:.0f}%).",
            },
            {
                "Goal": "O3: alarm workload stays near the budget",
                "Target set in advance": "Between 2 and 8 alarms per 100 runs",
                "Measured": f"{o3.get('realised_alarms_per_100', 3.06):.2f} per 100 ({o3.get('alarms', 12)} alarms: 0 TP, 12 FP)",
                "95% uncertainty range": "Exact count",
                "Verdict": verdict_text.get(o3.get("verdict", "PASS"), "Met"),
                "What it means": "The frozen cutoff kept the alarm count within [2, 8] per 100, but all 12 alarms were false alarms (0 TP, 12 FP).",
            },
            {
                "Goal": "O4: machine learning vs. classic charts (PR-AUC difference)",
                "Target set in advance": "No target; report the difference and its range",
                "Measured": f"{o4.get('diff_ml_minus_spc_dev_pair', 0.0153):+.4f} (XGBoost E08 minus SPC E02)",
                "95% uncertainty range": f"{o4_ci[0]:+.4f} to {o4_ci[1]:+.4f}",
                "Verdict": verdict_text.get(o4.get("verdict", "REPORTED"), "Reported"),
                "What it means": "The range includes zero: no reliable advantage for machine learning.",
            },
            {
                "Goal": "O5: sensor ranking finds a planted fault",
                "Target set in advance": "Planted 5σ single-sensor fault in the top 3 at least 90% of the time",
                "Measured": "100.0% in top 3 (100% top 1 by MSPC, 96% by SPE)",
                "95% uncertainty range": "100 synthetic trials",
                "Verdict": "Met",
                "What it means": "When one sensor clearly misbehaves, the ranking points at it.",
            },
            {
                "Goal": "O6 and O7: same answer every time, on every interface",
                "Target set in advance": "Score difference ≤ 1e-6 on re-run; ≤ 1e-9 between command line and API",
                "Measured": "0.0 difference; 63.8 ms per request (warm, 95th percentile)",
                "95% uncertainty range": "100 benchmark trials",
                "Verdict": "Met",
                "What it means": "Training, command line, API and this dashboard share one scoring path.",
            },
        ]
        scorecard_columns = ["Goal", "Verdict", "Measured", "Target set in advance", "What it means", "95% uncertainty range"]
        st.dataframe(pd.DataFrame(scorecard_rows)[scorecard_columns], width="stretch", hide_index=True)
        st.caption(
            "Source: `reports/holdout_metrics.json`. Predictions were saved and hashed before the test labels were "
            "opened. Uncertainty ranges are the wider of two bootstrap methods."
        )

        d_data = load_json_report("reports/drift/drift_summary.json")
        if d_data is not None:
            n_cols = int(d_data.get("number_of_columns", 0))
            n_drifted = int(d_data.get("number_of_drifted_columns", 0))
            n_missing_shift = d_data.get("missing_rate_summary", {}).get("columns_with_missingness_change_gt_5pct")
            with st.container(border=True):
                st.markdown(
                    "**A likely reason for the weak results: the process changed over time**\n\n"
                    f"Comparing the {d_data.get('reference_count', 0)} training runs with the "
                    f"{d_data.get('current_count', 0)} unseen runs, **{n_drifted} of {n_cols} sensors "
                    f"({float(d_data.get('share_of_drifted_columns', 0.0)) * 100:.1f}%) behaved measurably "
                    "differently**"
                    + (
                        f", and {n_missing_shift} sensors changed how often their readings were missing by more "
                        "than 5 percentage points"
                        if n_missing_shift is not None
                        else ""
                    )
                    + ". A detector tuned on the earlier period is being asked about equipment that has moved on. "
                    "This is an observed pattern, not proof of cause."
                )
                st.caption("Source: `reports/drift/drift_summary.json` (Evidently data-drift preset, one statistical test per sensor).")


def render_project_overview_view(bundle: LoadedBundle) -> None:
    """Render the plain-language landing page: the problem, the method, the findings and a glossary."""
    q = bundle.quality
    n_raw = len(q.input_columns)
    n_after_missing = n_raw - len(q.dropped_high_missing)
    n_after_constant = n_after_missing - len(q.dropped_constant)
    n_retained = len(q.retained_columns)
    budget = float(bundle.policy.get("primary_alarm_budget_per_100", 5.0))

    st.header("Start here: what is YieldGuard?")

    st.subheader("The problem")
    st.markdown(
        "Computer chips are built on silicon discs called **wafers**, and a wafer is only fully tested at the end "
        "of production. During every production **run**, hundreds of sensors record what the equipment is doing. "
        "If those readings could warn that a run is going wrong, engineers could step in before more wafers are "
        "wasted.\n\n"
        "The catch is attention. A system that raises an alarm on most runs gets ignored. So YieldGuard works to an "
        f"**alarm budget** of about {budget:g} alarms per 100 runs and asks one question: "
        "*within that budget, how many of the truly bad runs can we catch?*"
    )

    st.subheader("How it works")
    steps = [
        (
            "1. Learn what normal looks like",
            f"Use the earliest runs to learn each sensor's usual range. Drop sensors that are mostly empty, never "
            f"change or duplicate another one ({n_raw} sensors become {n_retained}).",
        ),
        (
            "2. Score every run",
            "Eight detectors each give a run an 'unusualness' score. Some are classic factory control charts "
            "(SPC), some are machine-learning models.",
        ),
        (
            "3. Set the alarm line",
            f"Pick each detector's cutoff so it raises about {budget:g} alarms per 100 runs, using a separate "
            "block of runs kept for that purpose.",
        ),
        (
            "4. Test once on unseen runs",
            "The most recent quarter of the runs stayed locked away until everything was frozen, then was scored "
            "a single time. Runs are split by time, never shuffled, because a real system must predict the future "
            "from the past.",
        ),
    ]
    for col, (step_title, step_text) in zip(st.columns(4), steps):
        with col, st.container(border=True):
            st.markdown(f"**{step_title}**")
            st.markdown(step_text)

    holdout = load_json_report("reports/holdout_metrics.json")
    if holdout is not None:
        st.subheader("What we found")
        render_headline_findings(holdout)
        st.markdown(
            "**Bottom line:** on this dataset, no detector predicted failing runs much better than chance. What the "
            "project demonstrates is a fair, leak-free way to compare detectors under a fixed alarm budget, with "
            "every missed target reported as measured. The **Results** page has the full numbers and a likely "
            "reason: the process changed over time."
        )

    st.subheader("Where to look next")
    st.markdown(
        "Use the menu on the left. Each page answers one question:\n\n"
        "| Page | Question it answers |\n"
        "|---|---|\n"
        "| **Single-sensor trends** | Is one particular sensor drifting away from its normal range? |\n"
        "| **All sensors at once** | Did a run look abnormal overall, even if no single sensor stood out? |\n"
        "| **Why was a run flagged?** | For a flagged run, which sensors made it stand out? |\n"
        "| **Results** | How do the eight detectors compare, and did they meet their targets? |"
    )

    st.subheader("Key terms")
    st.markdown(
        "| Term | Meaning |\n|---|---|\n" + "\n".join(f"| **{term}** | {meaning} |" for term, meaning in GLOSSARY)
    )

    st.subheader("Technical reference")
    with st.expander("How the runs were split by time (D1–D5 and holdout)"):
        st.markdown(
            "Equipment goes through maintenance, recipe changes and sensor drift. Shuffling rows or using standard "
            "k-fold cross-validation would leak future equipment states into training, so the 1,567 SECOM runs are "
            "split strictly by timestamp:"
        )
        split_rows = [
            {"Block": "D1", "Runs": 235, "Fails": "32 (13.62%)", "Role": "Training (folds 1, 2, 3)"},
            {"Block": "D2", "Runs": 235, "Fails": "26 (11.06%)", "Role": "Training (folds 1, 2, 3)"},
            {"Block": "D3", "Runs": 235, "Fails": "8 (3.40%)", "Role": "Fold 1 validation; training for folds 2 and 3"},
            {"Block": "D4", "Runs": 235, "Fails": "10 (4.26%)", "Role": "Fold 2 validation; training for fold 3"},
            {"Block": "D5", "Runs": 235, "Fails": "4 (1.70%)", "Role": "Fold 3 validation and alarm-cutoff calibration"},
            {"Block": "Holdout", "Runs": 392, "Fails": "24 (6.12%)", "Role": "Final test, opened once after the freeze"},
        ]
        st.dataframe(pd.DataFrame(split_rows), width="stretch", hide_index=True)

    with st.expander(f"How the sensors were cleaned ({n_raw} to {n_retained})"):
        st.markdown(
            f"Every rule is fitted on the {q.n_train_runs} training runs only and then applied identically to batch "
            "files, API requests and this dashboard."
        )
        clean_rows = [
            {"Step": "1. Load", "Rule": "Check all sensor columns are numeric; reject infinite values", "Sensors left": n_raw},
            {
                "Step": "2. Remove mostly-empty sensors",
                "Rule": f"More than {q.max_missing_fraction * 100:.0f}% of training readings missing",
                "Sensors left": n_after_missing,
            },
            {
                "Step": "3. Remove sensors that never change",
                "Rule": f"One value makes up at least {q.max_constant_dominant_fraction * 100:.0f}% of training readings",
                "Sensors left": n_after_constant,
            },
            {
                "Step": "4. Remove near-duplicates",
                "Rule": f"Correlation |r| of at least {q.collinearity_threshold:g} with a kept sensor; duplicates are recorded as its group",
                "Sensors left": n_retained,
            },
            {
                "Step": "5. Fill gaps and rescale",
                "Rule": "Missing readings filled with the training median; scaled using passing training runs",
                "Sensors left": n_retained,
            },
        ]
        st.dataframe(pd.DataFrame(clean_rows), width="stretch", hide_index=True)

    with st.expander("The eight detectors and their alarm cutoffs"):
        det_rows = []
        thresholds = bundle.policy.get("thresholds", {})
        for det_id, info in DETECTOR_CATALOG.items():
            t_dict = thresholds.get(det_id, {})
            det_rows.append(
                {
                    "ID": det_id,
                    "Detector": info["short"],
                    "Type": info["family"],
                    "In plain words": DETECTOR_PLAIN[det_id],
                    "Cutoff at 2 per 100": round(float(t_dict.get("budget_2", 0.0)), 4),
                    "Cutoff at 5 per 100": round(float(t_dict.get("budget_5", 0.0)), 4),
                    "Cutoff at 10 per 100": round(float(t_dict.get("budget_10", 0.0)), 4),
                    "Method": info["desc"],
                }
            )
        st.dataframe(pd.DataFrame(det_rows), width="stretch", hide_index=True)


WAFER_DETECTOR_NAMES: dict[str, str] = {
    "E01_v2": "E01_v2 — Chance Baseline",
    "E02_v2": "E02_v2 — Univariate SPC (Individuals)",
    "E03_v2": "E03_v2 — Univariate SPC (EWMA)",
    "E04_v2": "E04_v2 — Hotelling T² (MSPC)",
    "E05_v2": "E05_v2 — SPE / Q-Residual (Best SPC)",
    "E06_v2": "E06_v2 — Combined MSPC (T² + SPE)",
    "E07_v2": "E07_v2 — Isolation Forest",
    "E08_v2": "E08_v2 — Class-Weighted XGBoost (Primary)",
    "E09_v2": "E09_v2 — L2 Logistic Regression",
}


@st.cache_data
def load_wafer_dashboard_data(use_synthetic: bool = False) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    """Load Wafer Etch confirmatory metrics and sample sensor runs with graceful fallback."""
    metrics_path = Path("reports/v0_2/wafer_confirmatory_metrics.json")
    metrics = load_json_report(str(metrics_path))

    runs_path = Path("data/processed/wafer_runs.parquet")
    if not use_synthetic and runs_path.exists():
        try:
            df_runs = pd.read_parquet(runs_path)
            return metrics, df_runs
        except (OSError, ValueError):
            pass

    # Synthetic demo data fallback (20 runs with 152 time steps)
    rng = np.random.default_rng(42)
    synthetic_rows = []
    sensor_cols = [f"s{i:03d}" for i in range(152)]
    for i in range(20):
        is_abnormal = 1 if i in (3, 7, 12, 18) else -1
        base_signal = np.sin(np.linspace(0, 3 * np.pi, 152))
        noise = rng.normal(0, 0.2, 152)
        if is_abnormal == 1:
            noise[50:90] += rng.uniform(1.5, 3.0, 40)
        row: dict[str, Any] = {
            "run_id": f"wafer_syn_{i:04d}",
            "timestamp": f"2001-01-01 {i:02d}:00:00",
            "label": is_abnormal,
        }
        for s_idx, s_col in enumerate(sensor_cols):
            row[s_col] = float(base_signal[s_idx] + noise[s_idx])
        synthetic_rows.append(row)

    df_runs = pd.DataFrame(synthetic_rows)
    return metrics, df_runs


def render_wafer_etch_view(use_synthetic: bool = False) -> None:
    """Render the Semiconductor Wafer Etch confirmatory benchmark (v0.2 Protocol)."""
    metrics, df_runs = load_wafer_dashboard_data(use_synthetic=use_synthetic)

    summary_text = (
        "Following the exploratory finding on SECOM that equipment sensor drift was the primary driver "
        "of false alarms, YieldGuard Protocol v0.2 was frozen and applied to an independent semiconductor "
        "wafer etch benchmark dataset (7,164 runs, 1 sensor trace of 152 time steps per wafer). "
        "Chronology limitation: Because original fab equipment timestamps are not included in the public UCR archive, "
        "timestamps were synthesized at 10-minute intervals in file order; block order is therefore arbitrary, so this "
        "confirmatory run benchmarks prequential evaluation mechanics rather than real process drift."
    )

    p_summary = metrics.get("prequential_summary", {}) if metrics else {}
    n_eval_runs = p_summary.get("n_evaluated_runs", 6268)
    n_eval_fails = p_summary.get("n_evaluated_fails", 676)
    n_eval_blocks = p_summary.get("n_evaluated_blocks", 7)

    technical_text = (
        f"Protocol v0.2 uses rolling-window prequential evaluation over {n_eval_blocks} sequential evaluation blocks "
        f"comprising {n_eval_runs:,} scored runs and {n_eval_fails:,} real etch faults. Baselines and alarm thresholds are refit "
        "per block on past data. Confirmatory evaluation was executed strictly once on frozen code. "
        "Targets O1_v2–O4_v2 are tracked dynamically from the confirmatory metrics report; targets O5_v2 (calibration fidelity) "
        "and O6_v2 (reproducibility) were not recorded in the metrics report."
    )

    render_view_intro(
        title="Semiconductor Wafer Etch: Confirmatory Benchmark (v0.2)",
        summary=summary_text,
        how_to_read=[
            "**Official Scorecard**: inspect the pre-registered confirmatory targets (O1_v2 to O4_v2).",
            "**Sensor Trace Inspector**: select individual wafer runs to inspect the 152-time-step inline sensor trace compared against the normal mean baseline.",
            "**Detector Comparison**: compare all 9 detectors across the operational alarm budgets (2, 5, 10 alarms per 100 runs).",
        ],
        technical=technical_text,
    )

    if use_synthetic:
        st.info("ℹ️ Running with synthetic wafer demo traces. The benchmark scorecard below reflects the frozen confirmatory metrics report.")

    if metrics is None:
        st.warning("Confirmatory metrics file `reports/v0_2/wafer_confirmatory_metrics.json` not found.")
        return

    # Top KPI Strip for v0.2 Targets (dynamically derived)
    targets = metrics.get("target_outcomes_v0_2", {})
    o1 = targets.get("O1_v2_prequential_pr_auc")
    o2 = targets.get("O2_v2_recall_at_5_far")
    o3 = targets.get("O3_v2_adaptive_alarm_budget")
    o4 = targets.get("O4_v2_spc_vs_ml_comparison")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if o1 and "actual_pr_auc" in o1:
            val_str = f"{float(o1['actual_pr_auc']):.4f}"
            delta_str = str(o1.get("verdict", "N/A"))
            target_str = f"Target ≥ {float(o1.get('target_min_pr_auc_2x_prevalence', 0.2157)):.4f}"
        else:
            val_str = "n/a"
            delta_str = "N/A"
            target_str = "Target n/a"
        st.metric(
            label=f"O1: PR-AUC ({target_str})",
            value=val_str,
            delta=delta_str,
            delta_color="normal",
        )
    with col2:
        if o2 and "actual_recall_at_5_far" in o2:
            val_str = f"{float(o2['actual_recall_at_5_far']) * 100:.1f}%"
            delta_str = str(o2.get("verdict", "N/A"))
            target_str = f"Target ≥ {float(o2.get('target_min_recall', 0.20)) * 100:.0f}%"
        else:
            val_str = "n/a"
            delta_str = "N/A"
            target_str = "Target n/a"
        st.metric(
            label=f"O2: Recall@5% FAR ({target_str})",
            value=val_str,
            delta=delta_str,
            delta_color="normal",
        )
    with col3:
        if o3 and "realised_alarms_per_100" in o3:
            val_str = f"{float(o3['realised_alarms_per_100']):.2f}"
            delta_str = f"{o3.get('verdict', 'N/A')} ({o3.get('true_alarms', 0)} TP, {o3.get('false_alarms', 0)} FP)"
            rng = o3.get("allowed_range_per_100", [2.0, 8.0])
            target_str = f"Target [{rng[0]:.0f}, {rng[1]:.0f}]"
        else:
            val_str = "n/a"
            delta_str = "N/A"
            target_str = "Target n/a"
        st.metric(
            label=f"O3: Alarms/100 ({target_str})",
            value=val_str,
            delta=delta_str,
            delta_color="normal",
        )
    with col4:
        if o4 and "diff_ml_minus_spc" in o4:
            diff_ml_spc = float(o4["diff_ml_minus_spc"])
            val_str = f"{diff_ml_spc:+.4f}"
            delta_str = str(o4.get("verdict", "REPORTED"))
        else:
            val_str = "n/a"
            delta_str = "N/A"
        st.metric(
            label="O4: ML Advantage over SPC",
            value=val_str,
            delta=delta_str,
            delta_color="normal",
        )

    st.divider()

    # Confirmatory Scorecard Table
    st.subheader("1. Confirmatory Scorecard (Protocol v0.2 Pre-Registered Targets)")

    scorecard_data = []

    # O1 row
    if o1:
        o1_ci = o1.get("wider_95_ci")
        ci_str = f"[{o1_ci[0]:.4f}, {o1_ci[1]:.4f}]" if o1_ci and len(o1_ci) == 2 else "n/a"
        scorecard_data.append(
            {
                "Objective": "O1_v2: Prequential PR-AUC",
                "Pre-Registered Target": (
                    f"≥ 2× prevalence ({float(o1['target_min_pr_auc_2x_prevalence']):.4f})"
                    if "target_min_pr_auc_2x_prevalence" in o1 else "n/a"
                ),
                "Measured Outcome": f"{float(o1['actual_pr_auc']):.4f}" if "actual_pr_auc" in o1 else "n/a",
                "95% Confidence Interval": ci_str,
                "Verdict": str(o1.get("verdict", "N/A")),
                "Interpretation": (
                    f"Primary detector ({o1.get('primary_detector', 'E08_v2')}) achieved top fault ranking across {o1.get('n_fails', 0)} actual fails."
                    if "n_fails" in o1 else "No outcome available."
                ),
            }
        )

    # O2 row
    if o2:
        o2_ci = o2.get("wider_95_ci")
        ci_str = f"[{o2_ci[0]*100:.1f}%, {o2_ci[1]*100:.1f}%]" if o2_ci and len(o2_ci) == 2 else "n/a"
        rec_val = float(o2.get("actual_recall_at_5_far", 0.0))
        n_f = o2.get("n_fails", 0)
        tp_est = int(round(rec_val * n_f))
        scorecard_data.append(
            {
                "Objective": "O2_v2: Recall at 5% FAR",
                "Pre-Registered Target": (
                    f"≥ {float(o2['target_min_recall']) * 100:.0f}%"
                    if "target_min_recall" in o2 else "n/a"
                ),
                "Measured Outcome": f"{rec_val * 100:.2f}%" if "actual_recall_at_5_far" in o2 else "n/a",
                "95% Confidence Interval": ci_str,
                "Verdict": str(o2.get("verdict", "N/A")),
                "Interpretation": (
                    f"Caught {tp_est} of {n_f} etch faults ({rec_val * 100:.1f}%) at a fixed 5% false alarm rate."
                    if n_f > 0 else "No outcome available."
                ),
            }
        )

    # O3 row
    if o3:
        n_r = o3.get("n_runs", 0)
        scorecard_data.append(
            {
                "Objective": "O3_v2: Adaptive Alarm Budget Stability",
                "Pre-Registered Target": f"Within [{o3.get('allowed_range_per_100', [2.0, 8.0])[0]:.1f}, {o3.get('allowed_range_per_100', [2.0, 8.0])[1]:.1f}] alarms per 100 runs",
                "Measured Outcome": (
                    f"{float(o3['realised_alarms_per_100']):.2f} alarms per 100 runs"
                    if "realised_alarms_per_100" in o3 else "n/a"
                ),
                "95% Confidence Interval": f"Exact count across {n_r:,} runs" if n_r else "Exact count",
                "Verdict": str(o3.get("verdict", "N/A")),
                "Interpretation": (
                    f"Realised rate was {float(o3.get('realised_alarms_per_100', 0)):.2f}/100 "
                    f"({o3.get('alarms', 0)} alarms: {o3.get('true_alarms', 0)} True Positives, {o3.get('false_alarms', 0)} False Alarms, "
                    f"Precision {float(o3.get('precision', 0))*100:.1f}%)."
                    if "alarms" in o3 else "No outcome available."
                ),
            }
        )

    # O4 row
    if o4:
        o4_ci = o4.get("stratified_95_ci_diff")
        ci_str = f"[{o4_ci[0]:+.4f}, {o4_ci[1]:+.4f}]" if o4_ci and len(o4_ci) == 2 else "n/a"
        diff_val = float(o4.get("diff_ml_minus_spc", 0.0))
        scorecard_data.append(
            {
                "Objective": "O4_v2: Statistical SPC vs ML Comparison",
                "Pre-Registered Target": "Report PR-AUC difference with 95% bootstrap CI",
                "Measured Outcome": (
                    f"Difference = {diff_val:+.4f} ({o4.get('best_ml_detector', 'E08_v2')} vs {o4.get('best_spc_detector', 'E05_v2')})"
                    if "diff_ml_minus_spc" in o4 else "n/a"
                ),
                "95% Confidence Interval": ci_str,
                "Verdict": str(o4.get("verdict", "REPORTED")),
                "Interpretation": (
                    f"Supervised tree ensembles outperformed unsupervised SPC ({diff_val:+.4f} PR-AUC advantage)."
                    if "diff_ml_minus_spc" in o4 else "No outcome available."
                ),
            }
        )

    if scorecard_data:
        st.dataframe(pd.DataFrame(scorecard_data), width="stretch", hide_index=True)
    else:
        st.info("No confirmatory targets loaded.")

    st.divider()

    # Section 2: Interactive Wafer Run Sensor Trace Inspector
    st.subheader("2. Interactive Wafer Run Sensor Trace Inspector")
    st.markdown(
        "Each wafer run records one sensor trace of 152 sequential time steps during etch processing. "
        "Select a wafer run below to view its inline sensor profile against the normal baseline."
    )

    sensor_cols = [c for c in df_runs.columns if c.startswith("s") and len(c) == 4]
    if sensor_cols:
        col_sel1, col_sel2 = st.columns([2, 3])
        with col_sel1:
            filter_option = st.radio(
                "Filter wafer runs by status:",
                ["All", "Abnormal / Fail only (+1)", "Normal / Pass only (-1)"],
                horizontal=True,
            )

        filtered_df = df_runs
        if filter_option == "Abnormal / Fail only (+1)":
            filtered_df = df_runs[df_runs["label"] == 1]
        elif filter_option == "Normal / Pass only (-1)":
            filtered_df = df_runs[df_runs["label"] == -1]

        sample_options = filtered_df["run_id"].head(200).tolist()
        with col_sel2:
            selected_run_id = st.selectbox(
                "Select Wafer Run ID:",
                sample_options,
                index=0 if len(sample_options) > 0 else None,
            )

        if selected_run_id:
            run_row = df_runs[df_runs["run_id"] == selected_run_id].iloc[0]
            run_label = int(run_row["label"])
            status_badge = (
                '<span class="yg-badge yg-badge-amber">ABNORMAL / ETCH FAULT (+1)</span>'
                if run_label == 1
                else '<span class="yg-badge yg-badge-teal">NORMAL / PASS (-1)</span>'
            )
            st.markdown(
                f"**Run:** `{selected_run_id}` &nbsp;|&nbsp; **Timestamp:** `{run_row['timestamp']}` &nbsp;|&nbsp; {status_badge}",
                unsafe_allow_html=True,
            )

            # Compute normal baseline mean across normal runs
            normal_runs = df_runs[df_runs["label"] == -1]
            if len(normal_runs) > 0:
                baseline_mean = normal_runs[sensor_cols].mean().to_numpy()
            else:
                baseline_mean = np.zeros(len(sensor_cols))

            run_values = run_row[sensor_cols].to_numpy().astype(float)
            x_steps = np.arange(len(sensor_cols))

            fig_trace = go.Figure()
            fig_trace.add_trace(
                go.Scatter(
                    x=x_steps,
                    y=baseline_mean,
                    mode="lines",
                    name="Normal Baseline Mean (-1)",
                    line=dict(color="#94a3b8", width=1.5, dash="dash"),
                )
            )
            trace_color = "#dc2626" if run_label == 1 else "#0f766e"
            fig_trace.add_trace(
                go.Scatter(
                    x=x_steps,
                    y=run_values,
                    mode="lines",
                    name=f"Wafer Run {selected_run_id} ({'Fault' if run_label == 1 else 'Normal'})",
                    line=dict(color=trace_color, width=2.5),
                )
            )
            fig_trace.update_layout(
                title=f"Sensor Trace across 152 Time Steps — Wafer {selected_run_id}",
                xaxis_title="Time Step Index (0 to 151)",
                yaxis_title="Sensor Reading (Standardized)",
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
                margin=dict(l=20, r=20, t=50, b=20),
                height=380,
            )
            st.plotly_chart(fig_trace, width="stretch")

    st.divider()

    # Section 3: Detector Comparison Across Alarm Budgets
    st.subheader("3. Detector Comparison Across Operational Budgets")
    st.markdown(
        "Detectors evaluated across the 7-block prequential timeline under fixed alarm budgets "
        "(2, 5, and 10 alarms per 100 runs)."
    )

    detectors_dict = metrics.get("detectors", {})
    comparison_rows = []
    for d_id, d_data in detectors_dict.items():
        pr_auc = d_data.get("pooled_pr_auc", 0.0)
        roc_auc = d_data.get("pooled_roc_auc", 0.0)
        rec_5_far = d_data.get("recall_at_5_far", 0.0)
        b_dict = d_data.get("alarm_budgets", {})

        b2 = b_dict.get("budget_2", {})
        b5 = b_dict.get("budget_5", {})
        b10 = b_dict.get("budget_10", {})

        comparison_rows.append(
            {
                "Detector": WAFER_DETECTOR_NAMES.get(d_id, d_id),
                "PR-AUC": round(pr_auc, 4),
                "ROC-AUC": round(roc_auc, 4),
                "Recall@5% FAR": f"{rec_5_far * 100:.1f}%",
                "Budget 2 Prec / Rec": f"{b2.get('precision', 0.0)*100:.1f}% / {b2.get('recall', 0.0)*100:.1f}%",
                "Budget 5 Prec / Rec": f"{b5.get('precision', 0.0)*100:.1f}% / {b5.get('recall', 0.0)*100:.1f}%",
                "Budget 10 Prec / Rec": f"{b10.get('precision', 0.0)*100:.1f}% / {b10.get('recall', 0.0)*100:.1f}%",
            }
        )

    df_comp = pd.DataFrame(comparison_rows)
    st.dataframe(df_comp, width="stretch", hide_index=True)

    # Comparison Bar Chart of PR-AUC
    chart_names = [WAFER_DETECTOR_NAMES.get(d_id, d_id).split(" — ")[0] for d_id in detectors_dict]
    chart_pr_aucs = [detectors_dict[d_id].get("pooled_pr_auc", 0.0) for d_id in detectors_dict]
    chart_colors = [
        "#0f766e" if d_id == "E08_v2" else ("#3b82f6" if "v2" in d_id and d_id in ("E04_v2", "E05_v2", "E06_v2") else "#64748b")
        for d_id in detectors_dict
    ]

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=chart_names,
            y=chart_pr_aucs,
            marker_color=chart_colors,
            text=[f"{v:.3f}" for v in chart_pr_aucs],
            textposition="auto",
        )
    )
    prevalence = metrics.get("prequential_summary", {}).get("evaluated_prevalence", 0.1078)
    fig_bar.add_hline(
        y=prevalence,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text=f"Chance Baseline ({prevalence:.3f})",
        annotation_position="bottom right",
    )
    fig_bar.update_layout(
        title="Prequential PR-AUC Across All 9 Detectors (Red Line = Chance Baseline)",
        xaxis_title="Detector",
        yaxis_title="PR-AUC (Higher is Better)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
    )
    st.plotly_chart(fig_bar, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="YieldGuard: Semiconductor Fault Detection",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    st.sidebar.markdown("## 🔬 YieldGuard")
    st.sidebar.caption("Early warning for faulty chip-production runs.")

    real_available = check_real_data_available()
    mode_options = (
        ["Real Dataset (SECOM)", "Synthetic Demo Data"]
        if real_available
        else ["Synthetic Demo Data"]
    )
    selected_mode = st.sidebar.selectbox(
        "Data",
        mode_options,
        index=0,
        help="Real: the public UCI SECOM dataset (1,567 production runs). Synthetic: 100 made-up runs for a quick demo.",
    )
    is_synthetic = (selected_mode == "Synthetic Demo Data") or not real_available

    if is_synthetic:
        st.warning(
            "⚠️ Running in SYNTHETIC DEMO MODE. The runs on screen are computer-generated, not real factory data."
        )

    bundle_dir = "models/v0.1.0" if Path("models/v0.1.0").exists() else "models/staging"
    scored_df, labels_df, bundle, synthetic_active = load_data_and_scores(
        use_synthetic=is_synthetic,
        bundle_dir=bundle_dir,
    )

    if bundle is None:
        st.error(
            "No trained model was found in models/v0.1.0 or models/staging. Run the training command first "
            "(see docs/07_OPERATIONS_AND_COMMANDS.md)."
        )
        return

    # Option values stay as the original view names (tests select by them); labels are plain-language.
    views = [VIEW_OVERVIEW, VIEW_EWMA, VIEW_MSPC, VIEW_DRILLDOWN, VIEW_BUDGET, VIEW_WAFER]
    active_view = st.sidebar.radio(
        "Pages",
        views,
        index=0,
        format_func=lambda v: VIEW_LABELS[v][0],
        captions=[VIEW_LABELS[v][1] for v in views],
    )

    st.sidebar.divider()
    with st.sidebar.expander("The eight detectors (E01–E08)", expanded=False):
        primary_id = bundle.policy.get("primary_detector", "E02")
        st.markdown(
            "\n".join(
                f"- **{det_id}{' ★' if det_id == primary_id else ''} {info['short']}**: {DETECTOR_PLAIN[det_id]}"
                for det_id, info in DETECTOR_CATALOG.items()
            )
        )
    with st.sidebar.expander("Key terms", expanded=False):
        st.markdown("\n".join(f"- **{term}**: {meaning}" for term, meaning in GLOSSARY))
    st.sidebar.caption(
        f"Model version `{bundle.version}`  \n"
        "Results are offline benchmarks on the public UCI SECOM dataset (CC BY 4.0), "
        "not measured factory yield or cost savings."
    )

    render_top_header_and_kpis(scored_df, labels_df, bundle, is_synthetic)

    if active_view == VIEW_OVERVIEW:
        render_project_overview_view(bundle)
    elif active_view == VIEW_EWMA:
        render_ewma_view(scored_df, bundle)
    elif active_view == VIEW_MSPC:
        render_mspc_view(scored_df, bundle)
    elif active_view == VIEW_DRILLDOWN:
        render_drilldown_view(scored_df, labels_df, bundle)
    elif active_view == VIEW_BUDGET:
        render_alarm_budget_view(scored_df, labels_df, bundle)
    elif active_view == VIEW_WAFER:
        render_wafer_etch_view(use_synthetic=is_synthetic)


if __name__ == "__main__":
    main()
