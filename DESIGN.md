# YieldGuard Dashboard — Design Specification

> Design direction for the Streamlit SPC and fault-detection dashboard defined
> in T15. This file is the single source of visual and interaction direction;
> `antislop.md` acts as the filter on top of it.

---

## 1. Subject and audience

| Field | Value |
|---|---|
| Product | YieldGuard — semiconductor process fault detection and SPC demonstrator |
| Audience | Yield engineers, process engineers, ML engineers reviewing offline run data |
| Page job | Let the engineer see control-chart behaviour, spot flagged runs, understand why sensors triggered, and compare detector budgets — all from a single browser tab |
| Data domain | UCI SECOM: 590 sensors, ~1 567 runs, 6.6 % fail rate, time-ordered |

---

## 2. Design read

> Reading this as: **internal engineering tool** for **semiconductor process
> engineers**, in a **cleanroom-industrial** style, dial **ENERGY 1 / RHYTHM 2
> / MOTION 1**.

### Dial rationale

| Dial | Value | Reason |
|---|---|---|
| ENERGY | 1 (Calm) | Engineers scan dense numeric charts under time pressure; the interface must recede behind the data, not compete with it |
| RHYTHM | 2 (Balanced) | Four views have genuinely different content (time-series chart, scatter, bar contributions, summary table), so layouts should vary to match, but not wildly |
| MOTION | 1 (Hover only) | Streamlit re-runs the script on interaction; gratuitous animation would stutter on reload. Plotly hover tooltips are the only motion |

---

## 3. Palette

Drawn from the semiconductor fab environment: amber status lights on grey
steel enclosures, blue-white cleanroom illumination, and the red/green of
pass/fail lot disposition.

| Role | Hex | Name | Reason |
|---|---|---|---|
| Background | `#F7F8FA` | Cleanroom White | Neutral surface; maximises contrast for chart ink without harsh pure white glare |
| Surface (cards, sidebar) | `#EBEEF2` | Panel Grey | One step down from background; separates sidebar and metric cards without heavy borders |
| Primary text | `#1B1F26` | Charcoal | Near-black for body text; softer than `#000` under sustained reading |
| Secondary text | `#5A6270` | Steel | Labels, axis ticks, captions — clearly subordinate to primary |
| Accent | `#D4840A` | Fab Amber | Alarm markers, flagged-run highlights, threshold lines. Amber is the universal caution colour in semiconductor fabs |
| Pass | `#2E7D4F` | Lot Green | Pass-labelled runs in scatter and table rows |
| Fail | `#C0392B` | Reject Red | Fail-labelled runs, out-of-control points, rule violations |
| Chart ink | `#3C4858` | Graphite | Default trace colour; distinguishable from both axes and alarm markers |
| Limit lines | `#8896A7` | Control Grey | UCL/LCL and theoretical limits — visible but quieter than data traces |

**Neutrals** (white, near-white, near-black) are structural and do not count
toward the core palette. Core palette is three colours: Fab Amber, Lot Green,
Reject Red — plus Graphite as the single accent for chart ink.

---

## 4. Typography

Streamlit renders with its built-in font stack. Custom font loading is
unreliable in Streamlit's iframe model, so the design works *within* that
constraint rather than fighting it.

| Role | Font | Weight | Size | Reason |
|---|---|---|---|---|
| Page title | System sans-serif (Streamlit default) | 700 | 1.75 rem | Single use per view; bold weight anchors the page |
| Section heading | System sans-serif | 600 | 1.15 rem | Separates chart panels; semi-bold avoids shouting |
| Body / labels | System sans-serif | 400 | 0.95 rem | Comfortable reading size for control descriptions |
| Metric values | Streamlit `st.metric` | 600 | Default large | Numbers are the focal point; `st.metric` renders them prominently by design |
| Axis ticks / captions | Plotly default (Open Sans) | 400 | 11 px | Dense axis labels; must not crowd the plot area |
| Code / sensor names | Monospace (system) | 400 | 0.85 rem | Sensor IDs are quasi-code identifiers; monospace disambiguates `0`/`O`, `1`/`l` |

**Reason for not importing a custom typeface:** Streamlit injects its own
`<style>` and re-renders on every widget interaction. External font requests
add latency and FOUC to each re-run. The system stack is fast, readable, and
consistent across engineer workstations.

---

## 5. Layout — four views mapped to content

Each view exists because the PRD defines it (section 8 of
`04_TECHNICAL_DESIGN.md`), not because a template expects it.

### View 1 — EWMA Control Charts

| Element | Purpose |
|---|---|
| Sensor selector (sidebar dropdown) | Engineers monitor one sensor at a time; the 10 monitored sensors are the options |
| EWMA line chart (full width) | Primary content: EWMA values over run order with UCL/LCL shading and centre line |
| Rule-violation markers (diamonds on chart) | Overlaid where Western Electric rules fire; colour = Reject Red, shape = diamond to distinguish from data points |
| Run-detail hover tooltip | Shows run index, EWMA value, raw value, active rule — Plotly native hover, no separate panel |

**Layout reason:** A single full-width time-series chart with one selector is
the minimum viable SPC view. Adding a second chart on the same screen would
halve the x-axis resolution for no content gain.

### View 2 — T² and SPE Scatter

| Element | Purpose |
|---|---|
| Dual-axis scatter (T² left y, SPE right y, run order x) | Both multivariate scores on one timeline; engineers compare them visually |
| Threshold lines (dashed, Fab Amber) | Budget threshold; Limit Grey for theoretical limits |
| Colour encoding | Lot Green for pass, Reject Red for fail, Fab Amber outline for alarm-flagged |
| Budget selector (sidebar radio) | Switches between 2 / 5 / 10 alarms per 100 |

**Layout reason:** T² and SPE are correlated but not identical; overlaying them
avoids the engineer tabbing between two separate charts. Dual y-axes are
acceptable here because both quantities are scores on similar scales and the
engineer expects to see them together.

### View 3 — Flagged-Run Drill-Down

| Element | Purpose |
|---|---|
| Run selector (sidebar dropdown, filtered to flagged runs) | Scopes to one run at a time |
| Two side-by-side bar charts (MSPC contributions left, SHAP right) | Ranked sensors; the two methods are compared visually |
| Group-member annotation below each bar | Lists collinear sensors that share the contribution |
| Metric row at top (`st.metric` x 4) | T², SPE, IF score, XGB score for the selected run — immediate numeric context before the bars |

**Layout reason:** Side-by-side bars let the engineer compare MSPC and SHAP
rankings at a glance. Stacking them vertically would force scrolling and lose
the comparison. The metric row anchors the view with the exact numbers.

### View 4 — Alarm-Budget Summary Table

| Element | Purpose |
|---|---|
| `st.dataframe` styled table | Rows: budget 2, 5, 10. Columns: alarms, false alarms, recall, PR-AUC per detector |
| Conditional formatting | Reject Red background for recall below 0.5; Lot Green for recall above 0.8 |
| Data source label | States whether development or holdout data is shown |

**Layout reason:** A single table is the right format for a comparison matrix.
No chart would add information here; the numbers *are* the content.

### Sidebar (global)

| Element | Purpose |
|---|---|
| View selector (radio) | Switches between the four views; avoids a navbar with dead links |
| Data-mode indicator | "Synthetic demo data" or "SECOM" label — always visible, Fab Amber text when synthetic |
| Bundle version | Read from `metadata.json`; engineers need to know which model they are viewing |

---

## 6. Identity motif

**The motif is the amber threshold line.**

Every view includes at least one dashed amber line marking the alarm budget
threshold (or, in the table view, amber-highlighted cells for the active
budget). This is the visual element that ties the four views into one tool:
it is the boundary the engineer cares about, and it recurs in a consistent
style throughout.

**Reason:** The alarm budget is the central concept of YieldGuard — not
accuracy, not a model name. Making the budget threshold the recurring visual
motif reinforces the product's actual purpose.

---

## 7. Interaction patterns

| Pattern | Implementation | Reason |
|---|---|---|
| View switching | `st.sidebar.radio` | Single navigation control; no dead links (R-24) |
| Sensor / run selection | `st.sidebar.selectbox` | Scoped to the active view; hidden when irrelevant |
| Budget switching | `st.sidebar.radio` (View 2 and 4) | Three options match the three stored thresholds |
| Hover detail | Plotly `hovertemplate` | No click-to-expand panels; hover is faster for chart scanning |
| Synthetic-mode warning | `st.sidebar.warning` | Always visible when no real data is loaded; honest placeholder (R-38) |

**No dark mode toggle.** The audience (fab engineers) uses bright-lit office
monitors during shifts. A light theme with high-contrast chart ink is the
correct default. Adding a toggle would require validating every Plotly colour
in both themes (R-34), and there is no user request for it.

---

## 8. States

| State | Handling |
|---|---|
| No bundle loaded | Full-page `st.error`: "No model bundle found. Run `yieldguard freeze` first." No charts rendered (R-27) |
| Synthetic mode | All views functional; sidebar shows "Synthetic demo data" in Fab Amber; every chart subtitle appends "(synthetic)" (R-38) |
| Empty selection | Selectbox defaults to first valid option; no empty state reachable through normal interaction |
| Loading | Streamlit's built-in spinner on data load; no custom skeleton (R-27) |
| Error in scoring | `st.error` with the exception message; chart area replaced with error text, not blank (R-27) |

---

## 9. What is deliberately absent

| Omission | Reason |
|---|---|
| Logo / brand mark | No logo exists; product name as text in the sidebar title is the honest placeholder (R-23) |
| Animations beyond hover | MOTION dial is 1; Streamlit re-render kills CSS animations anyway |
| Background patterns | No grid, dots, or mesh — they add noise to a data-dense tool (R-07) |
| Glassmorphism / blur | An engineering dashboard is not an app store listing (R-10) |
| Gradient backgrounds | Flat `#F7F8FA` — gradients would interfere with chart readability (R-01) |
| Generic icons | No sparkle, lightning, or orb. Streamlit's native emoji markers (`:warning:`, `:chart_with_upwards_trend:`) are used sparingly where they add meaning (R-04) |
| Footer | No template footer; the sidebar contains all navigation and metadata |
| FAQ / testimonials / stats badges | This is an internal tool, not a marketing page (C-3) |

---

## 10. Streamlit CSS overrides

Minimal overrides injected via `st.markdown(unsafe_allow_html=True)` to align
with the palette. These are the only custom styles; everything else uses
Streamlit defaults.

```css
/* Override only what the palette requires */
[data-testid="stSidebar"] {
    background-color: #EBEEF2;   /* Panel Grey */
}
[data-testid="stMetricValue"] {
    color: #1B1F26;              /* Charcoal — metric numbers should be primary weight */
}
div.stAlert > div[data-baseweb="notification"] {
    /* Synthetic-mode warning uses Fab Amber */
    background-color: #FFF3E0;
    border-left-color: #D4840A;
}
```

**Reason for minimal overrides:** Streamlit's DOM structure changes between
versions. The fewer selectors we depend on, the fewer breakages on upgrade.
The three overrides above target stable `data-testid` attributes documented in
Streamlit's testing API.

---

## 11. Plotly chart template

A reusable Plotly `layout` dict applied to every chart for visual consistency.

```python
YIELDGUARD_LAYOUT = dict(
    paper_bgcolor="#F7F8FA",      # Cleanroom White
    plot_bgcolor="#FFFFFF",        # Pure white plot area for maximum ink contrast
    font=dict(family="Open Sans, sans-serif", size=12, color="#1B1F26"),
    title_font=dict(size=14, color="#1B1F26"),
    xaxis=dict(
        gridcolor="#E0E4EA",
        zerolinecolor="#E0E4EA",
        tickfont=dict(size=11, color="#5A6270"),
    ),
    yaxis=dict(
        gridcolor="#E0E4EA",
        zerolinecolor="#E0E4EA",
        tickfont=dict(size=11, color="#5A6270"),
    ),
    margin=dict(l=60, r=30, t=40, b=50),
    hoverlabel=dict(
        bgcolor="#1B1F26",
        font_color="#F7F8FA",
        font_size=12,
    ),
)
```

| Decision | Reason |
|---|---|
| White plot area on off-white page | Subtle lift separates data from chrome without a heavy border or shadow (R-12) |
| Tight margins | Maximises chart real estate in Streamlit's fixed-width container |
| Dark hover label | High contrast tooltip; readable on any trace colour |

---

## 12. Colour constants (Python)

```python
# Paste into app/dashboard.py or a shared constants module.
COLORS = {
    "background":    "#F7F8FA",   # Cleanroom White
    "surface":       "#EBEEF2",   # Panel Grey
    "text_primary":  "#1B1F26",   # Charcoal
    "text_secondary":"#5A6270",   # Steel
    "accent":        "#D4840A",   # Fab Amber
    "pass":          "#2E7D4F",   # Lot Green
    "fail":          "#C0392B",   # Reject Red
    "chart_ink":     "#3C4858",   # Graphite
    "limit_line":    "#8896A7",   # Control Grey
    "grid":          "#E0E4EA",   # Chart gridlines
}
```

---

## 13. Decision log (R-31 compliance)

Every major decision with its one-line reason, as required by antislop R-31.

| Decision | One-line reason |
|---|---|
| Light theme, no toggle | Fab engineers use bright monitors; one validated theme is better than two half-validated ones (R-34) |
| Fab Amber accent | Amber is the universal caution colour in semiconductor manufacturing; it grounds the tool in its domain |
| System font stack | Avoids FOUC on Streamlit re-render; every engineer's workstation has a system sans-serif |
| Monospace for sensor IDs | Sensor names are alphanumeric identifiers; monospace prevents character ambiguity |
| No background decoration | Chart-dense dashboards need a quiet canvas; decoration competes with data ink (Tufte) |
| Four views via sidebar radio | Maps 1:1 to the PRD's four dashboard views; no dead nav links |
| Full-width EWMA chart | Time-series charts need horizontal resolution; splitting width wastes the critical axis |
| Side-by-side contribution bars | MSPC vs SHAP comparison is the view's purpose; vertical stacking loses the comparison |
| Minimal CSS overrides | Streamlit DOM changes between versions; fewer selectors = fewer upgrade breakages |
| Amber threshold as identity motif | The alarm budget is YieldGuard's central concept; it should be the visual constant across views |
| No dark mode | No user request; validating every Plotly colour in both themes doubles the QA surface for no demonstrated need |
| Conditional table formatting | Red/green on recall values gives instant pass/fail reading without requiring the engineer to interpret numbers |
| Synthetic-mode label in sidebar | Always visible, honest about data source (R-38); amber text draws the eye without blocking interaction |

---

*Dial: ENERGY 1 / RHYTHM 2 / MOTION 1*
