import base64
import csv
import json
import re
from pathlib import Path

import streamlit as st

try:
    import pandas as pd
except ImportError:  # pragma: no cover - keeps the page from crashing if pandas is unavailable
    pd = None

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - optional for the choropleth page
    px = None

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional for the choropleth page
    go = None

try:
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover - optional for the EDA page
    make_subplots = None


st.set_page_config(
    page_title="Forced Displacement Tracker : South Sudan",
    layout="wide",
    initial_sidebar_state="collapsed",
)


SUMMARY_DATA_PATH = Path("data/state_round_enriched.csv")
DASHBOARD_DATA_PATH = Path("data/state_round_modeling_complete.csv")
MAP_PATH = Path("assets/south-sudan-map.png")
ABOUT_PREVIEW_PATH = Path("assets/dashboard-preview.png")
STATE_GEOJSON_PATH = Path("data/south_sudan_adm1.geojson")


@st.cache_data(show_spinner=False)
def load_summary_stats() -> dict:
    if SUMMARY_DATA_PATH.exists():
        if pd is not None:
            try:
                df = pd.read_csv(SUMMARY_DATA_PATH)
                if {"state", "round"}.issubset(df.columns):
                    return {
                        "states": int(df["state"].nunique()),
                        "rounds": int(df["round"].nunique()),
                        "rows": int(len(df)),
                        "loaded": True,
                    }
            except Exception:
                pass

        try:
            with SUMMARY_DATA_PATH.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            if reader.fieldnames and {"state", "round"}.issubset(reader.fieldnames):
                return {
                    "states": len({row["state"] for row in rows if row.get("state")}),
                    "rounds": len({row["round"] for row in rows if row.get("round")}),
                    "rows": len(rows),
                    "loaded": True,
                }
        except Exception:
            pass

    return {"states": None, "rounds": None, "rows": None, "loaded": False}


@st.cache_data(show_spinner=False)
def load_dashboard_data():
    if pd is None:
        return build_dummy_dashboard_data(), False

    try:
        df = pd.read_csv(DASHBOARD_DATA_PATH)
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

        numeric_columns = [
            "dtm_idp_ind",
            "idp_per_1000",
            "acled_events",
            "csi_0_100",
        ]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        return df, True
    except Exception:
        return build_dummy_dashboard_data(), False


def build_dummy_dashboard_data():
    if pd is None:
        return None

    dummy_rows = [
        {
            "state": "Central Equatoria",
            "month": "2021-01-01",
            "round": "R10",
            "dtm_idp_ind": 185000,
            "idp_per_1000": 118.4,
            "acled_events": 22,
            "csi_0_100": 48.6,
            "regime": "compound",
            "period": "post_flood",
        },
        {
            "state": "Jonglei",
            "month": "2021-01-01",
            "round": "R10",
            "dtm_idp_ind": 246000,
            "idp_per_1000": 171.2,
            "acled_events": 35,
            "csi_0_100": 63.4,
            "regime": "compound",
            "period": "post_flood",
        },
        {
            "state": "Unity",
            "month": "2021-01-01",
            "round": "R10",
            "dtm_idp_ind": 213000,
            "idp_per_1000": 164.7,
            "acled_events": 29,
            "csi_0_100": 58.1,
            "regime": "compound",
            "period": "post_flood",
        },
        {
            "state": "Central Equatoria",
            "month": "2021-07-01",
            "round": "R11",
            "dtm_idp_ind": 196000,
            "idp_per_1000": 121.9,
            "acled_events": 18,
            "csi_0_100": 44.2,
            "regime": "market_stress",
            "period": "post_flood",
        },
        {
            "state": "Jonglei",
            "month": "2021-07-01",
            "round": "R11",
            "dtm_idp_ind": 258000,
            "idp_per_1000": 176.6,
            "acled_events": 39,
            "csi_0_100": 66.9,
            "regime": "compound",
            "period": "post_flood",
        },
        {
            "state": "Unity",
            "month": "2021-07-01",
            "round": "R11",
            "dtm_idp_ind": 220000,
            "idp_per_1000": 168.3,
            "acled_events": 31,
            "csi_0_100": 61.7,
            "regime": "compound",
            "period": "post_flood",
        },
    ]
    df = pd.DataFrame(dummy_rows)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df


def normalize_state_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


@st.cache_data(show_spinner=False)
def load_state_geojson():
    if not STATE_GEOJSON_PATH.exists():
        return None

    try:
        with STATE_GEOJSON_PATH.open(encoding="utf-8") as file:
            geojson = json.load(file)
    except Exception:
        return None

    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        candidate_name = None
        for key in [
            "shapeName",
            "shapeGroup",
            "name",
            "admin1Name",
            "ADM1_EN",
            "state",
        ]:
            if properties.get(key):
                candidate_name = properties[key]
                break

        if candidate_name:
            properties["state_match"] = normalize_state_name(candidate_name)

    return geojson


def round_sort_key(value: str):
    match = re.search(r"(\d+)", str(value))
    if match:
        return int(match.group(1))
    return str(value)


def build_round_labels(df):
    if pd is None or df is None or df.empty:
        return {}

    round_labels = {}
    grouped = df.dropna(subset=["round"]).groupby("round")
    for round_value, group in grouped:
        month_value = group["month"].dropna().min() if "month" in group.columns else None
        if pd.notna(month_value):
            round_labels[str(round_value)] = f"{round_value} \u2014 {month_value.strftime('%b %Y')}"
        else:
            round_labels[str(round_value)] = str(round_value)
    return round_labels


def render_stat_cards(stats: dict) -> None:
    labels = [
        ("States Covered", stats["states"]),
        ("DTM Rounds", stats["rounds"]),
        ("State-Round Records", stats["rows"]),
    ]
    cols = st.columns(3, gap="medium")
    for col, (label, value) in zip(cols, labels):
        display_value = f"{value:,}" if isinstance(value, int) else "Unavailable"
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{display_value}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_about_page(stats: dict) -> None:
    st.markdown(
        """
        <style>
            .stApp,
            body,
            html,
            div[data-testid="stAppViewContainer"],
            div[data-testid="stAppViewContainer"] > .main {
                overflow: hidden;
            }

            .block-container {
                padding-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    map_markup = '<div class="preview-placeholder">Map / Dashboard Preview</div>'
    if MAP_PATH.exists():
        encoded = base64.b64encode(MAP_PATH.read_bytes()).decode("utf-8")
        map_markup = (
            f'<img class="about-map-image" src="data:image/png;base64,{encoded}" '
            'alt="South Sudan map preview" />'
        )

    preview_markup = '<div class="preview-card-placeholder">Dashboard preview coming soon</div>'
    if ABOUT_PREVIEW_PATH.exists():
        preview_encoded = base64.b64encode(ABOUT_PREVIEW_PATH.read_bytes()).decode("utf-8")
        preview_markup = (
            f'<img class="preview-card-image" src="data:image/png;base64,{preview_encoded}" '
            'alt="Dashboard preview" />'
        )

    left_col, right_col = st.columns([0.58, 0.42], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="content-card about-grid-card">
                <div class="section-title">Overview</div>
                <p class="body-copy">
                    South Sudan has experienced one of the most severe humanitarian crises since its independence in 2011, with displacement remaining a persistent and complex challenge across the country. Across 13 DTM assessment rounds from 2018 to 2025, an estimated 1.3–2.3 million people were internally displaced. These patterns are shaped by overlapping conditions, including armed conflict, food insecurity, flooding, market instability, and infrastructure disruption, which can place civilians at heightened risk of displacement.
                </p>
                <p class="body-copy">
                    This project examines how displacement rate changes across South Sudan's
                    states and DTM assessment rounds. Rather than treating displacement as the
                    result of a single factor, we explore how multiple contributing conditions, including
                    conflict intensity, food insecurity, food prices, and environmental shocks,
                    are associated with displacement patterns over time.
                </p>
            </div>
            <div class="content-card about-grid-card about-grid-bottom">
                <div class="section-title">About This Project</div>
                <p class="body-copy">
                    A major challenge in studying displacement is that humanitarian data is often
                    fragmented and incomplete across sources. To address this, we construct a
                    unified state-round panel dataset by integrating displacement records,
                    conflict data, food security indicators, market price data, and flood-related
                    variables.
                </p>
                <p class="body-copy">
                    We use interpolation to improve coverage for missing feature values, allowing
                    us to retain more usable observations for descriptive analysis. The final
                    dashboard translates this enriched dataset into an interactive tool for
                    exploring regional displacement trends, key contributing conditions, combined score index
                     , and state-level differences.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            f"""
            <div class="about-map-block">
                {map_markup}
            </div>
            <div class="preview-card">
                {preview_markup}
                <div class="preview-card-body">
                    <div class="preview-card-title">Explore Patterns and Regional Conditions</div>
                    <div class="preview-card-copy">
                        Gain deeper insight into how conflict patterns, food insecurity, market stress, and environmental shocks interact to shape displacement risk.
                    </div>
                </div>
                <div class="learn-more-button-wrapper">
            """,
            unsafe_allow_html=True,
        )
        if st.button("Learn More", key="about_learn_more"):
            st.session_state["active_page"] = "Analysis"
            st.rerun()
        st.markdown(
            """
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_state_round_filters(df, selected_state: str, selected_round: str):
    filtered_df = df.copy()
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["state"] == selected_state]
    if selected_round != "All":
        filtered_df = filtered_df[filtered_df["round"] == selected_round]
    return filtered_df


def render_dashboard() -> None:
    df, loaded_from_csv = load_dashboard_data()

    if pd is None or df is None:
        st.markdown(
            """
            <div class="content-card dashboard-placeholder">
                <div class="section-title">Dashboard</div>
                <p class="body-copy">
                    Pandas is not available in this environment, so the dashboard cannot load yet.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    states = ["All"] + sorted(df["state"].dropna().astype(str).unique().tolist())
    rounds = ["All"] + sorted(df["round"].dropna().astype(str).unique().tolist())

    filter_cols = st.columns(2, gap="medium")
    with filter_cols[0]:
        st.markdown('<div class="filter-shell">', unsafe_allow_html=True)
        selected_state = st.selectbox("State", states, index=0)
        st.markdown("</div>", unsafe_allow_html=True)

    with filter_cols[1]:
        st.markdown('<div class="filter-shell">', unsafe_allow_html=True)
        selected_round = st.selectbox("Round", rounds, index=0)
        st.markdown("</div>", unsafe_allow_html=True)

    filtered_df = apply_state_round_filters(df, selected_state, selected_round)

    if filtered_df.empty:
        st.markdown(
            """
            <div class="content-card">
                <div class="section-title">Dashboard</div>
                <p class="body-copy">
                    No records match the current filters. Try a different state or round selection.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    total_idps = filtered_df["dtm_idp_ind"].fillna(0).sum()
    avg_idp_per_1000 = filtered_df["idp_per_1000"].mean()
    total_conflict_events = filtered_df["acled_events"].fillna(0).sum()
    avg_csi = filtered_df["csi_0_100"].mean()

    metric_cols = st.columns(4, gap="medium")
    with metric_cols[0]:
        render_metric_card("Total IDPs", f"{total_idps:,.0f}")
    with metric_cols[1]:
        render_metric_card("Average IDPs per 1,000", f"{avg_idp_per_1000:,.1f}")
    with metric_cols[2]:
        render_metric_card("Total Conflict Events", f"{total_conflict_events:,.0f}")
    with metric_cols[3]:
        render_metric_card("Average Combined Pressure Score", f"{avg_csi:,.1f}")

    line_chart_df = (
        filtered_df.sort_values("month")
        .groupby("month", as_index=False)["idp_per_1000"]
        .mean()
        .set_index("month")
    )

    bar_source = filtered_df if selected_state == "All" else df[df["round"].isin(filtered_df["round"].unique())]
    if selected_round != "All":
        bar_source = df[df["round"] == selected_round]
    bar_chart_df = (
        bar_source.groupby("state", as_index=False)["idp_per_1000"]
        .mean()
        .sort_values("idp_per_1000", ascending=False)
        .set_index("state")
    )

    heatmap_df = (
        filtered_df.pivot_table(
            index="state",
            columns="round",
            values="csi_0_100",
            aggfunc="mean",
        )
        .sort_index()
    )

    latest_month = filtered_df["month"].max()
    latest_table_df = (
        filtered_df[filtered_df["month"] == latest_month][
            ["state", "round", "idp_per_1000", "csi_0_100", "regime", "period"]
        ]
        .sort_values(["state", "round"])
        .reset_index(drop=True)
    )

    chart_cols = st.columns([1.3, 1], gap="large")
    with chart_cols[0]:
        st.markdown(
            """
            <div class="content-card">
                <div class="section-title">Displacement Rate Over Time</div>
            """,
            unsafe_allow_html=True,
        )
        st.line_chart(line_chart_df, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_cols[1]:
        st.markdown(
            """
            <div class="content-card">
                <div class="section-title">Average IDPs per 1,000 by State</div>
            """,
            unsafe_allow_html=True,
        )
        st.bar_chart(bar_chart_df, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

    lower_cols = st.columns([1.05, 1.2], gap="large")
    with lower_cols[0]:
        st.markdown(
            """
            <div class="content-card">
                <div class="section-title">Combined Pressure Score by State and Round</div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(heatmap_df.round(1), width="stretch", height=320)
        st.markdown("</div>", unsafe_allow_html=True)

    with lower_cols[1]:
        st.markdown(
            """
            <div class="content-card">
                <div class="section-title">Latest Monitoring Snapshot</div>
            """,
            unsafe_allow_html=True,
        )
        snapshot_display = latest_table_df.copy()
        snapshot_display["idp_per_1000"] = snapshot_display["idp_per_1000"].map(
            lambda value: f"{value:,.1f}" if pd.notna(value) else ""
        )
        snapshot_display["csi_0_100"] = snapshot_display["csi_0_100"].map(
            lambda value: f"{value:,.1f}" if pd.notna(value) else ""
        )
        st.dataframe(snapshot_display, width="stretch", height=320)
        st.markdown("</div>", unsafe_allow_html=True)

    if not loaded_from_csv:
        st.caption(
            "The dashboard is currently showing fallback dummy data because the CSV could not be loaded."
        )


def render_dual_axis_time_chart(
    chart_df,
    left_col_name: str,
    right_col_name: str,
    left_label: str,
    right_label: str,
):
    if go is not None and make_subplots is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df[left_col_name],
                name=left_label,
                mode="lines+markers",
                line={"color": "#2b67c8", "width": 3},
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df[right_col_name],
                name=right_label,
                mode="lines+markers",
                line={"color": "#c6553d", "width": 2.5},
            ),
            secondary_y=True,
        )
        fig.update_layout(
            height=300,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend={"orientation": "h", "y": 1.08, "x": 0},
        )
        fig.update_yaxes(title_text=left_label, secondary_y=False)
        fig.update_yaxes(title_text=right_label, secondary_y=True)
        st.plotly_chart(fig, config={"responsive": True})
    else:
        st.line_chart(chart_df[[left_col_name, right_col_name]], height=300)


PLOTLY_STATIC_CONFIG = {"displayModeBar": False, "scrollZoom": False}


def apply_static_plot_style(fig, xaxis_title: str, yaxis_title: str):
    fig.update_layout(
        dragmode=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    fig.update_xaxes(title_text=xaxis_title, fixedrange=True)
    fig.update_yaxes(title_text=yaxis_title, fixedrange=True)
    return fig


def resolve_first_available_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def render_eda() -> None:
    df, loaded_from_csv = load_dashboard_data()

    if pd is None or df is None:
        st.markdown(
            """
            <div class="content-card dashboard-placeholder">
                <div class="section-title">Analysis</div>
                <p class="body-copy">
                    Pandas is not available in this environment, so the Analysis page cannot load yet.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if px is None:
        st.warning("Plotly is required to render the Analysis page charts.")
        return

    if "month" not in df.columns:
        st.warning("Analysis page cannot be rendered because the required column `month` is missing.")
        return

    patterns_df = df.copy()
    patterns_df["month"] = pd.to_datetime(patterns_df["month"], errors="coerce")
    patterns_df = patterns_df.dropna(subset=["month"]).sort_values("month")

    if patterns_df.empty:
        st.warning("Analysis page cannot be rendered because there are no valid `month` values in the CSV.")
        return

    if "round" not in patterns_df.columns:
        st.warning("Analysis page cannot be rendered because the required column `round` is missing.")
        return

    conflict_col = resolve_first_available_column(patterns_df, ["acled_events_lag1", "acled_events"])
    food_price_col = resolve_first_available_column(
        patterns_df, ["wfp_avg_usdprice_lag1", "wfp_avg_usdprice"]
    )
    flood_col = resolve_first_available_column(patterns_df, ["flood_flag", "flood_affected_people"])

    total_idps_value = "Unavailable"
    if "dtm_idp_ind" in patterns_df.columns:
        total_idps = pd.to_numeric(patterns_df["dtm_idp_ind"], errors="coerce").sum(min_count=1)
        if pd.notna(total_idps):
            total_idps_value = f"{total_idps / 1_000_000:.1f}M"

    avg_idp_rate_value = "Unavailable"
    if "idp_per_1000" in patterns_df.columns:
        avg_idp_rate = pd.to_numeric(patterns_df["idp_per_1000"], errors="coerce").mean()
        if pd.notna(avg_idp_rate):
            avg_idp_rate_value = f"{avg_idp_rate:.1f}"

    highest_state_value = "Unavailable"
    highest_state_description = "No state-round displacement rate is available."
    if {"state", "round", "idp_per_1000"}.issubset(patterns_df.columns):
        highest_rate_df = patterns_df.dropna(subset=["state", "round", "idp_per_1000"])
        if not highest_rate_df.empty:
            highest_row = highest_rate_df.loc[highest_rate_df["idp_per_1000"].idxmax()]
            highest_state_value = str(highest_row["state"])
            highest_state_description = (
                f"Reached {highest_row['idp_per_1000']:.0f} IDPs per 1,000 people in {highest_row['round']}."
            )

    compound_share_value = "Unavailable"
    if "regime" in patterns_df.columns:
        regime_series = patterns_df["regime"].dropna().astype(str).str.strip().str.lower()
        if not regime_series.empty:
            compound_share = (regime_series == "compound").mean() * 100
            compound_share_value = f"{compound_share:.1f}%"

    st.markdown("## Key Findings")

    finding_cols = st.columns(4, gap="medium")
    finding_cards = [
        (
            total_idps_value,
            "Recorded IDPs",
            "Total IDP count across all state-round records.",
        ),
        (
            avg_idp_rate_value,
            "Avg. IDPs per 1,000",
            "Average displacement rate across states and rounds.",
        ),
        (
            highest_state_value,
            "Highest rate observed",
            highest_state_description,
        ),
        (
            compound_share_value,
            "Compound observations",
            "State-rounds with multiple elevated pressures.",
        ),
    ]
    for col, (value, label, description) in zip(finding_cols, finding_cards):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: #ffffff;
                    border: 1px solid #d9e2ee;
                    border-radius: 18px;
                    padding: 1rem 1rem 0.95rem;
                    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
                    min-height: 150px;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                ">
                    <div style="
                        color: #143764;
                        font-size: 1.65rem;
                        font-weight: 800;
                        line-height: 1.1;
                        margin-bottom: 0.45rem;
                    ">{value}</div>
                    <div style="
                        color: #10233a;
                        font-size: 0.98rem;
                        font-weight: 700;
                        line-height: 1.35;
                        margin-bottom: 0.5rem;
                        min-height: 2.65rem;
                    ">{label}</div>
                    <div style="
                        color: #6a7b8d;
                        font-size: 0.88rem;
                        line-height: 1.5;
                        margin-top: auto;
                    ">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1.15rem;'></div>", unsafe_allow_html=True)
    st.markdown("## Displacement Patterns Over Time")

    if "idp_per_1000" in patterns_df.columns:
        displacement_roundly = (
            patterns_df.groupby("round", as_index=False)["idp_per_1000"]
            .mean()
            .sort_values("round", key=lambda s: s.map(round_sort_key))
        )
        st.subheader("Average Displacement Rate Across States Over Time (IDPs per 1,000 people)")
        displacement_fig = px.line(
            displacement_roundly,
            x="round",
            y="idp_per_1000",
            markers=True,
            labels={
                "round": "DTM assessment round",
                "idp_per_1000": "Average IDPs per 1,000 people",
            },
        )
        displacement_fig.update_traces(line={"color": "#2b67c8", "width": 3})
        apply_static_plot_style(
            displacement_fig,
            "DTM assessment round",
            "Average IDPs per 1,000 people",
        )
        st.plotly_chart(
            displacement_fig,
            config=PLOTLY_STATIC_CONFIG,
            use_container_width=True,
        )
    else:
        st.warning("Displacement chart unavailable because the required column `idp_per_1000` is missing.")

    if {"state", "round", "idp_per_1000"}.issubset(patterns_df.columns):
        heatmap_df = patterns_df.pivot_table(
            index="state",
            columns="round",
            values="idp_per_1000",
            aggfunc="mean",
        )
        if not heatmap_df.empty:
            heatmap_df = heatmap_df.reindex(
                columns=sorted(heatmap_df.columns, key=round_sort_key)
            )
            st.subheader(
                "Rate of Displacement by State and Round (IDPs per 1,000 people)"
            )
            heatmap_fig = px.imshow(
                heatmap_df,
                color_continuous_scale="OrRd",
                text_auto=".1f",
                aspect="auto",
                labels={
                    "x": "DTM assessment round",
                    "y": "State",
                    "color": "IDPs per 1,000",
                },
            )
            heatmap_fig.update_traces(
                hovertemplate="<b>%{y}</b><br>Round: %{x}<br>IDPs per 1,000: %{z:.1f}<extra></extra>",
                xgap=2,
                ygap=2,
            )
            heatmap_fig.update_layout(
                height=360,
                dragmode=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
                coloraxis_colorbar={
                    "title": "IDPs per 1,000",
                    "len": 0.75,
                    "thickness": 18,
                    "y": 0.5,
                },
            )
            heatmap_fig.update_xaxes(title_text="DTM assessment round", fixedrange=True)
            heatmap_fig.update_yaxes(title_text="State", fixedrange=True)
            st.plotly_chart(
                heatmap_fig,
                config=PLOTLY_STATIC_CONFIG,
                use_container_width=True,
            )
        else:
            st.warning("Displacement heatmap could not be generated because no state-round values were available.")
    else:
        missing_heatmap_cols = [
            column for column in ["state", "round", "idp_per_1000"] if column not in patterns_df.columns
        ]
        st.warning(
            "Displacement heatmap unavailable because these required columns are missing: "
            + ", ".join(f"`{column}`" for column in missing_heatmap_cols)
        )

    st.markdown(
        """
        <div style="
            background: #f7fafd;
            border: 1px solid #d9e2ee;
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin: 0.85rem 0 0.5rem;
        ">
            <div style="color: #24456f; font-size: 0.96rem; line-height: 1.6;">
                <strong>What to notice:</strong> Unity stands out with the highest displacement rates in later rounds, especially R14 and R15, while Jonglei also rises strongly after R14. Some states show missing or lower values, so the heatmap helps reveal where displacement pressure is concentrated.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)

    required_pre_post_cols = ["state", "period", "idp_per_1000"]
    missing_pre_post_cols = [column for column in required_pre_post_cols if column not in patterns_df.columns]
    if missing_pre_post_cols:
        st.warning(
            "Context chart unavailable because these required columns are missing: "
            + ", ".join(f"`{column}`" for column in missing_pre_post_cols)
        )
    else:
        period_df = (
            patterns_df.dropna(subset=["state", "period", "idp_per_1000"])
            .groupby(["state", "period"], as_index=False)["idp_per_1000"]
            .mean()
        )

        if period_df.empty:
            st.warning("Context chart could not be generated because no usable period data was available.")
        else:
            period_label_map = {
                "pre_war": "Pre-war",
                "post_war": "Post-war",
            }
            period_df["period_label"] = period_df["period"].map(period_label_map)
            period_df = period_df.dropna(subset=["period_label"])

            if period_df.empty:
                st.warning(
                    "Context chart could not be generated because `period` values were unavailable or unsupported."
                )
            else:
                post_war_order = (
                    period_df[period_df["period_label"] == "Post-war"]
                    .sort_values("idp_per_1000", ascending=False)["state"]
                    .tolist()
                )
                remaining_states = [
                    state for state in period_df["state"].unique().tolist() if state not in post_war_order
                ]
                state_order = post_war_order + sorted(remaining_states)

                pre_post_fig = px.bar(
                    period_df,
                    x="state",
                    y="idp_per_1000",
                    color="period_label",
                    barmode="group",
                    category_orders={
                        "state": state_order,
                        "period_label": ["Pre-war", "Post-war"],
                    },
                    color_discrete_map={
                        "Pre-war": "#2b67c8",
                        "Post-war": "#c0392b",
                    },
                    labels={
                        "state": "State",
                        "idp_per_1000": "Average IDPs per 1,000 people",
                        "period_label": "Period",
                    },
                )
                pre_post_fig.update_layout(
                    dragmode=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin={"l": 0, "r": 0, "t": 10, "b": 0},
                    legend={"title": "Period", "orientation": "h", "y": 1.08, "x": 0},
                )
                pre_post_fig.update_xaxes(title_text="State", fixedrange=True, tickangle=-30)
                pre_post_fig.update_yaxes(title_text="Average IDPs per 1,000 people", fixedrange=True)

                st.subheader("Average Displacement Rate Before and After April 2023 (IDPs per 1,000 people)")
                st.plotly_chart(
                    pre_post_fig,
                    config=PLOTLY_STATIC_CONFIG,
                    use_container_width=True,
                )

                st.markdown(
                    """
                    <div style="
                        background: #f7fafd;
                        border: 1px solid #d9e2ee;
                        border-radius: 16px;
                        padding: 0.95rem 1rem;
                        margin: 0.85rem 0 0.2rem;
                    ">
                        <div style="color: #24456f; font-size: 0.96rem; line-height: 1.6;">
                            <strong>What to notice:</strong> April 2023 marks the outbreak of the Sudan conflict in neighboring Sudan. This is not the South Sudanese civil war, but it affected the wider region through cross-border displacement, disrupted trade, food price pressure, and added strain on humanitarian systems in South Sudan.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)
    st.markdown("## Contributing Conditions Over Time")
    top_condition_cols = st.columns(2, gap="large")
    bottom_condition_cols = st.columns(2, gap="large")

    with top_condition_cols[0]:
        st.markdown("**Conflict Events Over Time (number of ACLED events)**")
        if conflict_col is not None:
            conflict_roundly = (
                patterns_df.groupby("round", as_index=False)[conflict_col]
                .sum(min_count=1)
                .sort_values("round", key=lambda s: s.map(round_sort_key))
            )
            conflict_fig = px.line(
                conflict_roundly,
                x="round",
                y=conflict_col,
                markers=True,
                labels={
                    "round": "DTM assessment round",
                    conflict_col: "Number of ACLED events",
                },
            )
            conflict_fig.update_traces(line={"color": "#c6553d", "width": 2.5})
            apply_static_plot_style(
                conflict_fig,
                "DTM assessment round",
                "Number of ACLED events",
            )
            st.plotly_chart(
                conflict_fig,
                config=PLOTLY_STATIC_CONFIG,
                use_container_width=True,
            )
        else:
            st.warning(
                "Conflict chart unavailable because neither `acled_events_lag1` nor `acled_events` exists in the CSV."
            )

    with top_condition_cols[1]:
        st.markdown("**Food Price Over Time (average WFP food price in USD)**")
        if food_price_col is not None:
            price_roundly = (
                patterns_df.groupby("round", as_index=False)[food_price_col]
                .mean()
                .sort_values("round", key=lambda s: s.map(round_sort_key))
            )
            price_fig = px.line(
                price_roundly,
                x="round",
                y=food_price_col,
                markers=True,
                labels={
                    "round": "DTM assessment round",
                    food_price_col: "Average WFP food price (USD)",
                },
            )
            price_fig.update_traces(line={"color": "#6b7280", "width": 2.5})
            apply_static_plot_style(
                price_fig,
                "DTM assessment round",
                "Average WFP food price (USD)",
            )
            st.plotly_chart(
                price_fig,
                config=PLOTLY_STATIC_CONFIG,
                use_container_width=True,
            )
        else:
            st.warning(
                "Food price chart unavailable because neither `wfp_avg_usdprice_lag1` nor `wfp_avg_usdprice` exists in the CSV."
            )

    with bottom_condition_cols[0]:
        st.markdown("**Food Insecurity Over Time (population share in IPC Phase 3+)**")
        st.caption(
            "IPC Phase 3+ means the share of people facing Crisis-level or worse food insecurity, including Crisis, Emergency, and Catastrophe/Famine conditions."
        )
        if "ipc_phase3plus_pct" in patterns_df.columns:
            ipc_roundly = (
                patterns_df.groupby("round", as_index=False)["ipc_phase3plus_pct"]
                .mean()
                .sort_values("round", key=lambda s: s.map(round_sort_key))
            )
            ipc_is_proportion = False
            if not ipc_roundly["ipc_phase3plus_pct"].dropna().empty:
                ipc_is_proportion = ipc_roundly["ipc_phase3plus_pct"].dropna().max() <= 1
            if ipc_is_proportion:
                ipc_roundly["ipc_phase3plus_pct"] = ipc_roundly["ipc_phase3plus_pct"] * 100

            ipc_fig = px.line(
                ipc_roundly,
                x="round",
                y="ipc_phase3plus_pct",
                markers=True,
                labels={
                    "round": "DTM assessment round",
                    "ipc_phase3plus_pct": "Population in IPC Phase 3+ (%)",
                },
            )
            ipc_fig.update_traces(line={"color": "#d28b26", "width": 2.5})
            apply_static_plot_style(
                ipc_fig,
                "DTM assessment round",
                "Population in IPC Phase 3+ (%)",
            )
            st.plotly_chart(
                ipc_fig,
                config=PLOTLY_STATIC_CONFIG,
                use_container_width=True,
            )
        else:
            st.warning("Food insecurity chart unavailable because the required column `ipc_phase3plus_pct` is missing.")

    with bottom_condition_cols[1]:
        st.markdown("**Flood Exposure Over Time (number of states with recent flood exposure)**")
        if flood_col == "flood_flag":
            flood_roundly = (
                patterns_df.assign(flood_flag_binary=(patterns_df["flood_flag"].fillna(0) == 1).astype(int))
                .groupby("round", as_index=False)["flood_flag_binary"]
                .sum()
                .sort_values("round", key=lambda s: s.map(round_sort_key))
            )
            flood_fig = px.line(
                flood_roundly,
                x="round",
                y="flood_flag_binary",
                markers=True,
                labels={
                    "round": "DTM assessment round",
                    "flood_flag_binary": "Number of states with recent flood exposure",
                },
            )
            flood_fig.update_traces(line={"color": "#3b82f6", "width": 2.5})
            apply_static_plot_style(
                flood_fig,
                "DTM assessment round",
                "Number of states with recent flood exposure",
            )
            st.plotly_chart(
                flood_fig,
                config=PLOTLY_STATIC_CONFIG,
                use_container_width=True,
            )
        elif flood_col == "flood_affected_people":
            flood_roundly = (
                patterns_df.groupby("round", as_index=False)["flood_affected_people"]
                .mean()
                .sort_values("round", key=lambda s: s.map(round_sort_key))
            )
            flood_fig = px.line(
                flood_roundly,
                x="round",
                y="flood_affected_people",
                markers=True,
                labels={
                    "round": "DTM assessment round",
                    "flood_affected_people": "Average flood-affected people",
                },
            )
            flood_fig.update_traces(line={"color": "#3b82f6", "width": 2.5})
            apply_static_plot_style(
                flood_fig,
                "DTM assessment round",
                "Average flood-affected people",
            )
            st.plotly_chart(
                flood_fig,
                config=PLOTLY_STATIC_CONFIG,
                use_container_width=True,
            )
        else:
            st.warning(
                "Flood exposure chart unavailable because neither `flood_flag` nor `flood_affected_people` exists in the CSV."
            )

    st.markdown(
        """
        <div style="
            background: #f7fafd;
            border: 1px solid #d9e2ee;
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin: 0.85rem 0 0.45rem;
        ">
            <div style="color: #24456f; font-size: 0.96rem; line-height: 1.6;">
                <strong>What to notice:</strong> The contributing conditions shift at different times. Conflict events rise sharply in R14 and remain high in R15, while food price pressure peaks earlier around R10 before declining. Food insecurity also rises in R14, and flood exposure appears mainly in later rounds. This suggests that displacement pressure is shaped by overlapping but uneven conditions, not one single driver.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)
    st.markdown("## Overlapping Pressures Across States")

    with st.expander("What is the CSI Score?", expanded=False):
        st.markdown(
            """
            The Compound Shock Index (CSI) is a way to summarize multiple pressures affecting a region at the same time. It combines four key factors:

            - Conflict events (number of ACLED conflict events)
            - Food insecurity (share of population in IPC Phase 3+)
            - Food price pressure (average WFP food price in USD)
            - Flood exposure (presence of recent flood events: yes/no)

            Each factor is standardized and weighted equally, then combined into a single score from 0 to 100.

            Higher CSI values mean a state is experiencing stronger overall pressure from multiple conditions at once.
            """
        )

    st.subheader("CSI Score Over Time")
    if "csi_0_100" in patterns_df.columns:
        csi_roundly = (
            patterns_df.groupby("round", as_index=False)["csi_0_100"]
            .mean()
            .sort_values("round", key=lambda s: s.map(round_sort_key))
        )
        csi_fig = px.line(
            csi_roundly,
            x="round",
            y="csi_0_100",
            markers=True,
            labels={
                "round": "DTM assessment round",
                "csi_0_100": "Average CSI score (0–100)",
            },
        )
        csi_fig.update_traces(line={"color": "#b91c1c", "width": 3})
        apply_static_plot_style(
            csi_fig,
            "DTM assessment round",
            "Average CSI score (0–100)",
        )
        st.plotly_chart(
            csi_fig,
            config=PLOTLY_STATIC_CONFIG,
            use_container_width=True,
        )
    else:
        st.warning("CSI chart unavailable because the required column `csi_0_100` is missing.")

    st.markdown(
        """
        <div style="
            background: #f7fafd;
            border: 1px solid #d9e2ee;
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin: 0.85rem 0 0.2rem;
        ">
            <div style="color: #24456f; font-size: 0.96rem; line-height: 1.6;">
                <strong>What to notice:</strong> The CSI trend follows a similar shape to the average displacement rate trend: both rise sharply and peak around R14. This suggests that higher combined pressure coincides with higher displacement rates in the monitoring data. Unity also reaches its highest displacement rate around this later period.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
    st.subheader("CSI Score by State Over Time")
    if {"month", "state", "csi_0_100"}.issubset(patterns_df.columns):
        csi_state_df = (
            patterns_df.dropna(subset=["csi_0_100"])
            .groupby(["round", "state"], as_index=False)["csi_0_100"]
            .mean()
            .sort_values("round", key=lambda s: s.map(round_sort_key))
        )
        if not csi_state_df.empty:
            csi_state_fig = px.line(
                csi_state_df,
                x="round",
                y="csi_0_100",
                color="state",
                markers=True,
                labels={
                    "round": "DTM assessment round",
                    "csi_0_100": "CSI score (0–100)",
                    "state": "State",
                },
            )
            csi_state_fig.update_layout(
                height=340,
                dragmode=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
                legend={"title": "State"},
            )
            csi_state_fig.update_xaxes(title_text="DTM assessment round", fixedrange=True)
            csi_state_fig.update_yaxes(title_text="CSI score (0–100)", fixedrange=True)
            st.plotly_chart(
                csi_state_fig,
                config=PLOTLY_STATIC_CONFIG,
                use_container_width=True,
            )
        else:
            st.warning(
                "Combined pressure score by state chart could not be generated because no CSI values were available."
            )
    else:
        missing_csi_state_cols = [
            column for column in ["month", "state", "csi_0_100"] if column not in patterns_df.columns
        ]
        st.warning(
            "Combined pressure score by state chart unavailable because these required columns are missing: "
            + ", ".join(f"`{column}`" for column in missing_csi_state_cols)
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    with st.expander("What are Active Shocks?", expanded=False):
        st.markdown(
            """
            Active shocks count how many contributing conditions are elevated at the same time in a state-round.
            """
        )
    st.subheader("Average Displacement Rate (IDPs per 1,000 people) vs. Number of Active Shocks")
    if "shock_count" in patterns_df.columns and "idp_per_1000" in patterns_df.columns:
        shock_df = (
            patterns_df.dropna(subset=["shock_count", "idp_per_1000"])
            .groupby("shock_count", as_index=False)
            .agg(mean_idp=("idp_per_1000", "mean"))
            .sort_values("shock_count")
        )
        if not shock_df.empty:
            shock_df["shock_count"] = shock_df["shock_count"].astype(int).astype(str)
            shock_fig = px.bar(
                shock_df,
                x="shock_count",
                y="mean_idp",
                text=shock_df["mean_idp"].round(1),
                labels={
                    "shock_count": "Number of active shocks",
                    "mean_idp": "Average IDPs per 1,000 people",
                },
                color_discrete_sequence=["#2b67c8"],
            )
            shock_fig.update_traces(textposition="outside")
            shock_fig.update_layout(
                height=360,
                dragmode=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin={"l": 0, "r": 0, "t": 30, "b": 0},
                showlegend=False,
            )
            shock_fig.update_xaxes(title_text="Number of active shocks", fixedrange=True)
            shock_fig.update_yaxes(title_text="Average IDPs per 1,000 people", fixedrange=True)
            st.plotly_chart(
                shock_fig,
                config=PLOTLY_STATIC_CONFIG,
                use_container_width=True,
            )
        else:
            st.warning(
                "Displacement by active shocks chart could not be generated because no usable `shock_count` values were available."
            )
    elif "shock_count" not in patterns_df.columns:
        st.warning(
            "Displacement by active shocks chart unavailable because the required column `shock_count` is missing."
        )
    else:
        st.warning(
            "Displacement by active shocks chart unavailable because the required column `idp_per_1000` is missing."
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    with st.expander("What do the regime labels mean?", expanded=False):
        st.markdown(
            """
            **Compound:** 2+ conditions are happening at the same time. This is the most complex and high-pressure situation.

            **Conflict:** Displacement is mainly associated with high levels of conflict, while other conditions are less severe.

            **Structural:** Displacement remains high even without strong current shocks. This reflects long-term or persistent conditions where people are unable to return.

            **Stable:** Fewer pressures are present, and displacement levels are lower or improving over time.
            """
        )
    st.subheader("Displacement Regime Matrix with CSI Score")
    if go is None:
        st.warning("Displacement regime matrix unavailable because `plotly.graph_objects` is not available.")
    elif not {"state", "round", "regime", "csi_0_100"}.issubset(patterns_df.columns):
        missing_regime_cols = [
            column for column in ["state", "round", "regime", "csi_0_100"] if column not in patterns_df.columns
        ]
        st.warning(
            "Displacement regime matrix unavailable because these required columns are missing: "
            + ", ".join(f"`{column}`" for column in missing_regime_cols)
        )
    else:
        regime_pivot = patterns_df.pivot_table(
            index="state",
            columns="round",
            values="regime",
            aggfunc=lambda x: x.mode()[0] if len(x) > 0 else "stable",
        )
        csi_pivot = patterns_df.pivot_table(
            index="state",
            columns="round",
            values="csi_0_100",
            aggfunc="mean",
        ).round(1)

        common_states = regime_pivot.index.intersection(csi_pivot.index)
        common_rounds = regime_pivot.columns.intersection(csi_pivot.columns)
        regime_pivot = regime_pivot.loc[common_states, common_rounds]
        csi_pivot = csi_pivot.loc[common_states, common_rounds]
        regime_pivot = regime_pivot.reindex(columns=sorted(regime_pivot.columns, key=round_sort_key))
        csi_pivot = csi_pivot.reindex(columns=sorted(csi_pivot.columns, key=round_sort_key))
        csi_numeric = csi_pivot.copy()

        if csi_numeric.empty:
            st.warning(
                "Displacement regime matrix could not be generated because no state-round CSI values were available."
            )
        else:
            annot_text = []
            for state in regime_pivot.index:
                row_text = []
                for rnd in regime_pivot.columns:
                    reg = regime_pivot.loc[state, rnd]
                    csi_val = csi_pivot.loc[state, rnd]
                    if pd.notna(csi_val) and str(reg) not in ("", "nan"):
                        row_text.append(f"{reg}<br>{csi_val}")
                    elif pd.notna(csi_val):
                        row_text.append(f"{csi_val}")
                    else:
                        row_text.append("")
                annot_text.append(row_text)

            regime_fig = go.Figure(
                go.Heatmap(
                    z=csi_numeric.values.tolist(),
                    x=[str(c) for c in csi_numeric.columns],
                    y=list(csi_numeric.index),
                    text=annot_text,
                    texttemplate="%{text}",
                    textfont={"size": 9, "color": "black"},
                    colorscale="OrRd",
                    zmin=0,
                    zmax=100,
                    showscale=True,
                    colorbar={
                        "title": "CSI score",
                        "len": 0.75,
                        "thickness": 18,
                        "y": 0.5,
                    },
                    xgap=2,
                    ygap=2,
                    hovertemplate="<b>%{y}</b><br>Round: %{x}<br>%{text}<extra></extra>",
                    hoverongaps=False,
                )
            )
            regime_fig.update_layout(
                height=360,
                dragmode=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
            )
            regime_fig.update_xaxes(title_text="DTM assessment round", fixedrange=True)
            regime_fig.update_yaxes(title_text="State", fixedrange=True)
            st.plotly_chart(
                regime_fig,
                config=PLOTLY_STATIC_CONFIG,
                use_container_width=True,
            )

    st.markdown(
        """
        <div style="
            background: #f7fafd;
            border: 1px solid #d9e2ee;
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin: 0.85rem 0 0.2rem;
        ">
            <div style="color: #24456f; font-size: 0.96rem; line-height: 1.6;">
                <strong>What to notice:</strong> The regime matrix shows that many states shift toward compound conditions in R14 and R15, meaning multiple pressures are elevated at once. Jonglei, Upper Nile, Lakes, Warrap, and Unity show especially high CSI values in later rounds. Stable and structural labels appear more often in earlier or lower-pressure state-rounds.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not loaded_from_csv:
        st.caption(
            "The Analysis page is currently showing fallback dummy data because the CSV could not be loaded."
        )


def render_pre_post_war_page() -> None:
    df, loaded_from_csv = load_dashboard_data()

    if pd is None or df is None:
        st.markdown(
            """
            <div class="content-card dashboard-placeholder">
                <div class="section-title">Pre/Post War</div>
                <p class="body-copy">
                    Pandas is not available in this environment, so this page cannot load yet.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if px is None:
        st.warning("Plotly is required to render the Pre/Post War page charts.")
        return

    required_columns = ["state", "period", "idp_per_1000"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        st.warning(
            "Pre/Post War page cannot be rendered because these required columns are missing: "
            + ", ".join(f"`{column}`" for column in missing_columns)
        )
        return

    st.subheader("Pre/Post War")
    st.markdown(
        """
        <div style="
            background: #ffffff;
            border: 1px solid #d9e2ee;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: 0.35rem 0 1rem;
        ">
            <div style="
                color: #10233a;
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.55rem;
            ">Sudan Civil War</div>
            <div style="
                color: #4d5e72;
                font-size: 0.96rem;
                line-height: 1.6;
            ">
                In April 2023, Sudan experienced a major outbreak of internal armed conflict between rival military groups. While the conflict is centered in Sudan, it has had strong regional effects, including increased displacement pressure, disrupted trade, and rising food prices in South Sudan.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    period_df = (
        df.dropna(subset=["state", "period", "idp_per_1000"])
        .groupby(["state", "period"], as_index=False)["idp_per_1000"]
        .mean()
    )

    if period_df.empty:
        st.warning("Pre/Post War chart could not be generated because no usable period data was available.")
        return

    period_label_map = {
        "pre_war": "Pre-war",
        "post_war": "Post-war",
    }
    period_df["period_label"] = period_df["period"].map(period_label_map)
    period_df = period_df.dropna(subset=["period_label"])

    if period_df.empty:
        st.warning("Pre/Post War chart could not be generated because `period` values were unavailable or unsupported.")
        return

    post_war_order = (
        period_df[period_df["period_label"] == "Post-war"]
        .sort_values("idp_per_1000", ascending=False)["state"]
        .tolist()
    )
    remaining_states = [state for state in period_df["state"].unique().tolist() if state not in post_war_order]
    state_order = post_war_order + sorted(remaining_states)

    pre_post_fig = px.bar(
        period_df,
        x="state",
        y="idp_per_1000",
        color="period_label",
        barmode="group",
        category_orders={
            "state": state_order,
            "period_label": ["Pre-war", "Post-war"],
        },
        color_discrete_map={
            "Pre-war": "#2b67c8",
            "Post-war": "#c0392b",
        },
        labels={
            "state": "State",
            "idp_per_1000": "Average IDPs per 1,000 people",
            "period_label": "Period",
        },
    )
    pre_post_fig.update_layout(
        dragmode=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"title": "Period", "orientation": "h", "y": 1.08, "x": 0},
    )
    pre_post_fig.update_xaxes(title_text="State", fixedrange=True, tickangle=-30)
    pre_post_fig.update_yaxes(title_text="Average IDPs per 1,000 people", fixedrange=True)

    st.subheader("Displacement by State in South Sudan Pre/Post April 2023 (IDPs per 1,000 people)")
    st.plotly_chart(
        pre_post_fig,
        config=PLOTLY_STATIC_CONFIG,
        use_container_width=True,
    )

    if not loaded_from_csv:
        st.caption(
            "The Pre/Post War page is currently showing fallback dummy data because the CSV could not be loaded."
        )


def render_choreograph_page() -> None:
    df, loaded_from_csv = load_dashboard_data()

    if pd is None or df is None:
        st.markdown(
            """
            <div class="content-card">
                <div class="section-title">Maps</div>
                <p class="body-copy">
                    Pandas is not available in this environment, so the Maps page cannot load yet.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    geojson = load_state_geojson()
    round_labels = build_round_labels(df)
    round_options = sorted(
        df["round"].dropna().astype(str).unique().tolist(),
        key=round_sort_key,
    )
    selected_round = st.selectbox(
        "Time Period / DTM Round",
        options=round_options,
        index=len(round_options) - 1 if round_options else 0,
        format_func=lambda value: round_labels.get(value, value),
    )
    st.caption(
        "Displacement data is collected in periodic DTM assessment rounds, so each round represents a snapshot in time."
    )

    round_df = df[df["round"].astype(str) == selected_round].copy()
    available_states = set(round_df["state"].dropna().astype(str).unique().tolist())
    all_states = sorted(
        {
            feature.get("properties", {}).get("shapeName")
            for feature in (geojson or {}).get("features", [])
            if feature.get("properties", {}).get("shapeName")
        }
        or set(df["state"].dropna().astype(str).unique().tolist())
    )

    layout_cols = st.columns([0.3, 0.7], gap="large")
    with layout_cols[0]:
        st.markdown(
            """
            <div class="state-filter-panel">
                <div class="section-title">State Filter</div>
            """,
            unsafe_allow_html=True,
        )
        selected_states = []
        for state in all_states:
            is_available = state in available_states
            if st.checkbox(
                state,
                value=is_available,
                disabled=not is_available,
                key=f"risk_map_state_{selected_round}_{state}",
            ):
                selected_states.append(state)
        st.markdown("</div>", unsafe_allow_html=True)

    filtered_round_df = round_df[round_df["state"].isin(selected_states)].copy()
    if filtered_round_df.empty:
        filtered_round_df = round_df.iloc[0:0].copy()

    with layout_cols[1]:
        summary_source = filtered_round_df if not filtered_round_df.empty else round_df.iloc[0:0].copy()
        csi_source = summary_source.dropna(subset=["csi_0_100"]).copy()
        avg_csi = csi_source["csi_0_100"].mean() if not csi_source.empty else None
        highest_state = (
            csi_source.sort_values("csi_0_100", ascending=False).iloc[0]["state"]
            if not csi_source.empty
            else "Unavailable"
        )
        summary_cols = st.columns(2, gap="medium")
        with summary_cols[0]:
            render_metric_card("Average Score", f"{avg_csi:,.1f}" if avg_csi is not None else "Unavailable")
        with summary_cols[1]:
            render_metric_card("Highest-Score State", str(highest_state))

        st.markdown(
            """
            <div class="map-title">South Sudan Regional Displacement and Risk Map</div>
            """,
            unsafe_allow_html=True,
        )

        if px is not None and go is not None and geojson is not None:
            state_lookup_df = pd.DataFrame(
                [
                    {
                        "state": feature.get("properties", {}).get("shapeName"),
                        "state_match": feature.get("properties", {}).get("state_match"),
                    }
                    for feature in geojson.get("features", [])
                    if feature.get("properties", {}).get("shapeName")
                ]
            )
            round_map_df = state_lookup_df.merge(
                round_df[
                    ["state", "csi_0_100", "idp_per_1000", "dtm_idp_ind", "regime"]
                ].drop_duplicates(subset=["state"]),
                on="state",
                how="left",
            )
            round_map_df["is_selected"] = round_map_df["state"].isin(selected_states)
            deselected_map_df = round_map_df[~round_map_df["is_selected"]].copy()
            selected_map_df = round_map_df[round_map_df["is_selected"]].copy()
            missing_map_df = selected_map_df[selected_map_df["csi_0_100"].isna()].copy()
            colored_map_df = selected_map_df[selected_map_df["csi_0_100"].notna()].copy()

            fig = go.Figure()
            if not deselected_map_df.empty:
                fig.add_trace(
                    go.Choroplethmapbox(
                        geojson=geojson,
                        featureidkey="properties.state_match",
                        locations=deselected_map_df["state_match"],
                        z=[1] * len(deselected_map_df),
                        colorscale=[[0, "#cfd5dd"], [1, "#cfd5dd"]],
                        showscale=False,
                        marker_opacity=0.72,
                        marker_line_color="#545454",
                        marker_line_width=1.4,
                        customdata=deselected_map_df[["state"]].values,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Not selected in current filter<extra></extra>"
                        ),
                    )
                )

            if not missing_map_df.empty:
                fig.add_trace(
                    go.Choroplethmapbox(
                        geojson=geojson,
                        featureidkey="properties.state_match",
                        locations=missing_map_df["state_match"],
                        z=[1] * len(missing_map_df),
                        colorscale=[[0, "#d6dbe3"], [1, "#d6dbe3"]],
                        showscale=False,
                        marker_opacity=0.92,
                        marker_line_color="#2f2f2f",
                        marker_line_width=1.6,
                        customdata=missing_map_df[["state", "idp_per_1000", "dtm_idp_ind"]].values,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Score: Missing<br>"
                            "IDPs per 1,000: %{customdata[1]:.1f}<br>"
                            "Raw IDP count: %{customdata[2]:,.0f}<extra></extra>"
                        ),
                    )
                )

            if not colored_map_df.empty:
                fig.add_trace(
                    go.Choroplethmapbox(
                        geojson=geojson,
                        featureidkey="properties.state_match",
                        locations=colored_map_df["state_match"],
                        z=colored_map_df["csi_0_100"],
                        zmin=0,
                        zmax=100,
                        colorscale="Reds",
                        colorbar={
                            "title": "Score",
                            "thickness": 18,
                            "len": 0.75,
                            "y": 0.5,
                        },
                        marker_opacity=0.9,
                        marker_line_color="#2f2f2f",
                        marker_line_width=1.6,
                        customdata=colored_map_df[
                            ["state", "csi_0_100", "idp_per_1000", "dtm_idp_ind"]
                        ].values,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Score: %{customdata[1]:.1f}<br>"
                            "IDPs per 1,000: %{customdata[2]:.1f}<br>"
                            "Raw IDP count: %{customdata[3]:,.0f}<extra></extra>"
                        ),
                    )
                )

            fig.add_trace(
                go.Choroplethmapbox(
                    geojson=geojson,
                    featureidkey="properties.state_match",
                    locations=state_lookup_df["state_match"],
                    z=[0] * len(state_lookup_df),
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    showscale=False,
                    marker_opacity=0,
                    marker_line_color="#111111",
                    marker_line_width=2.1,
                    hoverinfo="skip",
                )
            )

            fig.update_layout(
                height=455,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                paper_bgcolor="#edf1f5",
                plot_bgcolor="#edf1f5",
                mapbox={
                    "style": "carto-positron",
                    "center": {"lat": 7.35, "lon": 30.1},
                    "zoom": 4.7,
                },
            )
            st.plotly_chart(fig, config={"responsive": True})
        elif filtered_round_df.empty:
            st.info("Select at least one state to display the map.")
        else:
            missing_parts = []
            if px is None:
                missing_parts.append("`plotly` is not installed")
            if go is None:
                missing_parts.append("`plotly.graph_objects` is not available")
            if geojson is None:
                missing_parts.append(f"`{STATE_GEOJSON_PATH}` is missing")
            missing_message = " and ".join(missing_parts)
            st.warning(
                f"The regional map needs {missing_message}. Once both are available, this page will render."
            )

    if not loaded_from_csv:
        st.caption(
            "The regional map page is currently showing fallback dummy data because the CSV could not be loaded."
        )


st.markdown(
    """
    <style>
        .stApp {
            background: #edf1f5;
            color: #17212f;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 0;
            padding-bottom: 2.5rem;
        }

        .main .block-container {
            padding-top: 0.2rem;
        }

        div[data-testid="stAppViewContainer"] > .main {
            padding-top: 0;
        }

        header[data-testid="stHeader"] {
            height: 0;
            min-height: 0;
            background: transparent;
        }

        div[data-testid="stDecoration"] {
            display: none;
        }

        div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0;
            position: fixed;
        }

        .top-header {
            background: linear-gradient(135deg, #2b67c8 0%, #2158ae 100%);
            color: #ffffff;
            border-radius: 0 0 24px 24px;
            box-shadow: 0 16px 36px rgba(33, 88, 174, 0.24);
            margin: 0 -1rem 1.4rem;
            padding: 1.35rem 1.5rem 1.15rem;
        }

        .header-title {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .header-subtitle {
            font-size: 0.98rem;
            opacity: 0.92;
            margin-top: 0.3rem;
        }

        div[role="radiogroup"] {
            background: transparent;
            display: inline-flex;
            border-radius: 999px;
            padding: 0;
            gap: 0.35rem;
        }

        div[role="radiogroup"] label {
            background: #e5eaf1;
            border: 1px solid transparent;
            border-radius: 999px;
            padding: 0;
            overflow: hidden;
            cursor: pointer;
            transition: background 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
        }

        div[role="radiogroup"] label:hover {
            background: #d7deea;
            box-shadow: 0 8px 18px rgba(27, 47, 94, 0.10);
        }

        div[role="radiogroup"] label[data-checked="true"],
        div[role="radiogroup"] label:has(input:checked),
        div[role="radiogroup"] label[aria-checked="true"] {
            background: #2b67c8;
            border-color: #1f4f96;
            box-shadow: 0 10px 22px rgba(27, 47, 94, 0.20);
            transform: translateY(-1px);
        }

        div[role="radiogroup"] label > div:first-child {
            display: none;
        }

        div[role="radiogroup"] label > div:last-child {
            padding: 0.55rem 1rem;
        }

        div[role="radiogroup"] label span {
            color: #334155;
            font-weight: 600;
        }

        div[role="radiogroup"] label[data-checked="true"] span,
        div[role="radiogroup"] label:has(input:checked) span,
        div[role="radiogroup"] label[aria-checked="true"] span,
        div[role="radiogroup"] label[data-checked="true"] p,
        div[role="radiogroup"] label:has(input:checked) p,
        div[role="radiogroup"] label[aria-checked="true"] p,
        div[role="radiogroup"] label[data-checked="true"] * ,
        div[role="radiogroup"] label:has(input:checked) *,
        div[role="radiogroup"] label[aria-checked="true"] * {
            color: #ffffff;
        }

        .content-card {
            background: #ffffff;
            border: 1px solid #d9e2ee;
            border-radius: 22px;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
            padding: 1.45rem 1.45rem 1.35rem;
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #10233a;
            margin-bottom: 0.85rem;
        }

        .body-copy {
            color: #4d5e72;
            font-size: 1rem;
            line-height: 1.62;
            margin-bottom: 0.95rem;
        }

        .body-copy-tight {
            margin-bottom: 0.6rem;
        }

        .explore-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
            margin-bottom: 1rem;
        }

        .mini-card {
            background: #f5f8fc;
            border: 1px solid #dbe5f0;
            border-radius: 16px;
            padding: 1rem;
            color: #243447;
            font-weight: 600;
            line-height: 1.45;
            min-height: 96px;
            display: flex;
            align-items: center;
        }

        .note-box {
            background: #eef5ff;
            border: 1px solid #cddcf7;
            color: #24456f;
            border-radius: 16px;
            padding: 0.95rem 1rem;
            line-height: 1.55;
            font-size: 0.96rem;
        }

        .stat-card {
            background: #ffffff;
            border: 1px solid #d9e2ee;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
            margin-top: 0.2rem;
        }

        .stat-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #1b4e9b;
            line-height: 1.1;
            margin-bottom: 0.35rem;
        }

        .stat-label {
            color: #637487;
            font-size: 0.92rem;
            font-weight: 600;
        }

        .preview-placeholder {
            min-height: 320px;
            border-radius: 18px;
            border: 1px dashed #b8c7d8;
            background: linear-gradient(180deg, #f7f9fc 0%, #edf2f7 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #5d6d81;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .about-hero {
            padding: 0.15rem 0 0.55rem;
            margin-bottom: 0.75rem;
        }

        .about-hero-quote {
            color: #b42318;
            font-size: 2.55rem;
            font-style: italic;
            font-weight: 600;
            line-height: 1.22;
            letter-spacing: -0.01em;
            max-width: 1120px;
            white-space: nowrap;
            margin-bottom: 0.9rem;
        }

        .about-hero-stat-row {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.35rem;
            flex-wrap: nowrap;
        }

        .about-hero-stat {
            color: #1b4e9b;
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.03em;
            flex: 0 0 auto;
            margin-bottom: 0;
        }

        .about-hero-label {
            color: #243447;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.18;
            max-width: 760px;
            margin-bottom: 0;
        }

        .about-hero-support {
            color: #46586e;
            font-size: 1.02rem;
            line-height: 1.55;
            max-width: 620px;
            margin-top: 0.3rem;
        }

        .about-grid {
            display: grid;
            grid-template-columns: 58% 42%;
            gap: 32px;
            align-items: stretch;
            width: 100%;
            height: 100%;
            margin-bottom: 0.5rem;
        }

        .about-col {
            display: flex;
            flex-direction: column;
            gap: 12px;
            height: 100%;
        }

        .about-grid-card {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            height: auto;
            padding: 1.15rem 1.2rem 1.05rem;
        }

        .about-grid-bottom {
            margin-top: 0;
        }

        .about-map-block {
            display: flex;
            flex-direction: column;
            justify-content: stretch;
            width: 100%;
            margin-bottom: 1rem;
        }

        .about-map-image {
            width: 100%;
            aspect-ratio: 1.15 / 1;
            object-fit: cover;
            border-radius: 16px;
            display: block;
        }

        .preview-card {
            background: #ffffff;
            border: 1px solid #d9e2ee;
            border-radius: 22px;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
            overflow: hidden;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }

        .preview-card-image {
            width: 100%;
            height: 220px;
            object-fit: cover;
            display: block;
            background: #eef3f8;
        }

        .preview-card-placeholder {
            width: 100%;
            height: 220px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(180deg, #f7f9fc 0%, #edf2f7 100%);
            color: #5d6d81;
            font-weight: 700;
            border-bottom: 1px solid #d9e2ee;
        }

        .preview-card-body {
            padding: 1rem 1.15rem 0;
            flex: 1 1 auto;
        }

        .preview-card-title {
            color: #10233a;
            font-size: 1.08rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .preview-card-copy {
            color: #4d5e72;
            font-size: 0.96rem;
            line-height: 1.58;
        }

        .learn-more-button-wrapper {
            padding: 0 1.15rem 1.1rem;
            margin-top: 0.55rem;
        }

        .compact-card {
            margin-top: 1rem;
        }

        .use-list {
            margin: 0;
            padding-left: 1.2rem;
            color: #4d5e72;
            line-height: 1.8;
            font-size: 1rem;
        }

        .filter-shell {
            background: #ffffff;
            border: 1px solid #d9e2ee;
            border-top: 5px solid #2b67c8;
            border-radius: 18px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
            padding: 0.2rem 0.95rem 0.4rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #d9e2ee;
            border-radius: 18px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
            padding: 1rem 1.05rem;
            margin-bottom: 1rem;
        }

        .metric-label {
            color: #6a7b8d;
            font-size: 0.88rem;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }

        .metric-value {
            color: #143764;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.1;
        }

        .eda-caption {
            color: #5c6c7f;
            font-size: 0.95rem;
            line-height: 1.6;
            padding-top: 2rem;
        }

        .state-filter-panel {
            position: sticky;
            top: 1rem;
            padding-top: 0.15rem;
        }

        .map-title {
            color: #10233a;
            font-size: 1.22rem;
            font-weight: 700;
            text-align: center;
            margin: 0.25rem 0 0.85rem;
        }

        div[data-testid="stSelectbox"] > div {
            background: #ffffff;
            border-radius: 14px;
        }

        div[data-testid="stSelectbox"] label {
            color: #10233a;
            font-weight: 700;
        }

        div[data-testid="stSelectbox"] [data-baseweb="select"] {
            background: #ffffff;
            border: 1px solid #d9e2ee;
            border-radius: 14px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: #ffffff;
        }

        div[data-testid="stSelectbox"] svg {
            color: #4d5e72;
        }

        div[data-testid="stPlotlyChart"] {
            background: transparent;
        }

        .learn-more-button-wrapper div[data-testid="stButton"] {
            margin: 0;
        }

        .learn-more-button-wrapper div[data-testid="stButton"] > button {
            background: #ef4444;
            color: #ffffff;
            border: 1px solid #ef4444;
            border-radius: 12px;
            padding: 0.42rem 1rem;
            font-size: 0.94rem;
            font-weight: 700;
            line-height: 1.2;
            box-shadow: 0 8px 18px rgba(239, 68, 68, 0.18);
        }

        .learn-more-button-wrapper div[data-testid="stButton"] > button:hover {
            background: #dc2626;
            border-color: #dc2626;
            color: #ffffff;
        }

        .learn-more-button-wrapper div[data-testid="stButton"] > button:active,
        .learn-more-button-wrapper div[data-testid="stButton"] > button:focus,
        .learn-more-button-wrapper div[data-testid="stButton"] > button:focus-visible {
            background: #dc2626;
            border-color: #dc2626;
            color: #ffffff;
        }

        .learn-more-button-wrapper + div[data-testid="stButton"] {
            margin-top: -0.45rem;
            margin-bottom: 1rem;
            padding-left: 1.15rem;
        }

        .learn-more-button-wrapper + div[data-testid="stButton"] > button {
            background: linear-gradient(180deg, #ff5a4f 0%, #e23d34 100%);
            color: #ffffff;
            border: 1px solid #d93a30;
            border-radius: 12px;
            padding: 0.42rem 1rem;
            font-size: 0.94rem;
            font-weight: 700;
            box-shadow: 0 10px 20px rgba(226, 61, 52, 0.16);
        }

        .learn-more-button-wrapper + div[data-testid="stButton"] > button:hover {
            border-color: #cd332a;
            color: #ffffff;
        }

        .dashboard-placeholder {
            min-height: 260px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        @media (max-width: 900px) {
            .top-header {
                margin-left: -0.5rem;
                margin-right: -0.5rem;
                border-radius: 0 0 20px 20px;
            }

            .about-hero-quote {
                white-space: normal;
                font-size: 1.55rem;
            }

            .about-hero-stat-row {
                flex-wrap: wrap;
                align-items: flex-start;
            }

            .explore-grid {
                grid-template-columns: 1fr;
            }

            .about-grid {
                grid-template-columns: 1fr;
                gap: 24px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


if "active_page" not in st.session_state:
    st.session_state["active_page"] = "About"


stats = load_summary_stats()

st.markdown(
    """
    <div class="top-header">
        <div class="header-title">Forced Displacement Tracker : South Sudan</div>
        <div class="header-subtitle">
            Monitoring displacement patterns, contributing conditions, and regional variation across South Sudan        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


selected_page = st.radio(
    "Navigation",
    ["About", "Analysis", "Maps"],
    index=["About", "Analysis", "Maps"].index(st.session_state["active_page"])
    if st.session_state["active_page"] in ["About", "Analysis", "Maps"]
    else 0,
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state["active_page"] = selected_page


if selected_page == "About":
    render_about_page(stats)
elif selected_page == "Analysis":
    render_eda()
else:
    render_choreograph_page()
