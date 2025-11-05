import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------
# Constants and configuration
# ---------------------------
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(WORKSPACE_DIR)  # go up from pages/
OUTCOME_FILE = os.path.join(WORKSPACE_DIR, "ATL Speech - Correlation Workflow - 136 Outcome.csv")
PROGRAM_FILE = os.path.join(WORKSPACE_DIR, "ATL Speech - Correlation Workflow - Presence of Aud Program.csv")

STANDARD_COLUMN_STATE = "State"
STANDARD_COLUMN_PROGRAM = "Audiology Program Presence"
METRIC_COLUMNS_CANONICAL = {
    "% Meeting 1": ["% meeting 1", "% Meeting 1", "Percent Meeting 1", "Meeting1", "Pct Meeting 1"],
    "% Meeting 3": ["% meeting 3", "% Meeting 3", "Percent Meeting 3", "Meeting3", "Pct Meeting 3"],
    "% Meeting 6": ["% meeting 6", "% Meeting 6", "Percent Meeting 6", "Meeting6", "Pct Meeting 6"],
}

STATE_TO_USPS = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
    "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
    "Wyoming": "WY"
}

# CMV-required states list
CMV_STATES = {
    "Connecticut", "Florida", "Iowa", "Kentucky", "New York", "Pennsylvania", "Utah", "Virginia"
}


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: c.strip() for c in df.columns}
    # state
    for cand in ["State", "State (rank)", "State (Sort by pop ratio)", "Jurisdiction", "Location", "State/Territory"]:
        if cand in renamed.values():
            orig = next(k for k, v in renamed.items() if v == cand)
            renamed[orig] = STANDARD_COLUMN_STATE
            break
    # metrics
    present = set(renamed.values()) | {v.lower() for v in renamed.values()}
    inv = {v: k for k, v in renamed.items()}
    for canonical, aliases in METRIC_COLUMNS_CANONICAL.items():
        if canonical in present:
            continue
        for alias in aliases:
            if alias in present or alias.lower() in present:
                orig = inv.get(alias)
                if not orig:
                    for k, v in renamed.items():
                        if v.lower() == alias.lower():
                            orig = k
                            break
                if orig:
                    renamed[orig] = canonical
                break
    # program column
    for cand in ["Audiology Program Presence", "Presence of Audiology Program", "Program Presence", "Has Program"]:
        if cand in renamed.values():
            orig = next(k for k, v in renamed.items() if v == cand)
            renamed[orig] = STANDARD_COLUMN_PROGRAM
            break
    return df.rename(columns=renamed)


def coerce_program_presence(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    true_values = {"yes", "true", "1", "y", "present", "with", "has", "available"}
    false_values = {"no", "false", "0", "n", "absent", "without", "none"}
    return s.map(lambda x: True if x in true_values else (False if x in false_values else np.nan))


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    outcome = standardize_columns(read_csv(OUTCOME_FILE))
    program = standardize_columns(read_csv(PROGRAM_FILE))

    if STANDARD_COLUMN_PROGRAM not in program.columns:
        other_cols = [c for c in program.columns if c != STANDARD_COLUMN_STATE]
        if other_cols:
            program[STANDARD_COLUMN_PROGRAM] = coerce_program_presence(program[other_cols[-1]])

    # normalize state names
    outcome["_key"] = outcome[STANDARD_COLUMN_STATE].astype(str).str.strip()
    program["_key"] = program[STANDARD_COLUMN_STATE].astype(str).str.strip()

    merged = pd.merge(
        outcome,
        program[["_key", STANDARD_COLUMN_PROGRAM]],
        on="_key",
        how="inner",
        validate="m:1",
    )
    merged[STANDARD_COLUMN_STATE] = merged[STANDARD_COLUMN_STATE].astype(str).str.strip()
    # keep columns
    keep = [STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM] + list(METRIC_COLUMNS_CANONICAL.keys())
    keep += [c for c in ["Year"] if c in merged.columns]
    merged = merged[[c for c in keep if c in merged.columns]].copy()
    # numerics
    for metric in METRIC_COLUMNS_CANONICAL.keys():
        if metric in merged.columns:
            merged[metric] = pd.to_numeric(
                merged[metric].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip(),
                errors='coerce'
            )
    merged[STANDARD_COLUMN_PROGRAM] = coerce_program_presence(merged[STANDARD_COLUMN_PROGRAM])
    merged = merged.dropna(subset=[STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM])
    merged["State_Code"] = merged[STANDARD_COLUMN_STATE].map(STATE_TO_USPS)
    return merged


st.set_page_config(page_title="CMV States - 1-3-6 Outcomes", layout="wide")
st.markdown("<div class='page-title'>CMV-Required States: 1–3–6 Outcomes</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Connecticut, Florida, Iowa, Kentucky, New York, Pennsylvania, Utah, Virginia</div>", unsafe_allow_html=True)

data = load_data()

with st.sidebar:
    st.header("CMV States Filters")
    available_metrics = [m for m in METRIC_COLUMNS_CANONICAL.keys() if m in data.columns]
    metric = st.selectbox("Outcome Metric", options=available_metrics, index=0 if available_metrics else None)


def filter_cmv(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[STANDARD_COLUMN_STATE].isin(CMV_STATES)].copy()


if data.empty or not metric:
    st.warning("No data available.")
    st.stop()

cmv_df = filter_cmv(data)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-header'>Distribution by Program Presence (CMV States)</div>", unsafe_allow_html=True)
    tmp = cmv_df.dropna(subset=[metric]).copy()
    tmp["Program"] = np.where(tmp[STANDARD_COLUMN_PROGRAM], "With Program", "Without Program")
    fig_box = px.box(
        tmp,
        x="Program",
        y=metric,
        points="all",
        color="Program",
        color_discrete_sequence=px.colors.sequential.Viridis,
        hover_name=STANDARD_COLUMN_STATE,
        hover_data={metric: ":.2f"},
    )
    fig_box.update_layout(margin=dict(t=10, b=20, l=10, r=10), yaxis_title=f"{metric} (%)", showlegend=False)
    st.plotly_chart(fig_box, width='stretch')

with col2:
    st.markdown("<div class='section-header'>U.S. Choropleth (CMV States Highlighted)</div>", unsafe_allow_html=True)
    tmp = cmv_df.dropna(subset=[metric, "State_Code"]).copy()
    if "Year" in tmp.columns:
        tmp = tmp.sort_values("Year", ascending=False).groupby("State_Code").first().reset_index()
    else:
        tmp = tmp.groupby("State_Code", as_index=False).agg({metric: 'mean', STANDARD_COLUMN_STATE: 'first'})
    # For consistent coloring, compute global range among CMV states
    color_range = [float(tmp[metric].min()), float(tmp[metric].max())] if not tmp.empty else None
    fig_map = px.choropleth(
        tmp,
        locations="State_Code",
        locationmode="USA-states",
        color=metric,
        color_continuous_scale=px.colors.sequential.Viridis[::-1],
        range_color=color_range,
        scope="usa",
        hover_name=STANDARD_COLUMN_STATE,
        title=None,
    )
    fig_map.update_coloraxes(colorbar_title=f"{metric} (%)")
    fig_map.update_layout(margin=dict(t=10, b=20, l=10, r=10))
    st.plotly_chart(fig_map, width='stretch')


