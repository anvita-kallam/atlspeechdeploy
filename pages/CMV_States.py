import os
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import streamlit as st

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

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
    "% Meeting 1": [
        "% Meeting 1", "Percent Meeting 1", "Meeting1", "Pct Meeting 1",
        "% meeting 1", "percent meeting 1", "pct meeting 1"
    ],
    "% Meeting 3": [
        "% Meeting 3", "Percent Meeting 3", "Meeting3", "Pct Meeting 3",
        "% meeting 3", "percent meeting 3", "pct meeting 3"
    ],
    "% Meeting 6": [
        "% Meeting 6", "Percent Meeting 6", "Meeting6", "Pct Meeting 6",
        "% meeting 6", "percent meeting 6", "pct meeting 6"
    ],
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

# Fix typos found in the CSV files (e.g., "Noebraska" -> "Nebraska")
STATE_NAME_CORRECTIONS = {
    "noebraska": "nebraska",
    "noevada": "nevada",
    "noew hampshire": "new hampshire",
    "noew jersey": "new jersey",
    "noew mexico": "new mexico",
    "noew york": "new york",
}

# CMV-required states list
CMV_STATES = {
    "Connecticut", "Florida", "Iowa", "Kentucky", "New York", "Pennsylvania", "Utah", "Virginia"
}


def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: c.strip() for c in df.columns}

    found_state_key = None
    for candidate in [
        "State", "state", "STATE", "State Name", "Jurisdiction", "State/Territory",
        "STATE_NAME", "Location", "Location Name", "State/Area"
    ]:
        if candidate in renamed.values():
            found_state_key = next(k for k, v in renamed.items() if v == candidate)
            renamed[found_state_key] = STANDARD_COLUMN_STATE
            break
    if STANDARD_COLUMN_STATE not in renamed.values():
        for original, clean in renamed.items():
            low = clean.lower()
            if ("state" in low) or (low in {"jurisdiction", "location"}):
                renamed[original] = STANDARD_COLUMN_STATE
                found_state_key = original
                break

    present_cols = set(renamed.values()) | {v.lower() for v in renamed.values()}
    inverse = {v: k for k, v in renamed.items()}
    for canonical, aliases in METRIC_COLUMNS_CANONICAL.items():
        if canonical in present_cols:
            continue
        for alias in aliases:
            if alias in present_cols or alias.lower() in present_cols:
                original = inverse.get(alias, None)
                if original is None:
                    for k, v in renamed.items():
                        if v.lower() == alias.lower():
                            original = k
                            break
                if original is None:
                    continue
                renamed[original] = canonical
                break

    for candidate in [
        "Audiology Program Presence", "Presence of Audiology Program", "Program", "Has Program", "Presence", "Audiology Program",
        "Audiology Presence", "Program Presence", "Has Audiology Program"
    ]:
        if candidate in renamed.values():
            original = next(k for k, v in renamed.items() if v == candidate)
            renamed[original] = STANDARD_COLUMN_PROGRAM
            break

    return df.rename(columns=renamed)


def coerce_program_presence(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    true_values = {"yes", "true", "1", "y", "present", "with", "has", "have", "available"}
    false_values = {"no", "false", "0", "n", "absent", "without", "none", "not available"}
    return s.map(lambda x: True if x in true_values else (False if x in false_values else np.nan))


@st.cache_data(show_spinner=False)
def prepare_data(outcome_path: str, program_path: str) -> pd.DataFrame:
    outcome_df = standardize_columns(read_csv_safely(outcome_path))
    program_df = standardize_columns(read_csv_safely(program_path))

    if STANDARD_COLUMN_PROGRAM not in program_df.columns:
        candidate_cols = [c for c in program_df.columns if c != STANDARD_COLUMN_STATE]
        if len(candidate_cols) > 0:
            program_df[STANDARD_COLUMN_PROGRAM] = coerce_program_presence(program_df[candidate_cols[-1]])

    if STANDARD_COLUMN_STATE not in outcome_df.columns:
        raise KeyError(f"State column not found in outcome data. Columns: {list(outcome_df.columns)}")
    if STANDARD_COLUMN_STATE not in program_df.columns:
        raise KeyError(f"State column not found in program data. Columns: {list(program_df.columns)}")
    if STANDARD_COLUMN_PROGRAM not in program_df.columns:
        raise KeyError("Program presence column not found/derivable in program data.")

    # Normalize state name for robust merging (case-insensitive, trimmed, and fix typos)
    def normalize_state_name(state_name: str) -> str:
        normalized = str(state_name).strip().lower()
        return STATE_NAME_CORRECTIONS.get(normalized, normalized)
    
    outcome_df["_state_key"] = outcome_df[STANDARD_COLUMN_STATE].apply(normalize_state_name)
    program_df["_state_key"] = program_df[STANDARD_COLUMN_STATE].apply(normalize_state_name)

    # Include audiologists per 100k population column
    audiologist_col = None
    for col in program_df.columns:
        if "audiologist" in col.lower() and "100k" in col.lower():
            audiologist_col = col
            break
    
    merge_cols = ["_state_key", STANDARD_COLUMN_PROGRAM]
    if audiologist_col:
        merge_cols.append(audiologist_col)
    
    merged = pd.merge(
        outcome_df,
        program_df[merge_cols],
        on="_state_key",
        how="inner",
        validate="m:1",
    )
    # Replace state column with the original display name and drop helper key
    merged[STANDARD_COLUMN_STATE] = outcome_df[STANDARD_COLUMN_STATE]
    merged = merged.drop(columns=["_state_key"])

    keep_cols = [STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM] + list(METRIC_COLUMNS_CANONICAL.keys())
    # Preserve Year column if it exists (for aggregation later)
    if "Year" in merged.columns:
        keep_cols.append("Year")
    # Preserve audiologists per 100k column
    if audiologist_col and audiologist_col in merged.columns:
        keep_cols.append(audiologist_col)
        # Standardize column name
        merged = merged.rename(columns={audiologist_col: "Audiologists_per_100k"})
        keep_cols = [c if c != audiologist_col else "Audiologists_per_100k" for c in keep_cols]
        # Convert to numeric
        merged["Audiologists_per_100k"] = pd.to_numeric(merged["Audiologists_per_100k"], errors='coerce')
    existing = [c for c in keep_cols if c in merged.columns]
    merged = merged[existing].copy()

    for metric in METRIC_COLUMNS_CANONICAL.keys():
        if metric in merged.columns:
            merged[metric] = (
                merged[metric]
                .astype(str)
                .str.replace('%', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
            merged[metric] = pd.to_numeric(merged[metric], errors='coerce')

    if STANDARD_COLUMN_PROGRAM in merged.columns:
        merged[STANDARD_COLUMN_PROGRAM] = coerce_program_presence(merged[STANDARD_COLUMN_PROGRAM])

    merged = merged.dropna(subset=[STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM])
    merged[STANDARD_COLUMN_STATE] = merged[STANDARD_COLUMN_STATE].astype(str).str.strip()
    merged["State_Code"] = merged[STANDARD_COLUMN_STATE].map(STATE_TO_USPS)
    return merged


def filter_cmv(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only include CMV-required states."""
    return df[df[STANDARD_COLUMN_STATE].isin(CMV_STATES)].copy()


def filter_cmv_with_georgia(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to include CMV-required states plus Georgia."""
    cmv_states_with_ga = CMV_STATES | {"Georgia"}
    return df[df[STANDARD_COLUMN_STATE].isin(cmv_states_with_ga)].copy()


def compute_group_stats(df: pd.DataFrame, metric: str) -> dict:
    """Compute statistical tests for CMV states only."""
    if metric not in df.columns:
        return {
            "n_with": 0, "n_without": 0,
            "mean_with": np.nan, "mean_without": np.nan,
            "sd_with": np.nan, "sd_without": np.nan,
            "median_with": np.nan, "median_without": np.nan,
            "t_stat": np.nan, "t_pvalue": np.nan,
            "mw_stat": np.nan, "mw_pvalue": np.nan,
            "cohens_d": np.nan, "reg_coef": np.nan,
            "reg_pvalue": np.nan, "reg_summary": None,
        }

    df_metric = df.dropna(subset=[metric])
    with_prog = df_metric[df_metric[STANDARD_COLUMN_PROGRAM] == True][metric]
    without_prog = df_metric[df_metric[STANDARD_COLUMN_PROGRAM] == False][metric]

    stats_dict = {
        "n_with": int(with_prog.size),
        "n_without": int(without_prog.size),
        "mean_with": float(with_prog.mean()) if with_prog.size > 0 else np.nan,
        "mean_without": float(without_prog.mean()) if without_prog.size > 0 else np.nan,
        "sd_with": float(with_prog.std(ddof=1)) if with_prog.size > 1 else np.nan,
        "sd_without": float(without_prog.std(ddof=1)) if without_prog.size > 1 else np.nan,
        "median_with": float(with_prog.median()) if with_prog.size > 0 else np.nan,
        "median_without": float(without_prog.median()) if without_prog.size > 0 else np.nan,
    }

    t_stat, t_pval = np.nan, np.nan
    if with_prog.size > 0 and without_prog.size > 0:
        try:
            t_res = stats.ttest_ind(with_prog, without_prog, equal_var=False)
            t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)
        except Exception:
            pass

    mw_stat, mw_pval = np.nan, np.nan
    if with_prog.size > 0 and without_prog.size > 0:
        try:
            mw_res = stats.mannwhitneyu(with_prog, without_prog, alternative='two-sided')
            mw_stat, mw_pval = float(mw_res.statistic), float(mw_res.pvalue)
        except ValueError:
            pass

    cohens_d = np.nan
    if with_prog.size > 1 and without_prog.size > 1:
        s1 = with_prog.std(ddof=1)
        s2 = without_prog.std(ddof=1)
        n1 = with_prog.size
        n2 = without_prog.size
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else np.nan
        if sp and sp > 0:
            cohens_d = (with_prog.mean() - without_prog.mean()) / sp

    reg_coef, reg_pvalue, reg_summary = np.nan, np.nan, None
    if HAS_STATSMODELS and df_metric.shape[0] >= 3:
        try:
            X = df_metric[[STANDARD_COLUMN_PROGRAM]].astype(int)
            X = sm.add_constant(X)
            y = df_metric[metric]
            model = sm.OLS(y, X, missing='drop')
            res = model.fit()
            reg_coef = float(res.params.get(STANDARD_COLUMN_PROGRAM, np.nan))
            reg_pvalue = float(res.pvalues.get(STANDARD_COLUMN_PROGRAM, np.nan))
            reg_summary = res.summary().as_text()
        except Exception:
            pass

    stats_dict.update({
        "t_stat": t_stat, "t_pvalue": t_pval,
        "mw_stat": mw_stat, "mw_pvalue": mw_pval,
        "cohens_d": float(cohens_d) if not pd.isna(cohens_d) else np.nan,
        "reg_coef": reg_coef, "reg_pvalue": reg_pvalue, "reg_summary": reg_summary,
    })
    return stats_dict


# ===========================
# Streamlit App
# ===========================
st.set_page_config(page_title="CMV States - 1-3-6 Outcomes", layout="wide")

# Global CSS styling - Nature-inspired theme (same as main app)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Lato:wght@300;400;700&display=swap');
    :root {
      --green-light: #e8f5e9;
      --green-soft: #a8e6cf;
      --green-medium: #56ab2f;
      --green-dark: #2d5016;
      --beige: #f5f1e8;
      --beige-dark: #e8e0d3;
      --off-white: #fafafa;
      --text-dark: #2d5016;
      --text-muted: #5a7c3f;
      --shadow-soft: rgba(86, 171, 47, 0.1);
      --shadow-medium: rgba(86, 171, 47, 0.15);
    }
    
    * {
      font-family: 'Poppins', 'Lato', sans-serif !important;
    }
    
    .stApp {
      background: linear-gradient(135deg, #ffffff 0%, #f8f9f8 100%);
      background-attachment: fixed;
    }
    
    .main .block-container {
      padding-top: 2rem;
      padding-bottom: 2rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
      color: var(--text-dark) !important;
      font-family: 'Poppins', sans-serif !important;
      font-weight: 600;
    }
    
    .page-title {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--green-medium) 0%, var(--green-soft) 100%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      margin-bottom: 0.5rem;
      font-family: 'Poppins', sans-serif;
      letter-spacing: -0.5px;
    }
    
    .subtitle {
      color: var(--text-muted);
      margin-bottom: 1.5rem;
      font-size: 1.1rem;
      font-weight: 300;
      font-family: 'Lato', sans-serif;
    }
    
    .section-header {
      padding: 0.8rem 1.2rem;
      border-left: 5px solid var(--green-medium);
      background: linear-gradient(90deg, rgba(168, 230, 207, 0.2) 0%, rgba(168, 230, 207, 0.05) 100%);
      border-radius: 12px;
      margin: 0.8rem 0 1rem 0;
      font-weight: 600;
      color: var(--text-dark);
      box-shadow: 0 2px 8px var(--shadow-soft);
      font-family: 'Poppins', sans-serif;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      position: relative;
    }
    
    /* Prevent absolute positioning overlap in custom elements only */
    .section-header *,
    .card *,
    .stat-block *,
    .page-title *,
    .subtitle * {
      position: relative !important;
    }
    
    .card {
      background: var(--off-white);
      border-radius: 16px;
      padding: 1.2rem;
      box-shadow: 0 4px 20px var(--shadow-soft);
      border: 1px solid rgba(168, 230, 207, 0.3);
      backdrop-filter: blur(10px);
      display: block;
      position: relative;
    }
    
    .stat-block {
      background: linear-gradient(135deg, var(--beige) 0%, var(--off-white) 100%);
      border-radius: 16px;
      padding: 1.2rem;
      border: 2px dashed rgba(86, 171, 47, 0.3);
      box-shadow: 0 2px 12px var(--shadow-soft);
    }
    
    .note {
      font-size: 0.9rem;
      color: var(--text-muted);
      font-style: italic;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #ffffff 0%, #f8f9f8 100%);
    }
    
    /* Revert Streamlit default arrows/icons - don't override */
    [data-testid="stSidebar"] svg,
    [data-testid="stAppViewContainer"] svg,
    .stExpander svg,
    .stSelectbox svg,
    button svg,
    .stButton svg {
      position: relative !important;
      display: inline-block !important;
    }
    
    /* Button styling */
    .stButton > button {
      background: linear-gradient(135deg, var(--green-soft) 0%, var(--green-medium) 100%);
      color: white;
      border: none;
      border-radius: 12px;
      padding: 0.5rem 1.5rem;
      font-weight: 500;
      font-family: 'Poppins', sans-serif;
      box-shadow: 0 4px 12px var(--shadow-medium);
      transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 16px var(--shadow-medium);
      background: linear-gradient(135deg, var(--green-medium) 0%, var(--green-dark) 100%);
    }
    
    /* Selectbox and input styling */
    .stSelectbox > div > div {
      background: var(--off-white);
      border-radius: 12px;
      border: 2px solid rgba(168, 230, 207, 0.4);
    }
    
    .stSelectbox > div > div:focus-within {
      border-color: var(--green-medium);
      box-shadow: 0 0 0 3px var(--shadow-soft);
    }
    
    .stCheckbox {
      color: var(--text-dark);
    }
    
    /* Info boxes */
    .stInfo {
      background: linear-gradient(135deg, rgba(168, 230, 207, 0.2) 0%, rgba(168, 230, 207, 0.1) 100%);
      border-left: 4px solid var(--green-medium);
      border-radius: 12px;
    }
    
    /* Expander styling */
    .stExpander {
      background: var(--off-white);
      border-radius: 12px;
      border: 1px solid rgba(168, 230, 207, 0.3);
    }
    
    /* Text colors */
    .stMarkdown, p, li {
      color: var(--text-dark);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='page-title'>🌿 CMV-Required States: 1–3–6 Outcomes 🌱</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Connecticut, Florida, Iowa, Kentucky, New York, Pennsylvania, Utah, Virginia</div>", unsafe_allow_html=True)

# Load data
with st.sidebar:
    st.markdown("### 🌿 CMV States Filters")
    try:
        data_df = prepare_data(OUTCOME_FILE, PROGRAM_FILE)
        available_metrics = [m for m in METRIC_COLUMNS_CANONICAL.keys() if m in data_df.columns]
        if not available_metrics:
            available_metrics = list(METRIC_COLUMNS_CANONICAL.keys())
    except Exception as e:
        st.error(f"Data loading error: {e}")
        data_df = pd.DataFrame(columns=[STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM] + list(METRIC_COLUMNS_CANONICAL.keys()))
        available_metrics = [m for m in METRIC_COLUMNS_CANONICAL.keys() if m in data_df.columns]

    selected_metric = st.selectbox("Outcome Metric", options=available_metrics, index=0 if available_metrics else None)

if data_df.empty or not selected_metric:
    st.warning("No data available.")
    st.stop()

cmv_df = filter_cmv(data_df)
cmv_df_viz = filter_cmv_with_georgia(data_df)  # Include Georgia for visualizations

if cmv_df.empty:
    st.warning("No CMV states found in the data.")
    st.stop()

# Visualizations (includes Georgia)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-header'>🍃 Distribution by Program Presence (CMV States + Georgia)</div>", unsafe_allow_html=True)
    tmp = cmv_df_viz.dropna(subset=[selected_metric]).copy()
    if not tmp.empty:
        tmp["Program"] = tmp[STANDARD_COLUMN_PROGRAM].map({True: "With Program", False: "Without Program"})
        fig_box = px.box(
            tmp,
            x="Program",
            y=selected_metric,
            points="all",
            color="Program",
            color_discrete_sequence=px.colors.sequential.Viridis,
            hover_name=STANDARD_COLUMN_STATE,
            hover_data={selected_metric: ":.2f", STANDARD_COLUMN_PROGRAM: True},
        )
        fig_box.update_layout(margin=dict(t=10, b=20, l=10, r=10), yaxis_title=f"{selected_metric} (%)", showlegend=False)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_box, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(f"No data available for {selected_metric}.")

with col2:
    st.markdown("<div class='section-header'>🌍 U.S. Choropleth (CMV States + Georgia Highlighted)</div>", unsafe_allow_html=True)
    tmp = cmv_df_viz.dropna(subset=[selected_metric, "State_Code"]).copy()
    if not tmp.empty:
        # Aggregate by state if multiple years exist (take most recent year)
        if "Year" in tmp.columns:
            tmp = tmp.sort_values("Year", ascending=False).groupby("State_Code").agg({
                selected_metric: 'first',
                STANDARD_COLUMN_STATE: 'first',
            }).reset_index()
        else:
            tmp = tmp.groupby("State_Code", as_index=False).agg({
                selected_metric: 'mean',
                STANDARD_COLUMN_STATE: 'first',
            })
        
        # For consistent coloring, compute global range among CMV states
        color_range = [float(tmp[selected_metric].min()), float(tmp[selected_metric].max())] if not tmp.empty else None
        
        fig_map = px.choropleth(
            tmp,
            locations="State_Code",
            locationmode="USA-states",
            color=selected_metric,
            color_continuous_scale=px.colors.sequential.Viridis[::-1],
            range_color=color_range,
            scope="usa",
            hover_name=STANDARD_COLUMN_STATE,
            title=None,
        )
        fig_map.update_coloraxes(colorbar_title=f"{selected_metric} (%)")
        fig_map.update_layout(margin=dict(t=10, b=20, l=10, r=10))
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_map, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(f"No data available for {selected_metric} choropleth.")

# Scattergram: Selected Outcome vs Audiologists per 100k (CMV States + Georgia)
if "Audiologists_per_100k" in cmv_df_viz.columns and selected_metric in cmv_df_viz.columns:
    st.markdown("<div class='section-header'>📊 Outcome vs Audiologists per 100k Population (CMV States + Georgia)</div>", unsafe_allow_html=True)
    
    tmp = cmv_df_viz.dropna(subset=[selected_metric, "Audiologists_per_100k"]).copy()
    if not tmp.empty:
        # Aggregate by state if multiple years exist (take most recent year)
        if "Year" in tmp.columns:
            tmp = tmp.sort_values("Year", ascending=False).groupby(STANDARD_COLUMN_STATE).agg({
                selected_metric: 'first',
                "Audiologists_per_100k": 'first',
                STANDARD_COLUMN_PROGRAM: 'first',
            }).reset_index()
        else:
            tmp = tmp.groupby(STANDARD_COLUMN_STATE, as_index=False).agg({
                selected_metric: 'mean',
                "Audiologists_per_100k": 'first',
                STANDARD_COLUMN_PROGRAM: 'first',
            })
        
        # Separate Georgia from CMV states for trendline calculation
        tmp_cmv = tmp[tmp[STANDARD_COLUMN_STATE] != "Georgia"].copy()
        tmp_georgia = tmp[tmp[STANDARD_COLUMN_STATE] == "Georgia"].copy()
        
        # Create scatterplot with CMV states only (Georgia will be added separately)
        fig_scatter = px.scatter(
            tmp_cmv,
            x="Audiologists_per_100k",
            y=selected_metric,
            color=tmp_cmv[STANDARD_COLUMN_PROGRAM].map({True: "With Program", False: "Without Program"}),
            color_discrete_sequence=px.colors.sequential.Viridis,
            hover_name=STANDARD_COLUMN_STATE,
            hover_data={STANDARD_COLUMN_PROGRAM: True, selected_metric: ":.2f", "Audiologists_per_100k": ":.1f"},
        )
        
        # Remove the program presence legend items
        fig_scatter.update_traces(showlegend=False, selector=dict(type='scatter', mode='markers'))
        
        # Highlight Georgia if present (excluded from trendline)
        if not tmp_georgia.empty:
            ga_hover = [
                f"{row[STANDARD_COLUMN_STATE]}<br>"
                f"{selected_metric}: {row[selected_metric]:.2f}%<br>"
                f"Audiologists per 100k: {row['Audiologists_per_100k']:.1f}"
                for _, row in tmp_georgia.iterrows()
            ]
            fig_scatter.add_scatter(
                x=tmp_georgia["Audiologists_per_100k"],
                y=tmp_georgia[selected_metric],
                mode='markers',
                name='Georgia (excluded from trendline)',
                marker=dict(size=12, symbol='star', color='orange', line=dict(width=2, color='darkorange')),
                hovertext=ga_hover,
                hoverinfo='text',
                showlegend=True,
            )
        
        # Add single OLS trendline for CMV states only (excluding Georgia)
        if tmp_cmv.shape[0] >= 3:
            x_vals = tmp_cmv["Audiologists_per_100k"].values
            y_vals = tmp_cmv[selected_metric].values
            # Simple linear regression
            x_mean = np.mean(x_vals)
            y_mean = np.mean(y_vals)
            numerator = np.sum((x_vals - x_mean) * (y_vals - y_mean))
            denominator = np.sum((x_vals - x_mean) ** 2)
            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                # Generate trendline points
                x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_trend = intercept + slope * x_trend
                fig_scatter.add_scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name='OLS Trendline (CMV States only)',
                    line=dict(color='rgba(68, 1, 84, 0.6)', width=2, dash='dash'),
                    showlegend=True,
                )
        
        fig_scatter.update_layout(
            margin=dict(t=10, b=20, l=10, r=10),
            xaxis_title="Audiologists per 100k Population",
            yaxis_title=f"{selected_metric} (%)",
        )
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_scatter, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(f"No data available for {selected_metric} vs Audiologists per 100k.")
elif "Audiologists_per_100k" not in cmv_df_viz.columns:
    st.info("Audiologists per 100k data not available.")

# Statistical Results
st.markdown("<div class='section-header'>📈 Statistical Analysis (CMV States Only)</div>", unsafe_allow_html=True)
stats_dict = compute_group_stats(cmv_df, selected_metric)

st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("**With Audiology Program**")
    st.write(f"n = {stats_dict['n_with']}")
    if not pd.isna(stats_dict['mean_with']):
        st.write(f"Mean ± SD: {stats_dict['mean_with']:.2f} ± {stats_dict['sd_with']:.2f}%")
        st.write(f"Median: {stats_dict['median_with']:.2f}%")

with col2:
    st.markdown("**Without Audiology Program**")
    st.write(f"n = {stats_dict['n_without']}")
    if not pd.isna(stats_dict['mean_without']):
        st.write(f"Mean ± SD: {stats_dict['mean_without']:.2f} ± {stats_dict['sd_without']:.2f}%")
        st.write(f"Median: {stats_dict['median_without']:.2f}%")

st.markdown("---")

st.markdown("**Statistical Tests**")
if not pd.isna(stats_dict['t_pvalue']):
    st.write(f"**Welch's t-test**: p = {stats_dict['t_pvalue']:.4f}")
    st.caption("Tests if the means of two groups are significantly different. Used when variances are unequal.")
if not pd.isna(stats_dict['mw_pvalue']):
    st.write(f"**Mann–Whitney U test**: p = {stats_dict['mw_pvalue']:.4f}")
    st.caption("Non-parametric test that compares distributions without assuming normality.")
if not pd.isna(stats_dict['cohens_d']):
    st.write(f"**Cohen's d (effect size)**: {stats_dict['cohens_d']:.3f}")
    st.caption("Measures the magnitude of difference between groups. |d| > 0.2 is small, > 0.5 is medium, > 0.8 is large.")
if not pd.isna(stats_dict['reg_pvalue']):
    st.write(f"**Linear regression coefficient**: {stats_dict['reg_coef']:.3f}, p = {stats_dict['reg_pvalue']:.4f}")
    st.caption("Shows the association between program presence and outcome in a regression model.")

st.markdown("</div>", unsafe_allow_html=True)
