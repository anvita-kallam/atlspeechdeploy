import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
IMPLEMENTATION_FILE = os.path.join(WORKSPACE_DIR, "Economic Model Data - Implementation Data.csv")
CORRELATION_FILE = os.path.join(WORKSPACE_DIR, "Economic Model Data - Special Education vs. AuD Correlation.csv")
LONGTERM_FILE = os.path.join(WORKSPACE_DIR, "Economic Model Data - Long-Term Economic Data.csv")


def clean_currency(value):
    """Convert currency string to float."""
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    # Remove $, commas, and parentheses (for negative values)
    cleaned = str(value).replace('$', '').replace(',', '').replace('(', '-').replace(')', '').strip()
    try:
        return float(cleaned)
    except:
        return np.nan


def clean_percentage(value):
    """Convert percentage string to float."""
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    # Remove % sign
    cleaned = str(value).replace('%', '').strip()
    try:
        return float(cleaned)
    except:
        return np.nan


@st.cache_data
def load_implementation_data():
    """Load and clean implementation data."""
    if not os.path.exists(IMPLEMENTATION_FILE):
        raise FileNotFoundError(f"CSV not found: {IMPLEMENTATION_FILE}")
    df = pd.read_csv(IMPLEMENTATION_FILE)
    
    # Clean currency values for each year column
    year_cols = [col for col in df.columns if 'Year' in col]
    for col in year_cols:
        df[col] = df[col].apply(clean_currency)
    
    return df, year_cols


@st.cache_data
def load_correlation_data():
    """Load and clean correlation data."""
    if not os.path.exists(CORRELATION_FILE):
        raise FileNotFoundError(f"CSV not found: {CORRELATION_FILE}")
    df = pd.read_csv(CORRELATION_FILE)
    
    # Clean currency columns
    if 'Sp Ed Spending per Child' in df.columns:
        df['Sp Ed Spending per Child'] = df['Sp Ed Spending per Child'].apply(clean_currency)
    if 'Medicaid Spending per Child' in df.columns:
        df['Medicaid Spending per Child'] = df['Medicaid Spending per Child'].apply(clean_currency)
    
    # Ensure numeric columns are numeric
    if 'Total AuD Program Enrollment' in df.columns:
        df['Total AuD Program Enrollment'] = pd.to_numeric(df['Total AuD Program Enrollment'], errors='coerce')
    
    # Remove rows with missing data for analysis
    df_clean = df.dropna(subset=['Total AuD Program Enrollment', 'Sp Ed Spending per Child', 'Medicaid Spending per Child'])
    
    return df, df_clean


@st.cache_data
def load_longterm_data():
    """Load and clean long-term economic data."""
    if not os.path.exists(LONGTERM_FILE):
        raise FileNotFoundError(f"CSV not found: {LONGTERM_FILE}")
    df = pd.read_csv(LONGTERM_FILE)
    
    # Remove the last row if it's a summary row (contains methodology text)
    if 'How I calculated it' in df.columns:
        df = df[~df['How I calculated it'].astype(str).str.contains('Methodology|Sources|ASHA|Academy', case=False, na=False)]
    
    # Clean currency columns
    if 'Medicaid Spending per Child' in df.columns:
        df['Medicaid Spending per Child'] = df['Medicaid Spending per Child'].apply(clean_currency)
    if 'Avg. special education spending (per child per state)' in df.columns:
        df['Avg. special education spending (per child per state)'] = df['Avg. special education spending (per child per state)'].apply(clean_currency)
    if 'State Revenue impacts (if available) -  Total medicaid spending (in millions)' in df.columns:
        df['State Revenue impacts (if available) -  Total medicaid spending (in millions)'] = df['State Revenue impacts (if available) -  Total medicaid spending (in millions)'].apply(clean_currency)
    
    # Clean percentage columns
    if 'Unemployment Rate (general)' in df.columns:
        df['Unemployment Rate (general)'] = pd.to_numeric(df['Unemployment Rate (general)'], errors='coerce')
    if 'Unemployment Rate (disabled)' in df.columns:
        df['Unemployment Rate (disabled)'] = df['Unemployment Rate (disabled)'].apply(clean_percentage)
    
    # Ensure numeric columns are numeric
    numeric_cols = ['# of AuD Programs', 'Programs', 'Total Capacity (4 yr)', 'Total Enrollment (4 yr)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows where State is missing
    df = df[df['State'].notna()]
    
    return df


def create_implementation_cost_chart():
    """Create stacked bar chart for implementation costs showing CAPEX and OPEX Baseline using image values."""
    # Use values from the image
    years = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    capex_values = [438250, 15000, 0, 5082, 0]  # From image
    opex_values = [459600, 459600, 459600, 549600, 549600]  # From image
    
    # Create stacked bar chart: OPEX Baseline (dark green, bottom), CAPEX (light green, top) - shades of green
    fig = go.Figure()
    
    # Bottom layer: OPEX Baseline (Dark Green)
    fig.add_trace(go.Bar(
        name='Total OPEX Baseline',
        x=years,
        y=opex_values,
        marker_color='#408830',  # Dark green
        hovertemplate='<b>Total OPEX Baseline</b><br>Year: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
    ))
    
    # Top layer: CAPEX (Light Green)
    fig.add_trace(go.Bar(
        name='CAPEX',
        x=years,
        y=capex_values,
        base=opex_values,
        marker_color='#b1d28a',  # Light green
        hovertemplate='<b>CAPEX</b><br>Year: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Total State Implementation Costs of an AuD Program",
        xaxis_title="Year",
        yaxis_title="Cost ($)",
        barmode='stack',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 900000], tickformat='$,.0f')
    )
    
    return fig


def create_tuition_netprofit_chart():
    """Create combination chart for tuition revenue vs net profit using image values."""
    # Use values from the image
    years = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    tuition_values = [238150, 476300, 714450, 952600, 952600]  # From image
    net_profit_values = [-659700, -658000, -403150, -5232, 397768]  # From image
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for tuition revenue (green - Viridis)
    fig.add_trace(
        go.Bar(
            name="Tuition Revenue",
            x=years,
            y=tuition_values,
            marker_color='#5ec962',  # Green - Viridis
            hovertemplate='<b>Tuition Revenue</b><br>Year: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Add line chart for net profit (teal - Viridis)
    fig.add_trace(
        go.Scatter(
            name="Net Profit",
            x=years,
            y=net_profit_values,
            mode='lines+markers',
            line=dict(color='#21918c', width=2),  # Teal - Viridis
            marker=dict(size=10, color='#21918c', symbol='square'),
            hovertemplate='<b>Net Profit</b><br>Year: %{x}<br>Profit: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=True,
    )
    
    # Add breakeven line at $0 (yellow - Viridis)
    fig.add_trace(
        go.Scatter(
            name="Break-even",
            x=years,
            y=[0] * len(years),
            mode='lines',
            line=dict(color='#fde725', width=2, dash='solid'),  # Yellow - Viridis
            hovertemplate='<b>Break-even</b><br>Year: %{x}<br>Amount: $0<extra></extra>',
            showlegend=True
        ),
        secondary_y=True,
    )
    
    # Set axis labels
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Annual Tuition Revenue ($)", secondary_y=False, tickformat='$,.0f', range=[0, 1200000])
    fig.update_yaxes(title_text="Net Profit ($)", secondary_y=True, tickformat='$,.0f', range=[-800000, 600000])
    
    fig.update_layout(
        title="Annual Tuition Revenue vs. Net Profit",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    return fig


def create_correlation_scatter(df, x_col, y_col, title):
    """Create scatter plot with regression line and statistics."""
    # Remove missing values
    df_clean = df.dropna(subset=[x_col, y_col])
    
    if len(df_clean) < 2:
        return None, None, None, None, None
    
    x = df_clean[x_col].values
    y = df_clean[y_col].values
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    # Create scatter plot with new color palette
    fig = px.scatter(
        df_clean,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: x_col, y_col: y_col},
        hover_data=['State'] if 'State' in df_clean.columns else None,
        color_discrete_sequence=['#5ec962']  # Green from palette
    )
    
    # Update marker color
    fig.update_traces(marker=dict(color='#5ec962', size=8, opacity=0.7))
    
    # Add regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='#3b528b', width=2, dash='dash'),  # Blue from palette
            hovertemplate='<b>Regression Line</b><br>%{x:.1f}<br>%{y:,.0f}<extra></extra>'
        )
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig, r_squared, slope, p_value, len(df_clean)


# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Economic Model", layout="wide")

# Apply styling (same as other pages)
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
    
    /* Font family applied selectively to avoid interfering with Streamlit widgets */
    body, .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
      font-family: 'Poppins', 'Lato', sans-serif;
    }
    
    /* Ensure Streamlit widgets use their default fonts and rendering */
    .stSelectbox, .stCheckbox, .stButton, button, input, select {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    
    /* Prevent CSS from interfering with Streamlit widget key rendering */
    [data-testid*="stWidget"], [data-baseweb], .stWidget {
      font-family: inherit !important;
    }
    
    /* Hide any text content that looks like widget keys being displayed */
    /* Using a more specific approach: target Streamlit's internal widget containers */
    [data-testid*="baseButton"]::before,
    [data-testid*="baseButton"]::after,
    [data-testid*="stSelectbox"]::before,
    [data-testid*="stSelectbox"]::after,
    [data-testid*="stCheckbox"]::before,
    [data-testid*="stCheckbox"]::after {
      content: none !important;
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
      position: relative;
      display: block;
    }
    
    /* Prevent absolute positioning overlap in custom elements only - but don't affect Streamlit internals */
    .section-header,
    .card,
    .stat-block,
    .page-title,
    .subtitle {
      position: relative;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='page-title'>Economic Model Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Implementation Costs, Revenue, and Correlation Analysis</div>", unsafe_allow_html=True)

# Load data
try:
    impl_df, year_cols = load_implementation_data()
    corr_df, corr_df_clean = load_correlation_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Financial Metrics Tables (Top Section)
st.markdown("<div class='section-header'>Program Financial Projections</div>", unsafe_allow_html=True)

# Use hardcoded values from the image
year_labels = ["Year 1 (Start Up)", "Year 2 (Candidacy)", "Year 3 (Scaling)", "Year 4 (Full Enrollment)", "Year 5 (Accrediation)"]
years = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]

# Values from the image
capex_values = [438250, 15000, 0, 5082, 0]
opex_values = [459600, 459600, 459600, 549600, 549600]
total_exp_values = [897850, 474600, 459600, 554682, 549600]
tuition_values = [238150, 476300, 714450, 952600, 952600]
subsidy_values = [-659700, 1700, 254850, 397918, 403000]
net_profit_values = [-659700, -658000, -403150, -5232, 397768]

# Create Financial Metrics Table (Left) - as dropdown
col1, col2 = st.columns(2)

with col1:
    with st.expander("Financial Metrics over 5 Years", expanded=False):
        financial_metrics_data = {
            'Metric': [
                'CAPEX (Accreditation Fees)',
                'Total OPEX Baseline (Persc)',
                'Total Annual Expenses',
                'Annual Tuition Revenue (A)',
                'Annual Subsidy Required / (Operational Surplus)'
            ],
            year_labels[0]: [capex_values[0], opex_values[0], total_exp_values[0], tuition_values[0], subsidy_values[0]],
            year_labels[1]: [capex_values[1], opex_values[1], total_exp_values[1], tuition_values[1], subsidy_values[1]],
            year_labels[2]: [capex_values[2], opex_values[2], total_exp_values[2], tuition_values[2], subsidy_values[2]],
            year_labels[3]: [capex_values[3], opex_values[3], total_exp_values[3], tuition_values[3], subsidy_values[3]],
            year_labels[4]: [capex_values[4], opex_values[4], total_exp_values[4], tuition_values[4], subsidy_values[4]]
        }
        
        financial_metrics_df = pd.DataFrame(financial_metrics_data)
        
        # Format the table with colors for subsidy row
        display_financial = financial_metrics_df.copy()
        for i, col in enumerate(year_labels):
            # Format subsidy row (last row) with colors
            subsidy_val = display_financial.iloc[4, i+1]
            if pd.notna(subsidy_val):
                try:
                    num_val = float(subsidy_val)
                    if num_val < 0:
                        display_financial.iloc[4, i+1] = f'<span style="color: #d62728; font-weight: bold;">${num_val:,.0f}</span>'
                    elif num_val > 0:
                        display_financial.iloc[4, i+1] = f'<span style="color: #2ca02c; font-weight: bold;">${num_val:,.0f}</span>'
                    else:
                        display_financial.iloc[4, i+1] = f'${num_val:,.0f}'
                except:
                    pass
            # Format other rows
            for row_idx in range(4):
                val = display_financial.iloc[row_idx, i+1]
                if pd.notna(val) and isinstance(val, (int, float)):
                    display_financial.iloc[row_idx, i+1] = f'${val:,.0f}'
        
        st.markdown(display_financial.to_html(escape=False, index=False, classes='dataframe'), unsafe_allow_html=True)

with col2:
    with st.expander("Summary of Tuition Revenue, Net Profit, and Break-even", expanded=False):
        summary_data = {
            'Year': years,
            'Tuition Revenue': tuition_values,
            'Net Profit': net_profit_values,
            'Break-even': [0, 0, 0, 0, 0]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Format the summary table
        display_summary = summary_df.copy()
        for i in range(len(display_summary)):
            display_summary.iloc[i, 1] = f'${display_summary.iloc[i, 1]:,.0f}'
            display_summary.iloc[i, 2] = f'${display_summary.iloc[i, 2]:,.0f}'
            display_summary.iloc[i, 3] = f'{display_summary.iloc[i, 3]:.0f}'
        
        st.markdown(display_summary.to_html(escape=False, index=False, classes='dataframe'), unsafe_allow_html=True)

# Charts (Bottom Section)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-header'>Total State Implementation Costs of an AuD Program</div>", unsafe_allow_html=True)
    fig_impl = create_implementation_cost_chart()
    st.plotly_chart(fig_impl, use_container_width=True)

with col2:
    st.markdown("<div class='section-header'>Annual Tuition Revenue vs. Net Profit</div>", unsafe_allow_html=True)
    fig_tuition = create_tuition_netprofit_chart()
    st.plotly_chart(fig_tuition, use_container_width=True)

# Display implementation data table with color formatting
with st.expander("View Implementation Data Table", expanded=False):
    # Create a display version with formatted values
    display_df = impl_df.copy()
    
    # Find the subsidy row
    subsidy_mask = display_df['Metric'] == 'Annual Subsidy Required / (Operational Surplus)'
    if subsidy_mask.any():
        # Get year columns
        year_cols_display = [col for col in display_df.columns if 'Year' in col]
        
        # Format the subsidy values with colors
        for col in year_cols_display:
            if col in display_df.columns:
                val = display_df.loc[subsidy_mask, col].values[0]
                if pd.notna(val):
                    # Values are already cleaned to float in impl_df
                    try:
                        num_val = float(val)
                        # Format with HTML colors
                        if num_val < 0:
                            display_df.loc[subsidy_mask, col] = f'<span style="color: #d62728; font-weight: bold;">${num_val:,.0f}</span>'
                        elif num_val > 0:
                            display_df.loc[subsidy_mask, col] = f'<span style="color: #2ca02c; font-weight: bold;">${num_val:,.0f}</span>'
                        else:
                            display_df.loc[subsidy_mask, col] = f'${num_val:,.0f}'
                    except:
                        pass
        
        # Format other numeric columns for display
        for col in year_cols_display:
            if col in display_df.columns:
                for idx in display_df.index:
                    if not subsidy_mask.loc[idx]:  # Only format non-subsidy rows
                        val = display_df.loc[idx, col]
                        if pd.notna(val) and isinstance(val, (int, float)):
                            display_df.loc[idx, col] = f'${val:,.0f}'
        
        # Display with HTML rendering
        st.markdown(display_df.to_html(escape=False, index=False, classes='dataframe'), unsafe_allow_html=True)

# Correlation Analysis
st.markdown("<div class='section-header'>Spending per Child vs. Total AuD Program Enrollment</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Medicaid Spending per Child vs. Total AuD Program Enrollment")
    if 'Medicaid Spending per Child' in corr_df_clean.columns and 'Total AuD Program Enrollment' in corr_df_clean.columns:
        fig_medicaid, r2_medicaid, coef_medicaid, p_medicaid, n_medicaid = create_correlation_scatter(
            corr_df_clean,
            'Total AuD Program Enrollment',
            'Medicaid Spending per Child',
            'Medicaid Spending per Child vs. Total AuD Program Enrollment'
        )
        if fig_medicaid:
            st.plotly_chart(fig_medicaid, use_container_width=True)
            st.write(f"**Statistical Results:**")
            st.write(f"- R-squared: {r2_medicaid:.3f}")
            st.write(f"- Coefficient for Total AuD Program Enrollment: {coef_medicaid:.4f}")
            st.write(f"- P-value for Total AuD Program Enrollment: {p_medicaid:.3f}")
            st.write(f"- Sample size: {n_medicaid}")
            
            st.write("**Key Takeaways:**")
            st.write(f"- The R-squared value of {r2_medicaid:.3f} indicates that {r2_medicaid*100:.1f}% of the variance in Medicaid Spending per Child can be explained by Total AuD Program Enrollment.")
            if r2_medicaid < 0.1:
                st.write("- This suggests a very weak linear relationship between the two variables.")
            if p_medicaid > 0.05:
                st.write(f"- The p-value of {p_medicaid:.3f} indicates the relationship is not statistically significant.")

with col2:
    st.markdown("#### Special Education Spending per Child vs. Total AuD Program Enrollment")
    if 'Sp Ed Spending per Child' in corr_df_clean.columns and 'Total AuD Program Enrollment' in corr_df_clean.columns:
        fig_sped, r2_sped, coef_sped, p_sped, n_sped = create_correlation_scatter(
            corr_df_clean,
            'Total AuD Program Enrollment',
            'Sp Ed Spending per Child',
            'Special Education Spending per Child vs. Total AuD Program Enrollment'
        )
        if fig_sped:
            st.plotly_chart(fig_sped, use_container_width=True)
            st.write(f"**Statistical Results:**")
            st.write(f"- R-squared: {r2_sped:.3f}")
            st.write(f"- Coefficient for Total AuD Program Enrollment: {coef_sped:.4f}")
            st.write(f"- P-value for Total AuD Program Enrollment: {p_sped:.3f}")
            st.write(f"- Sample size: {n_sped}")
            
            st.write("**Key Takeaways:**")
            st.write(f"- The R-squared value of {r2_sped:.3f} indicates that {r2_sped*100:.1f}% of the variance in Special Education Spending per Child can be explained by Total AuD Program Enrollment.")
            if r2_sped < 0.1:
                st.write("- This suggests an extremely weak linear relationship between the two variables.")
            if p_sped > 0.05:
                st.write(f"- The p-value of {p_sped:.3f} indicates no statistically significant relationship.")

# Display correlation data table
st.markdown("<div class='section-header'>Correlation Data</div>", unsafe_allow_html=True)
with st.expander("View Correlation Data Table", expanded=False):
    st.dataframe(corr_df, use_container_width=True)

# Medicaid Spending vs 3 Month Benchmark
st.markdown("<div class='section-header'>Medicaid Spending per Child vs. 3 Month Benchmark</div>", unsafe_allow_html=True)

# Prepare data - paired row by row as provided
medicaid_spending = [
    3419, 3370, 5332, 2884, 3727, 4020, 3839, 4005, 3510, 3329, 4695, 2680, 3387, 3943, 3999, 3898, 4118, 3261, 4682, 3589,
    2087, 3343, 3763, 2970, 3431, 4593, 4182, 3777, 3741, 4320, 5199, 3716, 3829, 3822, 3518, 3020, 5911, 5242, 5145, 2747,
    3394, 4144, 4471, 2586, 4843, 4930, 4003, 1690, 3904, 2888, 4232, 5843, 3821
]

meeting_3 = [
    None, 16.1, 22.7, 18.3, 36.8, 32.0, 65.8, 6.1, 13.6, 34.6, 19.6, 66.3, 62.7, 56.4, 62.6, 49.7, 48.6, 56.2, 66.1, 73.8,
    28.4, 69.1, 32.1, 40.1, 32.9, 52.8, 25.1, 4.3, 22.7, 42.1, 35.6, 39.9, 15.6, 30.9, 23.9, 65.2, 5.5, 68.7, 32.9, 55.3,
    16.0, 10.9, 24.3, 22.8, 79.4, 82.8, 48.8, 26.5, 50.6, 35.8, 73.7
]

# Pair them up - take the minimum length to ensure proper pairing
min_len = min(len(medicaid_spending), len(meeting_3))

# Create DataFrame and remove null values
medicaid_benchmark_df = pd.DataFrame({
    'Medicaid Spending per Child': medicaid_spending[:min_len],
    '3 month benchmark': meeting_3[:min_len]
})

# Remove rows with null values
medicaid_benchmark_df = medicaid_benchmark_df.dropna()

if not medicaid_benchmark_df.empty:
    x = medicaid_benchmark_df['3 month benchmark'].values
    y = medicaid_benchmark_df['Medicaid Spending per Child'].values
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    # Create scatter plot
    fig_medicaid_benchmark = px.scatter(
        medicaid_benchmark_df,
        x='3 month benchmark',
        y='Medicaid Spending per Child',
        title='Medicaid Spending per Child vs. 3 month benchmark',
        labels={'3 month benchmark': '3 month benchmark', 'Medicaid Spending per Child': 'Medicaid Spending per Child ($)'},
        color_discrete_sequence=['#5ec962']  # Green from palette
    )
    
    # Update marker color
    fig_medicaid_benchmark.update_traces(marker=dict(color='#5ec962', size=8, opacity=0.7))
    
    # Add regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    fig_medicaid_benchmark.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name=f'Trendline for Medicaid Spending per Child R² = {r_squared:.3f}',
            line=dict(color='#3b528b', width=2),  # Blue from palette
            hovertemplate='<b>Trendline</b><br>3 month benchmark: %{x:.1f}<br>Medicaid Spending: $%{y:,.0f}<extra></extra>'
        )
    )
    
    fig_medicaid_benchmark.update_layout(
        height=500,
        showlegend=True,
        xaxis_title='3 month benchmark',
        yaxis_title='Medicaid Spending per Child ($)',
        yaxis=dict(tickformat='$,.0f')
    )
    
    st.plotly_chart(fig_medicaid_benchmark, use_container_width=True)
    
    st.write(f"**Statistical Results:**")
    st.write(f"- R-squared: {r_squared:.3f}")
    st.write(f"- Coefficient for 3 month benchmark: {slope:.4f}")
    st.write(f"- P-value: {p_value:.3f}")
    st.write(f"- Sample size: {len(medicaid_benchmark_df)}")
    
    st.write("**Key Takeaways:**")
    st.write(f"- The R-squared value of {r_squared:.3f} indicates that {r_squared*100:.1f}% of the variance in Medicaid Spending per Child can be explained by the 3 month benchmark.")
    if r_squared < 0.1:
        st.write("- This suggests a very weak linear relationship between the two variables.")
    if p_value > 0.05:
        st.write(f"- The p-value of {p_value:.3f} indicates the relationship is not statistically significant.")

# Long-Term Economic Data Section
st.markdown("<div class='section-header'>Long-Term Economic Analysis</div>", unsafe_allow_html=True)

try:
    longterm_df = load_longterm_data()
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### AuD Programs vs. Medicaid Spending per Child")
        if '# of AuD Programs' in longterm_df.columns and 'Medicaid Spending per Child' in longterm_df.columns:
            df_clean = longterm_df.dropna(subset=['# of AuD Programs', 'Medicaid Spending per Child'])
            if not df_clean.empty:
                fig = px.scatter(
                    df_clean,
                    x='# of AuD Programs',
                    y='Medicaid Spending per Child',
                    hover_data=['State'],
                    title='AuD Programs vs. Medicaid Spending per Child',
                    labels={'# of AuD Programs': '# of AuD Programs', 'Medicaid Spending per Child': 'Medicaid Spending per Child ($)'}
                )
                # Use teal from palette for scatter plots
                fig.update_traces(marker=dict(color='#21918c', size=8, opacity=0.7))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Total Enrollment vs. Unemployment Rate (Disabled)")
        if 'Total Enrollment (4 yr)' in longterm_df.columns and 'Unemployment Rate (disabled)' in longterm_df.columns:
            df_clean = longterm_df.dropna(subset=['Total Enrollment (4 yr)', 'Unemployment Rate (disabled)'])
            if not df_clean.empty:
                fig = px.scatter(
                    df_clean,
                    x='Total Enrollment (4 yr)',
                    y='Unemployment Rate (disabled)',
                    hover_data=['State'],
                    title='Total Enrollment vs. Unemployment Rate (Disabled)',
                    labels={'Total Enrollment (4 yr)': 'Total Enrollment (4 yr)', 'Unemployment Rate (disabled)': 'Unemployment Rate (Disabled) (%)'}
                )
                # Use teal from palette for scatter plots
                fig.update_traces(marker=dict(color='#21918c', size=8, opacity=0.7))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart: Top states by enrollment
    st.markdown("#### Top 15 States by Total Enrollment (4 yr)")
    if 'Total Enrollment (4 yr)' in longterm_df.columns:
        df_enrollment = longterm_df.dropna(subset=['Total Enrollment (4 yr)', 'State'])
        df_enrollment = df_enrollment.sort_values('Total Enrollment (4 yr)', ascending=False).head(15)
        if not df_enrollment.empty:
            fig = px.bar(
                df_enrollment,
                x='State',
                y='Total Enrollment (4 yr)',
                title='Top 15 States by Total Enrollment (4 yr)',
                labels={'Total Enrollment (4 yr)': 'Total Enrollment (4 yr)'}
            )
            # Use green from palette for enrollment chart
            fig.update_traces(marker_color='#5ec962')
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart: States by number of programs
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### States by Number of AuD Programs")
        if '# of AuD Programs' in longterm_df.columns:
            df_programs = longterm_df.dropna(subset=['# of AuD Programs', 'State'])
            df_programs = df_programs[df_programs['# of AuD Programs'] > 0].sort_values('# of AuD Programs', ascending=False)
            if not df_programs.empty:
                fig = px.bar(
                    df_programs,
                    x='State',
                    y='# of AuD Programs',
                    title='States by Number of AuD Programs',
                    labels={'# of AuD Programs': '# of AuD Programs'}
                )
                # Use blue from palette for bar charts
                fig.update_traces(marker_color='#3b528b')
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.markdown("#### Medicaid Spending per Child by State (Top 15)")
        if 'Medicaid Spending per Child' in longterm_df.columns:
            df_medicaid = longterm_df.dropna(subset=['Medicaid Spending per Child', 'State'])
            df_medicaid = df_medicaid.sort_values('Medicaid Spending per Child', ascending=False).head(15)
            if not df_medicaid.empty:
                fig = px.bar(
                    df_medicaid,
                    x='State',
                    y='Medicaid Spending per Child',
                    title='Medicaid Spending per Child by State (Top 15)',
                    labels={'Medicaid Spending per Child': 'Medicaid Spending per Child ($)'}
                )
                # Use blue from palette for bar charts
                fig.update_traces(marker_color='#3b528b')
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    # Full data table dropdown
    st.markdown("#### Full Long-Term Economic Data Table")
    with st.expander("View Full Data Table", expanded=False):
        st.dataframe(longterm_df, use_container_width=True)
        
except Exception as e:
    st.error(f"Error loading long-term economic data: {e}")

