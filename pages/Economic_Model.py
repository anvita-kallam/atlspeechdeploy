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
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPLEMENTATION_FILE = os.path.join(WORKSPACE_DIR, "Economic Model Data - Implementation Data.csv")
CORRELATION_FILE = os.path.join(WORKSPACE_DIR, "Economic Model Data - Special Education vs. AuD Correlation.csv")


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


@st.cache_data
def load_implementation_data():
    """Load and clean implementation data."""
    df = pd.read_csv(IMPLEMENTATION_FILE)
    
    # Clean currency values for each year column
    year_cols = [col for col in df.columns if 'Year' in col]
    for col in year_cols:
        df[col] = df[col].apply(clean_currency)
    
    return df, year_cols


@st.cache_data
def load_correlation_data():
    """Load and clean correlation data."""
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


def create_implementation_cost_chart(df, year_cols):
    """Create stacked bar chart for implementation costs."""
    # Filter to cost-related metrics
    cost_metrics = ['CAPEX (Accreditation Fees, Equipment, Build-out)', 
                    'Total Annual Expenses',
                    'Total OPEX Baseline (Personnel $860,437 + Operating $25,000)']
    
    cost_df = df[df['Metric'].isin(cost_metrics)].copy()
    
    # Prepare data for stacked bar
    years = [col.replace('Year ', '').split(' ')[0] for col in year_cols]
    years = [f"Year {y}" for y in years]
    
    # Extract values for each metric
    opex_values = []
    total_exp_values = []
    capex_values = []
    
    for col in year_cols:
        opex_row = cost_df[cost_df['Metric'] == 'Total OPEX Baseline (Personnel $860,437 + Operating $25,000)']
        total_exp_row = cost_df[cost_df['Metric'] == 'Total Annual Expenses']
        capex_row = cost_df[cost_df['Metric'] == 'CAPEX (Accreditation Fees, Equipment, Build-out)']
        
        opex_values.append(opex_row[col].values[0] if len(opex_row[col].values) > 0 else 0)
        total_exp_values.append(total_exp_row[col].values[0] if len(total_exp_row[col].values) > 0 else 0)
        capex_values.append(capex_row[col].values[0] if len(capex_row[col].values) > 0 else 0)
    
    # Create stacked bar chart
    # Based on the data: Total Annual Expenses = OPEX Baseline + additional expenses (which includes CAPEX in some years)
    # The chart shows: OPEX Baseline (blue, bottom), then the difference to Total Annual Expenses (red, middle), then CAPEX (yellow, top)
    # But actually, looking at the data more carefully, Total Annual Expenses already includes CAPEX in some cases
    # Let's calculate: Additional expenses = Total Annual Expenses - OPEX Baseline
    # Then CAPEX goes on top of Total Annual Expenses
    
    fig = go.Figure()
    
    # Bottom layer: OPEX Baseline (Blue)
    fig.add_trace(go.Bar(
        name='Total OPEX Baseline',
        x=years,
        y=opex_values,
        marker_color='#1f77b4',  # Blue
        hovertemplate='<b>Total OPEX Baseline</b><br>Year: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
    ))
    
    # Middle layer: Additional Expenses beyond OPEX (Red)
    # This is Total Annual Expenses - OPEX Baseline
    additional_heights = [total_exp_values[i] - opex_values[i] for i in range(len(years))]
    fig.add_trace(go.Bar(
        name='Total Annual Expenses',
        x=years,
        y=additional_heights,
        base=opex_values,
        marker_color='#d62728',  # Red
        hovertemplate='<b>Total Annual Expenses</b><br>Year: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
    ))
    
    # Top layer: CAPEX (Yellow/Orange)
    # Base is Total Annual Expenses
    fig.add_trace(go.Bar(
        name='CAPEX',
        x=years,
        y=capex_values,
        base=total_exp_values,
        marker_color='#ffbb78',  # Yellow/Orange
        hovertemplate='<b>CAPEX</b><br>Year: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Total State Implementation Costs of an AuD Program",
        xaxis_title="Year",
        yaxis_title="Cost ($)",
        barmode='stack',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 2000000], tickformat='$,.0f')
    )
    
    return fig


def create_tuition_subsidy_chart(df, year_cols):
    """Create combination chart for tuition revenue vs subsidy/gain."""
    # Get tuition revenue and subsidy data
    tuition_row = df[df['Metric'] == 'Annual Tuition Revenue (Assumes 15 students/cohort, 66% In-State)']
    subsidy_row = df[df['Metric'] == 'Annual Subsidy Required / (Operational Surplus)']
    
    years = [col.replace('Year ', '').split(' ')[0] for col in year_cols]
    years = [f"Year {y}" for y in years]
    
    tuition_values = [tuition_row[col].values[0] if len(tuition_row[col].values) > 0 else 0 for col in year_cols]
    subsidy_values = [subsidy_row[col].values[0] if len(subsidy_row[col].values) > 0 else 0 for col in year_cols]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for tuition revenue
    fig.add_trace(
        go.Bar(
            name="Annual Tuition Revenue",
            x=years,
            y=tuition_values,
            marker_color='#aec7e8',
            hovertemplate='<b>Annual Tuition Revenue</b><br>Year: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Add line chart for subsidy/gain
    # Determine marker colors: red for negative (subsidy), green for positive (gain)
    marker_colors = ['#d62728' if v < 0 else '#2ca02c' for v in subsidy_values]
    
    fig.add_trace(
        go.Scatter(
            name="Subsidy/Gain Trend",
            x=years,
            y=subsidy_values,
            mode='lines+markers',
            line=dict(color='black', dash='dash', width=2),
            marker=dict(size=10, color=marker_colors),
            hovertemplate='<b>Subsidy/Gain</b><br>Year: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=True,
    )
    
    # Add breakeven line at $0
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="red",
        line_width=2,
        annotation_text="Breakeven Point ($0)",
        annotation_position="right",
        secondary_y=True
    )
    
    # Set axis labels
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Annual Tuition Revenue ($)", secondary_y=False, tickformat='$,.0f', range=[0, 1000000])
    fig.update_yaxes(title_text="Annual Subsidy Required / (Gain) ($)", secondary_y=True, tickformat='$,.0f', range=[-700000, 100000])
    
    fig.update_layout(
        title="Annual Tuition Revenue vs. Subsidy/Gain Over 5 Years",
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
    
    # Create scatter plot
    fig = px.scatter(
        df_clean,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: x_col, y_col: y_col},
        hover_data=['State'] if 'State' in df_clean.columns else None
    )
    
    # Add regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=2, dash='dash')
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

# Implementation Costs Chart
st.markdown("<div class='section-header'>Total State Implementation Costs of an AuD Program</div>", unsafe_allow_html=True)
fig_impl = create_implementation_cost_chart(impl_df, year_cols)
st.plotly_chart(fig_impl, use_container_width=True)

# Tuition Revenue vs Subsidy/Gain Chart
st.markdown("<div class='section-header'>Annual Tuition Revenue vs. Subsidy/Gain Over 5 Years</div>", unsafe_allow_html=True)
fig_tuition = create_tuition_subsidy_chart(impl_df, year_cols)
st.plotly_chart(fig_tuition, use_container_width=True)

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
            st.markdown("<div class='stat-block'>", unsafe_allow_html=True)
            st.write(f"**Statistical Results:**")
            st.write(f"- R-squared: {r2_medicaid:.3f}")
            st.write(f"- Coefficient for Total AuD Program Enrollment: {coef_medicaid:.4f}")
            st.write(f"- P-value for Total AuD Program Enrollment: {p_medicaid:.3f}")
            st.write(f"- Sample size: {n_medicaid}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("**Key Takeaways:**")
            st.write(f"- The R-squared value of {r2_medicaid:.3f} indicates that {r2_medicaid*100:.1f}% of the variance in Medicaid Spending per Child can be explained by Total AuD Program Enrollment.")
            if r2_medicaid < 0.1:
                st.write("- This suggests a very weak linear relationship between the two variables.")
            if p_medicaid > 0.05:
                st.write(f"- The p-value of {p_medicaid:.3f} indicates the relationship is not statistically significant.")
            st.markdown("</div>", unsafe_allow_html=True)

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
            st.markdown("<div class='stat-block'>", unsafe_allow_html=True)
            st.write(f"**Statistical Results:**")
            st.write(f"- R-squared: {r2_sped:.3f}")
            st.write(f"- Coefficient for Total AuD Program Enrollment: {coef_sped:.4f}")
            st.write(f"- P-value for Total AuD Program Enrollment: {p_sped:.3f}")
            st.write(f"- Sample size: {n_sped}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("**Key Takeaways:**")
            st.write(f"- The R-squared value of {r2_sped:.3f} indicates that {r2_sped*100:.1f}% of the variance in Special Education Spending per Child can be explained by Total AuD Program Enrollment.")
            if r2_sped < 0.1:
                st.write("- This suggests an extremely weak linear relationship between the two variables.")
            if p_sped > 0.05:
                st.write(f"- The p-value of {p_sped:.3f} indicates no statistically significant relationship.")
            st.markdown("</div>", unsafe_allow_html=True)

