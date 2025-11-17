import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------
# Constants and configuration
# ---------------------------
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(WORKSPACE_DIR)  # go up from pages/

# ---------------------------
# Data Setup
# ---------------------------
def create_dcf_data():
    """Create DCF model data."""
    years = [1, 2, 3, 4, 5]
    students = [15, 30, 45, 60, 60]
    tuition_per_student = 15876.67
    revenue = [s * tuition_per_student for s in students]
    upfront_cost = 438250.00
    
    # Facility
    sq_ft = 10000
    maintenance_rate = 6.96
    maintenance_cost = sq_ft * maintenance_rate
    
    # Wages
    director_wage = 120000.00
    faculty_wage = 90000.00
    fte_wages = [270000.00, 270000.00, 270000.00, 360000.00, 360000.00]
    total_wages = [director_wage + faculty_wage + fte for fte in fte_wages]
    
    # State Revenue Impact (Bear, Base, Bull)
    bear_numeric = [10.36337, 20.72674, 31.09011, 41.45348, 41.45348]
    bear_dollar = [None, None, None, None, None]  # Not provided in image
    
    base_numeric = [45.33974375, 90.6794875, 136.01923125, 181.358975, 181.358975]
    base_dollar = [31919.18, 63838.36, 95757.54, 127676.72, 127676.72]
    
    bull_numeric = [129.542125, 259.08425, 388.626375, 518.1685, 518.1685]
    bull_dollar = [182395.31, 364790.62, 547185.93, 729581.25, 729581.25]
    
    # Discount rate
    discount_rate = 0.03694
    
    # Create main data DataFrame
    df = pd.DataFrame({
        'Year': years,
        'Students': students,
        'Tuition per Student': [tuition_per_student] * 5,
        'Revenue': revenue,
        'Director Wage': [director_wage] * 5,
        'Faculty Wage': [faculty_wage] * 5,
        'FTE Wages': fte_wages,
        'Total Wages': total_wages,
        'Maintenance Cost': [maintenance_cost] * 5,
        'Bear (Numeric)': bear_numeric,
        'Base (Numeric)': base_numeric,
        'Base (Dollar)': base_dollar,
        'Bull (Numeric)': bull_numeric,
        'Bull (Dollar)': bull_dollar
    })
    
    # Calculate total costs per year
    df['Total Costs'] = df['Total Wages'] + df['Maintenance Cost']
    
    # Calculate PV and NPV for each scenario
    # Bear: base revenue only (no state impact)
    # Base: base revenue + base state impact
    # Bull: base revenue + bull state impact
    scenarios = {
        'Bear': {
            'revenue_multiplier': revenue,  # Just base tuition revenue
            'npv': 277016.04,
            'pv_revenue': [229666.13, 231234.56, 232803.00, 234371.43, 794588.80],  # From image
            'pv_cost': [443227.19, 427018.52, 411809.85, 396601.18, 458435.86]  # From image
        },
        'Base': {
            'revenue_multiplier': [r + base_dollar[i] for i, r in enumerate(revenue)],
            'npv': 669984.20,
            'pv_revenue': [260448.22, 262016.66, 263585.09, 265153.53, 901087.32],  # From image
            'pv_cost': [443227.19, 427018.52, 411809.85, 396601.18, 458435.86]  # From image
        },
        'Bull': {
            'revenue_multiplier': [r + bull_dollar[i] for i, r in enumerate(revenue)],
            'npv': 2522548.40,
            'pv_revenue': [405563.79, 407132.22, 408700.66, 410269.09, 1403151.77],  # From image
            'pv_cost': [443227.19, 427018.52, 411809.85, 396601.18, 458435.86]  # From image
        }
    }
    
    # Use PV values from image (or calculate if not provided)
    for scenario_name, scenario_data in scenarios.items():
        if 'pv_revenue' in scenario_data and 'pv_cost' in scenario_data:
            # Use provided PV values from image
            df[f'{scenario_name} PV Revenue'] = scenario_data['pv_revenue']
            df[f'{scenario_name} PV Cost'] = scenario_data['pv_cost']
        else:
            # Calculate PV if not provided
            pv_revenue = []
            pv_cost = []
            for i, year in enumerate(years):
                # PV = FV / (1 + r)^n
                pv_r = scenario_data['revenue_multiplier'][i] / ((1 + discount_rate) ** year)
                pv_c = df['Total Costs'].iloc[i] / ((1 + discount_rate) ** year)
                if i == 0:
                    # Year 1 includes upfront cost
                    pv_c += upfront_cost / (1 + discount_rate)
                pv_revenue.append(pv_r)
                pv_cost.append(pv_c)
            
            df[f'{scenario_name} PV Revenue'] = pv_revenue
            df[f'{scenario_name} PV Cost'] = pv_cost
    
    return df, upfront_cost, sq_ft, maintenance_rate, discount_rate, scenarios


# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="DCF Model", layout="wide")

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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='page-title'>Discounted Cash Flow (DCF) Model</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Financial Projections and NPV Analysis</div>", unsafe_allow_html=True)

# Load data
df, upfront_cost, sq_ft, maintenance_rate, discount_rate, scenarios = create_dcf_data()

# Key Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Upfront Cost", f"${upfront_cost:,.2f}")
with col2:
    st.metric("Facility Size", f"{sq_ft:,} sq ft")
with col3:
    st.metric("Maintenance Rate", f"${maintenance_rate:.2f}/sq ft")
with col4:
    st.metric("Discount Rate", f"{discount_rate*100:.4f}%")

st.caption("Note: Discount rate based on long-term treasury yield. Lost income tax is negligible.")

# Student Enrollment and Revenue
st.markdown("<div class='section-header'>Student Enrollment and Revenue</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Student enrollment chart
    fig_students = go.Figure()
    fig_students.add_trace(go.Bar(
        x=df['Year'],
        y=df['Students'],
        name='Students',
        marker_color='#5ec962',
        hovertemplate='<b>Year %{x}</b><br>Students: %{y}<extra></extra>'
    ))
    fig_students.update_layout(
        title='Student Enrollment by Year',
        xaxis_title='Year',
        yaxis_title='Number of Students',
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_students, use_container_width=True)

with col2:
    # Revenue chart
    fig_revenue = go.Figure()
    fig_revenue.add_trace(go.Bar(
        x=df['Year'],
        y=df['Revenue'],
        name='Revenue',
        marker_color='#21918c',
        hovertemplate='<b>Year %{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>'
    ))
    fig_revenue.update_layout(
        title='Annual Revenue by Year',
        xaxis_title='Year',
        yaxis_title='Revenue ($)',
        yaxis=dict(tickformat='$,.0f'),
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_revenue, use_container_width=True)

# Costs Breakdown
st.markdown("<div class='section-header'>Costs Breakdown</div>", unsafe_allow_html=True)

# Wages breakdown
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Wages by Year")
    wages_df = df[['Year', 'Director Wage', 'Faculty Wage', 'FTE Wages', 'Total Wages']].copy()
    fig_wages = go.Figure()
    fig_wages.add_trace(go.Bar(
        x=wages_df['Year'],
        y=wages_df['Director Wage'],
        name='Director of Audiology',
        marker_color='#5ec962',
        hovertemplate='<b>Year %{x}</b><br>Director: $%{y:,.2f}<extra></extra>'
    ))
    fig_wages.add_trace(go.Bar(
        x=wages_df['Year'],
        y=wages_df['Faculty Wage'],
        name='Faculty',
        marker_color='#21918c',
        hovertemplate='<b>Year %{x}</b><br>Faculty: $%{y:,.2f}<extra></extra>'
    ))
    fig_wages.add_trace(go.Bar(
        x=wages_df['Year'],
        y=wages_df['FTE Wages'],
        name='3-4 FTE',
        marker_color='#3b528b',
        hovertemplate='<b>Year %{x}</b><br>FTE: $%{y:,.2f}<extra></extra>'
    ))
    fig_wages.update_layout(
        barmode='stack',
        xaxis_title='Year',
        yaxis_title='Wages ($)',
        yaxis=dict(tickformat='$,.0f'),
        height=400
    )
    st.plotly_chart(fig_wages, use_container_width=True)
    st.caption("Note: Lower estimate for DoA as it is a smaller program. FTE increases in Years 4-5.")

with col2:
    st.markdown("#### Total Costs by Year")
    costs_df = df[['Year', 'Total Wages', 'Maintenance Cost', 'Total Costs']].copy()
    fig_costs = go.Figure()
    fig_costs.add_trace(go.Bar(
        x=costs_df['Year'],
        y=costs_df['Total Wages'],
        name='Total Wages',
        marker_color='#5ec962',
        hovertemplate='<b>Year %{x}</b><br>Wages: $%{y:,.2f}<extra></extra>'
    ))
    fig_costs.add_trace(go.Bar(
        x=costs_df['Year'],
        y=costs_df['Maintenance Cost'],
        name='Maintenance',
        marker_color='#21918c',
        hovertemplate='<b>Year %{x}</b><br>Maintenance: $%{y:,.2f}<extra></extra>'
    ))
    fig_costs.update_layout(
        barmode='stack',
        xaxis_title='Year',
        yaxis_title='Costs ($)',
        yaxis=dict(tickformat='$,.0f'),
        height=400
    )
    st.plotly_chart(fig_costs, use_container_width=True)
    st.caption(f"Maintenance cost: ${maintenance_rate:.2f} per sq ft × {sq_ft:,} sq ft = ${df['Maintenance Cost'].iloc[0]:,.2f}")

# Facility Information
st.markdown("<div class='section-header'>Facility Information</div>", unsafe_allow_html=True)
st.info("**Space estimate:** Increased to 10,000 sq ft to account for university needs including classrooms, student work areas, faculty offices, storage, and circulation space, which typically double the footprint of a standalone clinic.")

# State Revenue Impact
st.markdown("<div class='section-header'>State Revenue Impact Scenarios</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Bear Scenario")
    bear_df = df[['Year', 'Bear (Numeric)', 'Base (Numeric)']].copy()
    bear_df['Bear Revenue'] = [df['Revenue'].iloc[i] * (bear_df['Bear (Numeric)'].iloc[i] / bear_df['Base (Numeric)'].iloc[i]) 
                               if bear_df['Base (Numeric)'].iloc[i] != 0 else 0 
                               for i in range(len(bear_df))]
    st.dataframe(bear_df[['Year', 'Bear (Numeric)', 'Bear Revenue']].style.format({
        'Bear (Numeric)': '{:.5f}',
        'Bear Revenue': '${:,.2f}'
    }), use_container_width=True)

with col2:
    st.markdown("#### Base Scenario")
    base_df = df[['Year', 'Base (Numeric)', 'Base (Dollar)', 'Revenue']].copy()
    base_df['Total Revenue'] = base_df['Revenue'] + base_df['Base (Dollar)']
    st.dataframe(base_df[['Year', 'Base (Numeric)', 'Base (Dollar)', 'Total Revenue']].style.format({
        'Base (Numeric)': '{:.5f}',
        'Base (Dollar)': '${:,.2f}',
        'Total Revenue': '${:,.2f}'
    }), use_container_width=True)

with col3:
    st.markdown("#### Bull Scenario")
    bull_df = df[['Year', 'Bull (Numeric)', 'Bull (Dollar)', 'Revenue']].copy()
    bull_df['Total Revenue'] = bull_df['Revenue'] + bull_df['Bull (Dollar)']
    st.dataframe(bull_df[['Year', 'Bull (Numeric)', 'Bull (Dollar)', 'Total Revenue']].style.format({
        'Bull (Numeric)': '{:.5f}',
        'Bull (Dollar)': '${:,.2f}',
        'Total Revenue': '${:,.2f}'
    }), use_container_width=True)

# State Revenue Impact Chart
fig_scenarios = go.Figure()
fig_scenarios.add_trace(go.Scatter(
    x=df['Year'],
    y=df['Base (Dollar)'],
    mode='lines+markers',
    name='Base Scenario',
    line=dict(color='#5ec962', width=3),
    marker=dict(size=10),
    hovertemplate='<b>Base</b><br>Year %{x}<br>Additional Revenue: $%{y:,.2f}<extra></extra>'
))
fig_scenarios.add_trace(go.Scatter(
    x=df['Year'],
    y=df['Bull (Dollar)'],
    mode='lines+markers',
    name='Bull Scenario',
    line=dict(color='#21918c', width=3),
    marker=dict(size=10),
    hovertemplate='<b>Bull</b><br>Year %{x}<br>Additional Revenue: $%{y:,.2f}<extra></extra>'
))
fig_scenarios.update_layout(
    title='State Revenue Impact by Scenario',
    xaxis_title='Year',
    yaxis_title='Additional Revenue ($)',
    yaxis=dict(tickformat='$,.0f'),
    height=400
)
st.plotly_chart(fig_scenarios, use_container_width=True)

# NPV Analysis
st.markdown("<div class='section-header'>Net Present Value (NPV) Analysis</div>", unsafe_allow_html=True)

# Create NPV comparison chart
fig_npv = go.Figure()
scenario_names = ['Bear', 'Base', 'Bull']
npv_values = [scenarios['Bear']['npv'], scenarios['Base']['npv'], scenarios['Bull']['npv']]
colors = ['#440154', '#5ec962', '#21918c']

fig_npv.add_trace(go.Bar(
    x=scenario_names,
    y=npv_values,
    marker_color=colors,
    hovertemplate='<b>%{x} Scenario</b><br>NPV: $%{y:,.2f}<extra></extra>',
    text=[f'${val:,.2f}' for val in npv_values],
    textposition='outside'
))
fig_npv.update_layout(
    title='Net Present Value by Scenario',
    xaxis_title='Scenario',
    yaxis_title='NPV ($)',
    yaxis=dict(tickformat='$,.0f'),
    height=400,
    showlegend=False
)
st.plotly_chart(fig_npv, use_container_width=True)

# Detailed PV and NPV tables for each scenario
for scenario_name in scenario_names:
    st.markdown(f"#### {scenario_name} Scenario - Present Value Analysis")
    pv_df = df[['Year', f'{scenario_name} PV Revenue', f'{scenario_name} PV Cost']].copy()
    pv_df['PV Net'] = pv_df[f'{scenario_name} PV Revenue'] - pv_df[f'{scenario_name} PV Cost']
    pv_df['Cumulative NPV'] = pv_df['PV Net'].cumsum()
    
    # Format for display
    display_df = pv_df.copy()
    display_df[f'{scenario_name} PV Revenue'] = display_df[f'{scenario_name} PV Revenue'].apply(lambda x: f'${x:,.2f}')
    display_df[f'{scenario_name} PV Cost'] = display_df[f'{scenario_name} PV Cost'].apply(lambda x: f'${x:,.2f}')
    display_df['PV Net'] = display_df['PV Net'].apply(lambda x: f'${x:,.2f}')
    display_df['Cumulative NPV'] = display_df['Cumulative NPV'].apply(lambda x: f'${x:,.2f}')
    
    st.dataframe(display_df, use_container_width=True)
    
    # Highlight final NPV
    final_npv = scenarios[scenario_name]['npv']
    st.metric(f"{scenario_name} Scenario NPV", f"${final_npv:,.2f}")

# Full Data Table
st.markdown("<div class='section-header'>Complete Financial Data</div>", unsafe_allow_html=True)
with st.expander("View Complete Data Table", expanded=False):
    # Format all currency columns
    display_full_df = df.copy()
    currency_cols = ['Tuition per Student', 'Revenue', 'Director Wage', 'Faculty Wage', 'FTE Wages', 
                     'Total Wages', 'Maintenance Cost', 'Total Costs', 'Base (Dollar)', 'Bull (Dollar)']
    for col in currency_cols:
        if col in display_full_df.columns:
            display_full_df[col] = display_full_df[col].apply(lambda x: f'${x:,.2f}' if pd.notna(x) else '')
    
    # Format PV columns
    for scenario in scenario_names:
        if f'{scenario} PV Revenue' in display_full_df.columns:
            display_full_df[f'{scenario} PV Revenue'] = display_full_df[f'{scenario} PV Revenue'].apply(
                lambda x: f'${x:,.2f}' if pd.notna(x) else '')
        if f'{scenario} PV Cost' in display_full_df.columns:
            display_full_df[f'{scenario} PV Cost'] = display_full_df[f'{scenario} PV Cost'].apply(
                lambda x: f'${x:,.2f}' if pd.notna(x) else '')
    
    st.dataframe(display_full_df, use_container_width=True)

