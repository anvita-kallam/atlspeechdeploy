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


def create_roi_data():
    """Create ROI data from the image."""
    data = {
        'Year': [1, 2, 3, 4, 5],
        'Students Enrolled': [15, 30, 45, 60, 60],
        'Tuition per Student': [15877.00, 15877.00, 15877.00, 15877.00, 15877.00],
        'Revenue': [238155.00, 476310.00, 714465.00, 952620.00, 952620.00],
        'Variable Cost per Student': [4933.33, 500.00, 0, 84.70, 0],
        'Total Variable Cost': [73999.95, 15000.00, 0, 5082.00, 0],
        'Fixed Costs': [885437.00, 885437.00, 885437.00, 885437.00, 885437.00],
        'Total Costs': [959436.95, 900437.00, 885437.00, 890519.00, 885437.00],
        'Earnings Before Tax': [-721281.95, -424127.00, -170972.00, 62101.00, 67183.00],
        'Tax': [0, 0, 0, 3347.24, 3621.16],
        'Earnings After Tax': [-721281.95, -424127.00, -170972.00, 58753.76, 63561.84],
        'Cumulative Cash Flow': [-911281.95, -1335408.95, -1506380.95, -1447627.19, -1384065.36]
    }
    return pd.DataFrame(data)


# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Final ROI Model", layout="wide")

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
    
    /* Prevent absolute positioning overlap in custom elements only - but don't affect Streamlit internals */
    .section-header,
    .card,
    .stat-block,
    .page-title,
    .subtitle {
      position: relative;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='page-title'>Georgia AuD Program - Return on Investment Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Financial Performance Analysis (10/31/2025)</div>", unsafe_allow_html=True)

# Load ROI data
roi_df = create_roi_data()

# Display key metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Initial Investment", "$190,000.00")
with col2:
    st.metric("Tax Rate", "5%")
with col3:
    final_cumulative = roi_df['Cumulative Cash Flow'].iloc[-1]
    st.metric("Final Cumulative Cash Flow", f"${final_cumulative:,.2f}")

# ROI Conclusion
st.markdown("<div class='section-header'>ROI Analysis Conclusion</div>", unsafe_allow_html=True)
st.warning("**Negative return on investment** - The investment is not earned back.")

# Create charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Cumulative Cash Flow vs. Earnings After Tax")
    # Line chart
    fig_line = go.Figure()
    
    # Cumulative cash flow line (blue from palette)
    fig_line.add_trace(go.Scatter(
        x=roi_df['Year'],
        y=roi_df['Cumulative Cash Flow'],
        mode='lines+markers',
        name='Cumulative Cash Flow',
        line=dict(color='#3b528b', width=3),
        marker=dict(size=10),
        hovertemplate='<b>Cumulative Cash Flow</b><br>Year: %{x}<br>Amount: $%{y:,.2f}<extra></extra>'
    ))
    
    # Earnings after tax line (dark purple from palette)
    fig_line.add_trace(go.Scatter(
        x=roi_df['Year'],
        y=roi_df['Earnings After Tax'],
        mode='lines+markers',
        name='Earnings After Tax',
        line=dict(color='#440154', width=3),
        marker=dict(size=10),
        hovertemplate='<b>Earnings After Tax</b><br>Year: %{x}<br>Amount: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add zero line
    fig_line.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        annotation_text="Break-even"
    )
    
    fig_line.update_layout(
        height=400,
        xaxis_title="Year",
        yaxis_title="Amount ($)",
        yaxis=dict(tickformat='$,.0f', range=[-1600000, 200000]),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.markdown("#### Revenue vs. Total Costs")
    # Bar chart
    fig_bar = go.Figure()
    
    # Revenue bars (green from palette)
    fig_bar.add_trace(go.Bar(
        x=roi_df['Year'],
        y=roi_df['Revenue'],
        name='Revenue',
        marker_color='#5ec962',
        hovertemplate='<b>Revenue</b><br>Year: %{x}<br>Amount: $%{y:,.2f}<extra></extra>'
    ))
    
    # Total costs bars (red)
    fig_bar.add_trace(go.Bar(
        x=roi_df['Year'],
        y=roi_df['Total Costs'],
        name='Total Costs',
        marker_color='#d62728',  # Red
        hovertemplate='<b>Total Costs</b><br>Year: %{x}<br>Amount: $%{y:,.2f}<extra></extra>'
    ))
    
    fig_bar.update_layout(
        height=400,
        xaxis_title="Year",
        yaxis_title="Amount ($)",
        yaxis=dict(tickformat='$,.0f', range=[0, 1200000]),
        barmode='group',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

# Display full data table in dropdown
with st.expander("Financial Data Table", expanded=False):
    st.dataframe(roi_df, use_container_width=True)

# Format the table with color coding for negative values
st.markdown("#### Financial Summary")
formatted_df = roi_df.copy()

# Format currency columns
currency_cols = ['Tuition per Student', 'Revenue', 'Total Variable Cost', 'Fixed Costs', 
                 'Total Costs', 'Earnings Before Tax', 'Tax', 'Earnings After Tax', 'Cumulative Cash Flow']

for col in currency_cols:
    if col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f'${x:,.2f}' if pd.notna(x) else '')

# Create HTML table with color coding
html_table = '<table style="width:100%; border-collapse: collapse;">'
html_table += '<thead><tr>'
for col in formatted_df.columns:
    html_table += f'<th style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;">{col}</th>'
html_table += '</tr></thead><tbody>'

for idx, row in formatted_df.iterrows():
    html_table += '<tr>'
    for col in formatted_df.columns:
        val = row[col]
        # Color code negative values in red
        if col in ['Earnings Before Tax', 'Earnings After Tax', 'Cumulative Cash Flow']:
            if isinstance(val, str) and val.startswith('$-'):
                html_table += f'<td style="padding: 8px; color: #d62728; font-weight: bold;">{val}</td>'
            elif isinstance(val, (int, float)) and val < 0:
                html_table += f'<td style="padding: 8px; color: #d62728; font-weight: bold;">${val:,.2f}</td>'
            else:
                html_table += f'<td style="padding: 8px;">{val}</td>'
        else:
            html_table += f'<td style="padding: 8px;">{val}</td>'
    html_table += '</tr>'

html_table += '</tbody></table>'
st.markdown(html_table, unsafe_allow_html=True)

