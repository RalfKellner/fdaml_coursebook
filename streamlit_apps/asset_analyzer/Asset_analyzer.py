import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(
    page_title="Asset analyzer",
    page_icon=":money_mouth_face:",
)

@st.cache_data
def collect_prices_and_returns(ticker, time_period):
    df = yf.download(ticker, period = time_period)
    df.loc[:, "r_t"] = df.loc[:, ["Adj Close"]].pct_change()
    df.dropna(inplace = True)
    return df

@st.cache_data
def get_info(ticker):
    yf_ticker = yf.Ticker(ticker)
    info_ = yf_ticker.get_info()
    return info_

def generate_descriptives(df_rt):
    descriptive = df_rt.describe(percentiles = [0.05, 0.95])
    descriptive.loc["skewness"] = df_rt.skew().values[0]
    descriptive.loc["kurtosis"] = df_rt.kurtosis().values[0]
    return descriptive

def make_ts_hist_plot(df_rt):
    fig = make_subplots(
        rows=3, cols=1, 
        vertical_spacing=0.10, 
        subplot_titles = ["Empirical distribution", "Discrete returns", "Absolute discrete returns"],
        row_width=[0.3, 0.3, 0.4]
    )
    
    fig.add_trace(go.Histogram(x = df_rt.values.flatten(), name = "empirical distribution", marker_color = '#636EFA', opacity=0.7), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = df_rt.index, y = df_rt.r_t, name = "time series", marker_color = '#636EFA', opacity = 0.7), row = 2, col = 1)
    fig.add_trace(go.Scatter(x = df_rt.index, y = df_rt.r_t.abs(), name = "abs. time series", marker_color = '#636EFA', opacity = 0.7), row = 3, col = 1)
    fig.update_layout(height = 700, width = 600, showlegend=False)

    return fig

st.title("Asset analyzer")
st.markdown("""The following is a *demo* to play around with different elements of the univariate asset return analysis.
            Go to [yahoo finance](https://finance.yahoo.com/), search for an asset of your interest in the search bar and find the
            corresponding ticker symbol. Enter this ticker symbol in the left.\n\n I am going to
            try to add more course relevant analyses to this page as we discuss corresponding methods in the course. On the left you
            can navigate through different analysis types. This is meant to give you the ability to experience the methods discussed
            throughout the course.
            """)


with st.sidebar:
    st.title("Selections")
    ticker = st.text_input(
        'Insert the ticker symbol:',
        value = "AAPL")
    time_period = st.selectbox(
        'Choose a historical time period:',
        ["5y","3y", "1y"])
    frequency = st.selectbox(
        "Choose the time frequency:",
        ["daily", "weekly", "monthly"]
    )
    

if ticker:
    df = collect_prices_and_returns(ticker, time_period)
    info_ = get_info(ticker)

    st.markdown("**General asset information**")
    if 'longBusinessSummary' in info_.keys():
        st.markdown(info_['longBusinessSummary'])
    elif 'description' in info_.keys():
        st.markdown(info_['description'])
    else:
        st.markdown("No general information available for this ticker.")

else:
    st.text("Please enter ticker symbol on the left!")

