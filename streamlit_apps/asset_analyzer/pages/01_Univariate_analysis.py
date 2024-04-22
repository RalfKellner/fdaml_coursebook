import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


@st.cache_data
def collect_prices_and_returns(ticker, time_period):
    df = yf.download(ticker, period = time_period)
    df.loc[:, "r_t"] = df.loc[:, ["Adj Close"]].pct_change()
    df.dropna(inplace = True)
    return df

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
    df_returns = df.loc[:, ["r_t"]]
    df_returns.columns = ["r_t"]
    df_returns_log = df_returns.apply(lambda x: np.log(1 + x))
    df_weekly_returns_log = df_returns_log.reset_index().groupby(pd.Grouper(key = 'Date', freq = "W")).mean()
    df_weekly_returns  = df_weekly_returns_log.apply(lambda x: np.exp(x) - 1)
    df_weekly_returns.columns = ["r_t"]
    df_monthly_returns_log = df_returns_log.reset_index().groupby(pd.Grouper(key = 'Date', freq = "ME")).mean()
    df_monthly_returns  = df_monthly_returns_log.apply(lambda x: np.exp(x) - 1)
    df_monthly_returns.columns = ["r_t"]
    skewness_estimates = [df_returns.skew().values[0], df_weekly_returns.skew().values[0], df_monthly_returns.skew().values[0]]
    kurtosis_estimates = [df_returns.kurtosis().values[0], df_weekly_returns.kurtosis().values[0], df_monthly_returns.kurtosis().values[0]]


    if frequency == "daily":
        descriptive = generate_descriptives(df_returns)
    elif frequency == "weekly":
        descriptive = generate_descriptives(df_weekly_returns)    
    elif frequency == "monthly":
        descriptive = generate_descriptives(df_monthly_returns)    

    if frequency == "daily":
        fig = make_ts_hist_plot(df_returns)
    elif frequency == "weekly":
        fig = make_ts_hist_plot(df_weekly_returns)
    elif frequency == "monthly":
        fig = make_ts_hist_plot(df_monthly_returns)

    st.title("Descriptive and visual analysis")
    st.markdown("""
                We start with some simple descriptive statistics and visualizations. Below you can see the number of observations,
                mean, standard deviation, minimum, maximum, some quantiles, skewness and kurtosis. These statistics provide a first
                insight how profitable and risky an asset may be. While the mean quantifies profitability and standard deviation the risk, 
                examining skewness, kurtosis and estimates for more extreme price movements can be used to estimate the so called tail
                risks of an investment. Left skewed distributions with high excess kurtosis come along with higher occurence probabilities for
                negative return realizations.
                """)

    st.markdown("**Descriptive statistics**")
    st.dataframe(descriptive.transpose())

    st.markdown("**Empirical distribution and time series**")
    st.plotly_chart(fig)
else:
    st.text("Please enter ticker symbol on the left!")

