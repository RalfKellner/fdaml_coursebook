import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm, t, nct
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf

@st.cache_data
def collect_prices_and_returns(ticker, time_period):
    df = yf.download(ticker, period = time_period)
    df.loc[:, "r_t"] = df.loc[:, ["Adj Close"]].pct_change()
    df.dropna(inplace = True)
    return df

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

def make_acf_plot(tseries, nlags = 10):

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.40,
        subplot_titles=["Returns", "Absolute returns"]                
    )


    autocorrs, confint = acf(tseries, nlags = nlags, alpha = 0.05)
    lower_bound = confint[1:, 0] - autocorrs[1:]
    upper_bound = confint[1:, 1] - autocorrs[1:]

    autocorrs_abs, confint_abs = acf(tseries.abs(), nlags = nlags, alpha = 0.05)
    lower_bound_abs = confint_abs[1:, 0] - autocorrs_abs[1:]
    upper_bound_abs = confint_abs[1:, 1] - autocorrs_abs[1:]

    lags = np.arange(1, len(autocorrs)+1)

    # Add autocorrelation line
    fig.add_trace(go.Scatter(x=lags, y=autocorrs[1:], mode='lines+markers', marker_color = '#636EFA', name='Autocorrelation'), row = 1, col = 1)

    # Add confidence band
    fig.add_trace(go.Scatter(x=lags, y=lower_bound, mode='lines', name='Upper Confidence Bound (95%)', line=dict(width=0.5, color='red')), row = 1, col = 1)
    fig.add_trace(go.Scatter(x=lags, y=upper_bound, mode='lines', name='Lower Confidence Bound (95%)', line=dict(width=0.5, color='red'),
                            fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)'), row = 1, col = 1)  # Fill area between bounds

    # Add autocorrelation line
    fig.add_trace(go.Scatter(x=lags, y=autocorrs_abs[1:], mode='lines+markers', marker_color = '#636EFA', name='Autocorrelation'), row = 2, col = 1)

    # Add confidence band
    fig.add_trace(go.Scatter(x=lags, y=lower_bound_abs, mode='lines', name='Upper Confidence Bound (95%)', line=dict(width=0.5, color='red')), row = 2, col = 1)
    fig.add_trace(go.Scatter(x=lags, y=upper_bound_abs, mode='lines', name='Lower Confidence Bound (95%)', line=dict(width=0.5, color='red'),
                            fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)'), row = 2, col = 1)  # Fill area between bounds

    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="Autocorrelation", row=1, col=1)
    fig.update_yaxes(title_text="Autocorrelation", row=2, col=1)

    # Update layout
    fig.update_layout(
    #                xaxis_title='Lag',
    #                yaxis_title='Autocorrelation',
    #                yaxis_range = [-1, 1],
    #                template='plotly_white', 
                    showlegend = False)
    
    return fig


def generate_descriptives(returns):
    descriptives = returns.describe(percentiles = [0.01, 0.025, 0.975, 0.99])
    descriptives.loc["skew"] = returns.skew()
    descriptives.loc["kurtosis"] = returns.kurtosis()
    descriptives.columns = ["Empirical"]
    descriptives = descriptives.loc[["mean", "std", "skew", "kurtosis", "1%", "2.5%", "97.5%", "99%"]]
    

    norm_fit = norm.fit(returns)
    t_fit = t.fit(returns)
    nct_fit = nct.fit(returns)

    ll_norm = np.sum(norm.logpdf(returns, norm_fit[0], norm_fit[1]))
    ll_t = np.sum(t.logpdf(returns, t_fit[0], t_fit[1], t_fit[2]))
    ll_nct = np.sum(nct.logpdf(returns, nct_fit[0], nct_fit[1], nct_fit[2], nct_fit[3]))
    aic_norm = -2 * ll_norm + 2 * len(norm_fit)
    aic_t = -2 * ll_t + 2*len(t_fit)
    aic_nct = -2 * ll_nct + 2*len(nct_fit)

    norm_quantiles = norm.ppf([0.01, 0.025, 0.975, 0.99], norm_fit[0], norm_fit[1])
    norm_moments = norm.stats(norm_fit[0], norm_fit[1], moments = "mvsk")

    t_quantiles = t.ppf([0.01, 0.025, 0.975, 0.99], t_fit[0], t_fit[1], t_fit[2])
    t_moments = t.stats(t_fit[0], t_fit[1], t_fit[2], moments = "mvsk")

    nct_quantiles = nct.ppf([0.01, 0.025, 0.975, 0.99], nct_fit[0], nct_fit[1], nct_fit[2], nct_fit[3])
    nct_moments = nct.stats(nct_fit[0], nct_fit[1], nct_fit[2], nct_fit[3], moments = "mvsk")

    descriptives.loc["mean":"kurtosis":, "Normal"] = norm_moments
    descriptives.loc["std", "Normal"] = np.sqrt(descriptives.loc["std", "Normal"])
    descriptives.loc["1%":, "Normal"] = norm_quantiles

    descriptives.loc["mean":"kurtosis":, "t"] = t_moments
    descriptives.loc["skew", "Normal":"t"] = np.nan
    descriptives.loc["std", "t"] = np.sqrt(descriptives.loc["std", "t"])
    descriptives.loc["1%":, "t"] = t_quantiles

    descriptives.loc["mean":"kurtosis":, "Skew t"] = nct_moments
    descriptives.loc["std", "Skew t"] = np.sqrt(descriptives.loc["std", "Skew t"])
    descriptives.loc["1%":, "Skew t"] = nct_quantiles

    descriptives.loc["LL", :] = np.nan
    descriptives.loc["AIC", :] = np.nan
    descriptives.loc["LL", ["Normal", "t", "Skew t"]] = [ll_norm, ll_t, ll_nct]
    descriptives.loc["AIC", ["Normal", "t", "Skew t"]] = [aic_norm, aic_t, aic_nct]
    return descriptives


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
    df_weekly_returns_log = df_returns_log.reset_index().groupby(pd.Grouper(key = 'Date', freq = "W")).sum()
    df_weekly_returns  = df_weekly_returns_log.apply(lambda x: np.exp(x) - 1)
    df_weekly_returns.columns = ["r_t"]
    df_monthly_returns_log = df_returns_log.reset_index().groupby(pd.Grouper(key = 'Date', freq = "ME")).sum()
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

    if frequency == "daily":
        fig_acf = make_acf_plot(df_returns)
    elif frequency == "weekly":
        fig_acf = make_acf_plot(df_weekly_returns)
    elif frequency == "monthly":
        fig_acf = make_acf_plot(df_monthly_returns)

    st.title("Descriptive, parametric and visual analysis")
    st.markdown("""
                We start with some simple descriptive statistics and visualizations. Below you can see the number of observations,
                mean, standard deviation, minimum, maximum, some quantiles, skewness and kurtosis. These statistics provide a first
                insight how profitable and risky an asset may be. While the mean quantifies profitability and standard deviation the risk, 
                examining skewness, kurtosis and estimates for more extreme price movements can be used to estimate the so called tail
                risks of an investment. Left skewed distributions with high excess kurtosis come along with higher occurrence probabilities for
                negative return realizations. The quantiles provide an impression of potential extreme return levels. The comparison of 
                descriptive statistics and the corresponding metrics for the parametric models further deepen the understanding of the 
                return distribution.
                """)

    st.markdown("**Descriptive statistics**")
    st.dataframe(descriptive, width = 600)

    st.markdown("**Empirical distribution and time series**")
    st.plotly_chart(fig)
    st.markdown("**Autocorrelation plots for the time series**")
    st.plotly_chart(fig_acf)
else:
    st.text("Please enter ticker symbol on the left!")

