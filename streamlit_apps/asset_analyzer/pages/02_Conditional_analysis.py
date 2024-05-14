import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def collect_prices_and_returns(ticker, time_period):
    df = yf.download(ticker, period = time_period)
    df.loc[:, "r_t"] = df.loc[:, ["Adj Close"]].pct_change()
    df.dropna(inplace = True)
    return df


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



if frequency == "daily":
    returns_ts = df_returns.r_t*100
elif frequency == "weekly":
    returns_ts = df_weekly_returns.r_t*100
elif frequency == "monthly":
    returns_ts = df_monthly_returns.r_t*100

am = arch_model(returns_ts, mean = "AR", lags = 1, vol = "GARCH", dist="StudentsT")
res = am.fit()

st.title("Time series analysis")

st.markdown(r"""
            This page is dedicated to the analysis of the time-varying behavior of the time series. Below a table with estimated parameters
            for a AR(1)-GARCH(1, 1) model with student t distributed innovations. The sign of the AR parameter provides information
            regarding the impact of the previous observation on the expected value for the current time step. Note, to examine the 
            corresponding p-value. If it is higher than, e.g., 5% we may treat the true value not to be different from zero.

            The $\alpha$ parameter of the GARCH model indicates how sensitive the variance reacts to previous volatility shocks.
            $\alpha + \beta$ can be interpreted as the rate at which the effect of a current variance shock vanishes.

            The smaller the $\nu$ parameter of the student t distribution, the heavier the tails of the return distribution, the higher
            the probability masses for extreme events.
            """
)

st.write(res.summary())

mu_t = returns_ts - res.resid
sigma_t = res.conditional_volatility
e_t = res.resid / sigma_t
q = e_t.quantile(0.05)
q_t = mu_t + sigma_t * q 

# Create 2x2 subplot layout
fig = make_subplots(rows=2, cols=2, subplot_titles=("Conditional mean", "Conditional vola", "Conditional 5% quantile", "Innovations"))

# Add each time series data to its respective subplot

# Plot r_t, mu_t, and sigma_t in different subplots
fig.add_trace(go.Scatter(x=returns_ts.index, y=returns_ts/100, marker_color = '#636EFA', name="r_t"), row=1, col=1)
fig.add_trace(go.Scatter(x=mu_t.index, y=mu_t/100, marker_color = "#660066", name="mu_t"), row=1, col=1)

fig.add_trace(go.Scatter(x=returns_ts.index, y=returns_ts/100, marker_color = '#636EFA', name="r_t"), row=1, col=2)
fig.add_trace(go.Scatter(x=sigma_t.index, y=sigma_t/100, marker_color = "#660066", name="sigma_t"), row=1, col=2)
fig.add_trace(go.Scatter(x=returns_ts.index, y=[returns_ts.std(ddof = 1)/100]*returns_ts.shape[0], name="std", marker_color = "#660066", line=dict(dash='dash')), row=1, col=2)

# r_t in the third subplot with quantile line
fig.add_trace(go.Scatter(x=returns_ts.index, y=returns_ts/100, marker_color = '#636EFA', name="r_t"), row=2, col=1)
fig.add_trace(go.Scatter(x=q_t.index, y=q_t/100, name="q_t_0.05", marker_color = "#660066", legendgroup = "3"), row=2, col=1)
fig.add_trace(go.Scatter(x=q_t.index, y=[returns_ts.quantile(0.05)/100]*q_t.shape[0], name="q_0.05", marker_color = "#660066", line=dict(dash='dash')), row=2, col=1)

# Plot e_t in the fourth subplot
fig.add_trace(go.Scatter(x=returns_ts.index, y=returns_ts/100, marker_color = '#636EFA', name="r_t"), row=2, col=2)
fig.add_trace(go.Scatter(x=e_t.index, y=e_t/100, marker_color = "#660066", name="e_t"), row=2, col=2)


# Update layout and show plot
fig.update_layout(title="AR(1)-GARCH(1,1) model visualization", height=800, width=800, showlegend = False)

st.markdown(
    r"""
    The plots below visualize the return over time and the conditional mean, standard deviation, 5% quantile and innovations.
    The conditional quantile can be compared with the unconditional estimation (dashed line). These plots illustrate how strongly time
    varying effects are pronounced. Especially the increase in standard deviations and corresponding lower levels for the 5% quantile
    provide an impression what may happen during more adverse conditions w.r.t the risk of asset returns. Such values may be seen
    as conservative worst case like estimates for the risk of an asset.
    """
)
st.plotly_chart(fig)

std_ests = [sigma_t.max()/100, returns_ts.std(ddof = 1)/100, sigma_t.min()/100]
quantile_ests = [q_t.min()/100, returns_ts.quantile(0.05)/100, q_t.max()/100]
qunatile_ests_df = pd.DataFrame([std_ests, quantile_ests], columns = ["worst case", "unconditional", "best case"], index = ["Volatility", "5% quantile"])

st.markdown(
    r"""
    The table below includes the highest and lowest value of the time-varying volatility and 5% quantile over time as well as the 
    unconditional estimates.
    """
)
st.dataframe(qunatile_ests_df, use_container_width=True)
