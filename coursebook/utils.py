import requests
import zipfile
import io
import statsmodels.api as sm
import re
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def hill_estimator(data: np.ndarray, k: int, tail: str = "upper") -> float:
    if k <= 0 or k >= len(data):
        raise ValueError("k must be between 1 and len(data) - 1")

    if tail == "upper":
        sorted_data = np.sort(data)[::-1]
    elif tail == "lower":
        sorted_data = np.sort(-data)[::-1]
    else:
        raise ValueError("tail must be 'upper' or 'lower'")

    top_k = sorted_data[:k]
    x_k1 = sorted_data[k]

    logs = np.log(top_k) - np.log(x_k1)
    hill_est = np.mean(logs)

    return hill_est

def describe_with_moments(data, percentiles = (0.05, 0.95), fisher=True, include_hill=True, k_ratio=0.05):
    """
    Extended descriptive stats with skewness, kurtosis, and Hill estimator.

    Parameters:
    - data: pd.Series or pd.DataFrame
    - fisher: If True, returns Fisher kurtosis (normal=0); else Pearson (normal=3)
    - include_hill: Whether to include Hill estimator for tails (only for Series)
    - k_ratio: Fraction of sample to use for Hill estimator (e.g., 0.05 means top 5%)

    Returns:
    - pd.Series or pd.DataFrame with added statistics
    """
    if isinstance(data, pd.Series):
        desc = data.describe(percentiles=percentiles)
        clean_data = data.dropna().values
        k = int(len(clean_data) * k_ratio)

        extra = {
            "skew": skew(clean_data),
            "kurtosis": kurtosis(clean_data, fisher=fisher)
        }

        if include_hill and k > 0:
            extra["hill_upper"] = hill_estimator(clean_data, k, tail="upper")
            extra["hill_lower"] = hill_estimator(clean_data, k, tail="lower")

        return pd.concat([desc, pd.Series(extra)])

    elif isinstance(data, pd.DataFrame):
        desc = data.describe(percentiles=percentiles)
        for col in data.columns:
            col_data = data[col].dropna().values
            k = int(len(col_data) * k_ratio)
            desc.loc["skew", col] = skew(col_data)
            desc.loc["kurtosis", col] = kurtosis(col_data, fisher=fisher)
            if include_hill and k > 0:
                desc.loc["hill_upper", col] = hill_estimator(col_data, k, "upper")
                desc.loc["hill_lower", col] = hill_estimator(col_data, k, "lower")
        return desc

    else:
        raise TypeError("Input must be a pandas Series or DataFrame")
    

def get_ff_factors(num_factors = 3 , frequency = 'daily', in_percentages = False):

    ''' 
    This function downloades directly the current txt files from the Keneth R. French homepage (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

    Parameters:
    -----------
    num_factors (3 or 5): int
        download either data for the three or the five factor portfolios

    frequency: str
        either daily, weekly or monthly for three factor data or daily or monthly for five factor data
    
    Returns:
    ---------
    pd.DataFrame
    
    '''

    assert num_factors in [3, 5], 'The number of factors must be 3 or 5'

    if num_factors == 3:
        assert frequency in ['daily', 'weekly', 'monthly'], 'frequency for the three factors model must be either daily, weekly or monthly'
        if frequency == 'daily':
            french_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_TXT.zip'
            r = requests.get(french_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open('F-F_Research_Data_Factors_daily.txt') as file:
                out = file.readlines()
            data_lines = []
            for line in out:
                string_line = line.decode('UTF-8')
                if string_line[0].isdigit():
                    string_line = string_line.split()
                    data_lines.append(string_line)
            ff_data = pd.DataFrame(data_lines, columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
            ff_data.loc[:, 'date'] = [date[:4] + '-' + date[4:6] + '-' + date[6:] for date in ff_data.date]
            ff_data.set_index('date', inplace = True)
            ff_data = ff_data.astype(float)
        elif frequency == 'weekly':
            french_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_TXT.zip'
            r = requests.get(french_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open('F-F_Research_Data_Factors_weekly.txt') as file:
                out = file.readlines()
            data_lines = []
            for line in out:
                string_line = line.decode('UTF-8')
                if string_line[0].isdigit():
                    string_line = string_line.split()
                    data_lines.append(string_line)
            ff_data = pd.DataFrame(data_lines, columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
            ff_data.loc[:, 'date'] = [date[:4] + '-' + date[4:6] + '-' + date[6:] for date in ff_data.date]
            ff_data.set_index('date', inplace = True)
            ff_data = ff_data.astype(float)
        elif frequency == 'monthly':
            french_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_TXT.zip'
            r = requests.get(french_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open('F-F_Research_Data_Factors.txt') as file:
                out = file.readlines()
            data_lines = []
            for line in out:
                string_line = line.decode('UTF-8')
                if re.search('Annual', string_line):
                    break
                elif string_line[0].isdigit():
                    string_line = string_line.split()
                    data_lines.append(string_line)
            ff_data = pd.DataFrame(data_lines, columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
            ff_data.loc[:, 'date'] = [date[:4] + '-' + date[4:6] for date in ff_data.date]
            ff_data.set_index('date', inplace = True)
            ff_data = ff_data.astype(float)
    elif num_factors == 5:
        assert frequency in ['daily', 'monthly'], 'frequency for the five factor model must be either daily or monthly'
        if frequency == 'daily':
            french_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_TXT.zip'
            r = requests.get(french_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open('F-F_Research_Data_5_Factors_2x3_daily.txt') as file:
                out = file.readlines()
            data_lines = []
            for line in out:
                string_line = line.decode('UTF-8')
                if string_line[0].isdigit():
                    string_line = string_line.split()
                    data_lines.append(string_line)
            ff_data = pd.DataFrame(data_lines, columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
            ff_data.loc[:, 'date'] = [date[:4] + '-' + date[4:6] + '-' + date[6:] for date in ff_data.date]
            ff_data.set_index('date', inplace = True)
            ff_data = ff_data.astype(float)
        elif frequency == 'monthly':
            french_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_TXT.zip'
            r = requests.get(french_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open('F-F_Research_Data_5_Factors_2x3.txt') as file:
                out = file.readlines()
            data_lines = []
            for line in out:
                string_line = line.decode('UTF-8')
                if re.search('Annual', string_line):
                    break
                elif string_line[0].isdigit():
                    string_line = string_line.split()
                    data_lines.append(string_line)
            ff_data = pd.DataFrame(data_lines, columns = ['date', 'Mkt-RF', 'SMB', 'HML','RMW', 'CMA', 'RF'])
            ff_data.loc[:, 'date'] = [date[:4] + '-' + date[4:6] for date in ff_data.date]
            ff_data.set_index('date', inplace = True)
            ff_data = ff_data.astype(float)

    if in_percentages == True:
        return ff_data
    else:
        return ff_data / 100
    

def empirical_value_at_risk(returns: pd.Series, alpha: float = 0.99) -> float:
    """
    Compute historical (unconditional) Value at Risk (VaR) at confidence level alpha.
    VaR is reported as a positive number indicating potential loss.
    """
    return -np.quantile(returns, 1 - alpha)

def empirical_expected_shortfall(returns: pd.Series, alpha: float = 0.99) -> float:
    """
    Compute historical (unconditional) Expected Shortfall (ES) at confidence level alpha.
    ES is the average loss beyond the VaR threshold.
    """
    var = np.quantile(returns, 1 - alpha)
    return -returns[returns <= var].mean()
    
