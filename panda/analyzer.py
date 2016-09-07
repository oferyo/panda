import numpy as np


class Analyzer:
    def __init__(self):
        pass


def calc_daily_returns(wealth):
    return np.log(wealth / wealth.shift(1))


def calc_annual_returns(daily_returns):
    grouped = np.exp(daily_returns.groupby(
        lambda date: date.year).sum())-1
    return grouped


def calc_portfolio_var(returns, weights=None):
    if weights is None:
        weights = np.ones(returns.columns.size) / \
        returns.columns.size
    sigma = np.cov(returns.T,ddof=0)
    var = (weights * sigma * weights.T).sum()
    return var


def sharpe_ratio(returns, weights = None, risk_free_rate = 0.0):
    n = returns.columns.size
    if weights is None: weights = np.ones(n)/n
    # get the portfolio variance
    var = calc_portfolio_var(returns, weights)
    # and the means of the stocks in the portfolio
    means = returns.mean()
    # and return the sharpe ratio
    return (means.dot(weights) - risk_free_rate)/np.sqrt(var)


def analyze(wealth):
    daily_returns = calc_daily_returns(wealth)

    annual_returns = calc_annual_returns(daily_returns)
    annual_returns.to_csv("annual_returns.csv")

    n = len(wealth.columns.index)
    sharpe_ratio_result = sharpe_ratio(daily_returns)  * np.sqrt(n)

    print "sharp", sharpe_ratio_result, "\n", annual_returns
