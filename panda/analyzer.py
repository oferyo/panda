import numpy as np


class Analyzer:
    def __init__(self):
        pass


def calc_daily_returns(wealth):
    # return np.log(wealth / wealth.shift(1))
    return wealth / wealth.shift(1) - 1


def calc_annual_returns(daily_returns):
    grouped = np.exp(daily_returns.groupby(
        lambda date: date.year).sum())-1
    return grouped


def calc_max_draw_down():
    pass


def calc_portfolio_var(returns, n, weights=None):
    if weights is None:
        weights = np.ones(returns.columns.size) / \
        returns.columns.size
    sigma = np.cov(returns.T,ddof=0)
    var = (weights * sigma * weights.T).sum()
    return var


def sharpe_ratio(returns, risk_free_rate = 0.0):
    means = returns.mean()
    var = returns.var()
    return (means - risk_free_rate)/np.sqrt(var)


def analyze(wealth):
    daily_returns = calc_daily_returns(wealth)

    annual_returns = calc_annual_returns(daily_returns)
    # annual_returns.to_csv("annual_returns.csv")
    sharpe_ratio_result = sharpe_ratio(daily_returns) * np.sqrt(252)

    print "sharp", sharpe_ratio_result, "\nannual_returns::\n",  annual_returns