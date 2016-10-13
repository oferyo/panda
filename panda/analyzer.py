import numpy as np
import pandas as pd
from panda.Logger import my_logger

logger = my_logger("panda")


class Analyzer:
    def __init__(self):
        pass


def calc_daily_returns(wealth):
    # return np.log(wealth / wealth.shift(1))
    return wealth / wealth.shift(1) - 1


def calc_growth_rate(wealth):
    return np.log(wealth / wealth[0])


def calc_annual_returns(daily_returns):
    grouped = daily_returns.groupby(lambda date: date.year).sum()
    return grouped


def calc_annual_sharp(daily_returns):
    group = daily_returns.groupby(lambda date: date.year)
    grouped_mean = group.mean()
    grouped_var = group.var()
    return (grouped_mean / np.sqrt(grouped_var)) * np.sqrt(252)


def calc_max_draw_down():
    pass


def calc_portfolio_var(returns, n, weights=None):
    if weights is None:
        weights = np.ones(returns.columns.size) / \
                  returns.columns.size
    sigma = np.cov(returns.T, ddof=0)
    var = (weights * sigma * weights.T).sum()
    return var


def sharpe_ratio(returns, risk_free_rate=0.0):
    means = returns.mean()
    var = returns.var()
    return (means - risk_free_rate) / np.sqrt(var)

def analyze(name, wealth, growth_result):
    g_r = calc_growth_rate(wealth)
    annual_sharp = calc_annual_sharp(calc_daily_returns(wealth))
    growth_result['g_r_' + name] = g_r