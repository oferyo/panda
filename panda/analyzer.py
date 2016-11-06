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


def calc_max_draw_down(wealth_history, start_date = None, end_date = None):
    if (start_date):
        wealth_history = wealth_history[(wealth_history.index > start_date)]
    max_loss = 0.0
    #iterate over all pairs and check the max_loss
    for i in range(len(wealth_history.index)):
        for j in range(len(wealth_history.index)):
            loss = wealth_history[j] / wealth_history[i] - 1
            if loss > max_loss:
                # print "new max_loss", loss, "time i: ", wealth_history.index[i], "time j: ", wealth_history.index[j]
                max_loss = loss
    return max_loss


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

def analyze(name, wealth, analysis_results):
    g_r = calc_growth_rate(wealth)
    # annual_sharp = calc_annual_sharp(calc_daily_returns(wealth))
    # max_draw_down = calc_max_draw_down(wealth, '2001-01-01')
    analysis_results['g_r_' + name] = g_r
    # analysis_results['m_d_d_' + name] = max_draw_down