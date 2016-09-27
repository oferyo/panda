import finsymbols
import numpy as np
import pandas as pd
from pandas_datareader import data, wb
import datetime
from panda import analyzer
from panda.Logger import my_logger

logger = my_logger("panda")


class Portfolio:

    def __init__(self, stocks_data, initial_wealth):
        logger.info("init start")
        num_symbols = stocks_data.columns.size
        self.fractions = self.set_fractions(num_symbols)
        self.initial_wealth = initial_wealth
        quantities = (initial_wealth * self.fractions) * np.array(1.0 / np.squeeze(stocks_data[0:1].values))
        self.quantities = np.floor(quantities)
        self.cash = initial_wealth - np.dot(stocks_data[0:1], self.quantities)

        self.wealth = self.quantities * stocks_data[0:]
        self.total_wealth = self.wealth.apply(lambda x : np.sum(x), axis = 1, raw=False)
        self.total_wealth.name = 'total_wealth  '
        print "end init portfolio cash ", self.cash

    def analyzer_buy_and_hold(self):
        analyzer.analyze('buy_and_hold', self.total_wealth)


    @staticmethod
    def set_fractions(num_symbols):
        return np.squeeze(np.ones((1, num_symbols), float) / num_symbols)

    def calc_wealth(self, stocks_data, fees_per_share = 0.005, min_fees = 1, reb_period = 1, partial_param = 1):
        len_data = len(stocks_data.index)
        current_quantities = self.quantities
        rebalanced_wealth = self.wealth.copy()
        total_rebalanced = self .total_wealth.copy()
        # total_rebalanced.name = "total_rebalanced"

        for i in range(len_data):
            row_values = stocks_data[i:i+1].values
            rebalanced_wealth[i: i + 1] = current_quantities * row_values
            current_wealth = np.sum(rebalanced_wealth[i: i + 1].values) + self.cash
            # print 'current_wealth ', current_wealth, 'cash', self.cash
            total_rebalanced[i : i+1] = current_wealth
            updated_quantities = np.floor((current_wealth * self.fractions) * np.array(1.0 / np.squeeze(row_values)))
            # logging.info("uq1 %s cq %s ", updated_quantities, current_quantities)
            updated_quantities = np.floor(updated_quantities*partial_param + (1.0 - partial_param)*current_quantities)
            # logging.info("uq2 %s", updated_quantities)

            # print 'cq', current_quantities, 'uq', updated_quantities, 'v', np.squeeze(row_values)
            change = current_wealth - np.sum((updated_quantities * row_values))
            # logging.info("change %s", change)
            self.cash = change
            self.pay_fees(current_quantities, updated_quantities, fees_per_share, min_fees)
            current_quantities = updated_quantities

        print 'end rebalance'
        analyzer.analyze('rebalnce', total_rebalanced)

        # self.data = self.data.join(all_quantities)
        # self.data = self.data.join(total_rebalanced)
        # self.data.to_csv("all_data.csv")
        print "end cash", self.cash

    def pay_fees(self, current_quantities, updated_quantities, fees_per_share, min_fees):
        num_shares = np.sum(np.abs(current_quantities - updated_quantities))
        if num_shares > 0:
            fees_to_pay = np.max((min_fees, num_shares*fees_per_share))
            if fees_to_pay > 0.00001:
                # print 'paid fees ', fees_to_pay,  'num_shares', num_shares, 'cash', (self.cash - fees_to_pay)
                self.cash -= fees_to_pay


def get_and_clean_data(symbols, start_time, end_time):
    stocks_data = data.get_data_yahoo(symbols, start=start_time, end=end_time)['Adj Close']
    len_data = len(stocks_data.index)
    orig_col_len = stocks_data.columns.size
    stocks_data = stocks_data.dropna(how='all')
    stocks_data = stocks_data.dropna(axis=1, how='any', thresh=int(len_data * 0.8))
    new_col_len = stocks_data.columns.size
    stocks_data = stocks_data.fillna(method='pad')
    stocks_data = stocks_data.dropna(how='any')
    new_len_data = len(stocks_data.index)

    print "len_data", len_data, "new_len_data", new_len_data, "col_len", orig_col_len, "new_col", new_col_len
    return stocks_data


def run_round(all_stocks, start_time, end_time):
    # symbol_len = 20
    symbol_len = 20
    symbols = []
    symbols_set = set()
    for j in range(symbol_len):
        symbols_set.add(np.random.randint(1, len(all_stocks)))

    for k in symbols_set:
        try:
            sym = all_stocks[k]["symbol"]
            symbols.append(sym)
        except:
            pass

    initial_wealth = 100000
    stocks_data = get_and_clean_data(symbols, start_time, end_time)
    portfolio = Portfolio(stocks_data, initial_wealth)

    portfolio.analyzer_buy_and_hold()
    fees = 0.01

    logger.info('start no fees D=1')
    portfolio.calc_wealth(stocks_data, 0.0, 0.0)

    logger.info('start with fees D=1')
    portfolio = Portfolio(stocks_data, initial_wealth)
    portfolio.calc_wealth(stocks_data, fees, 1.0, 1.0, 1.0)

    logger.info('start with fees D=0.5')
    portfolio = Portfolio(stocks_data, initial_wealth)
    portfolio.calc_wealth(stocks_data, fees, 1.0, 1.0, 0.5)

    logger.info('start with fees D=0.01')
    portfolio = Portfolio(stocks_data, initial_wealth)
    portfolio.calc_wealth(stocks_data, fees, 1.0, 1.0, 0.5)


    logger.info("the end")

def init():
    global sum
    sum = 0


def log_loss(label, pred):
    return -np.mean(label*np.log(pred[:,0]) + (1 - label) * np.log(1.0 - pred[:,0]))


class Metric:

    def __init__(self, func):
        self.f = func

    def activate(self, x, y):
        return self.f(x,y)



def foo(x, y):
    return x+y


def main():

    metric = Metric(foo)
    z = metric.activate(44, 44)

    # start_time = datetime.datetime(1980, 10, 1)
    # start_time = datetime.datetime(1980, 10, 1)
    start_time = datetime.datetime(1980, 10, 1)
    end_time = datetime.datetime(2016, 10, 8)

    # all_stocks = finsymbols.get_sp500_symbols()
    all_stocks = finsymbols.get_sp500_symbols()
    all_stocks.append(finsymbols.get_nasdaq_symbols())
    all_stocks.append(finsymbols.get_nyse_symbols())


    for j in range(5):
        print 'start_round\t', j
        logger.info('start round %s', j)
        run_round(all_stocks, start_time, end_time)
        print 'end_round\t', j

    print "the one end"

if __name__ == "__main__":
    # execute only if run as a script
    main()