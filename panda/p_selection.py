import string


import finsymbols
import pandas as pd
import numpy as np
from pandas_datareader import data, wb
import datetime

from panda import analyzer

class Portfolio:

    def __init__(self, symbols, initial_wealth, fractions = None):
        self.symbols = symbols
        self.initial_wealth = initial_wealth
        self.fractions = fractions


    def init_portfolio(self, start_time, end_time):
        self.stocks_data = stocks_data = data.get_data_yahoo(self.symbols, start=start_time, end=end_time)['Adj Close']
        len_data = len(self.stocks_data.index)
        orig_col_len = self.stocks_data.columns.size
        self.stocks_data = stocks_data = self.stocks_data.dropna(how='all')

        self.stocks_data = stocks_data = self.stocks_data.dropna(axis = 1, how='any', thresh = int(len_data / 1.4))

        new_len_data = len(self.stocks_data.index)
        self.stocks_data = stocks_data = self.stocks_data.fillna(method='pad')
        self.stocks_data = stocks_data = self.stocks_data.dropna(how='any')

        # self.stocks_data = stocks_data = self.stocks_data.dropna(how='any')
        new_len_data = len(self.stocks_data.index)
        self.len_data = new_len_data
        new_col_len = self.stocks_data.columns.size
        print "len_data", len_data, "new_len_data", new_len_data, "col_len",orig_col_len, "new_col", new_col_len

        self.fractions = np.squeeze(np.ones((1, new_col_len), float) / new_col_len)

        # self.stocks_data.to_csv("stocks.csv");
        # clean data
        # quantitis = np.dot((self.initial_wealth * self.fractions), 1.0 / np.squeeze(stocks_data[0:1].values))
        quantitis = (self.initial_wealth * self.fractions) * np.array(1.0 / np.squeeze(stocks_data[0:1].values))
        self.quantitis = np.floor(quantitis)
        self.cash = self.initial_wealth - np.dot(stocks_data[0:1], self.quantitis)

        self.wealth = self.quantitis * stocks_data[0:]
        # new_cols = ["w_" +x for x in self.stocks_data.columns]
        # self.wealth.columns = new_cols
        # self.wealth.to_csv("wealth.csv")

        # self.wealth.to_csv("sd.csv")
        self.total_wealth = self.wealth.apply(lambda x : np.sum(x), axis = 1, raw=False)
        # new_cols = ["tw_" +x for x in self.stocks_data.columns]
        # self.total_wealth.columns = new_cols
        self.total_wealth.name = 'total_wealth  '


        # merged = self.stocks_data.join(self.wealth)
        #
        # merged = merged.join(self.total_wealth)
        # # merged.to_csv("sd1.csv")
        # # self.stocks_data.to_csv("sd2.csv")
        # self.data = merged
        analyzer.analyze(self.total_wealth)
        print "end init portfolio cash ", self.cash

    def calc_wealth(self):
        fees = 0.005
        min_fees = 1
        current_wealth = self.initial_wealth
        current_quantitis = self.quantitis

        index = 0
        all_quantities = self.wealth.copy()
        rebalanced_wealth = self.wealth.copy()
        total_rebalanced = self .total_wealth.copy()
        total_rebalanced.name = "total_rebalanced"

        for i in range(self.len_data):
            row_values = self.stocks_data[i:i+1].values

            rebalanced_wealth[i: i + 1] = current_quantitis * row_values
            # current_wealth =  self.wealth[i: i + 1].apply(lambda x : np.sum(x), axis = 1, raw=False)
            current_wealth = np.sum(rebalanced_wealth[i: i + 1].values) + self.cash
            total_rebalanced[i : i+1] = current_wealth
            # print "cw ", current_wealth
            current_quantitis = np.floor((current_wealth * self.fractions) * np.array(1.0 / np.squeeze(row_values)))
            all_quantities[i : i+1] = [current_quantitis]

            change = current_wealth - np.sum((current_quantitis * row_values))
            self.cash = change

        # self.data = self.data.join(rebalanced_wealth)
        print "rebalance"
        analyzer.analyze(total_rebalanced)

        # self.data = self.data.join(all_quantities)
        # self.data = self.data.join(total_rebalanced)
        # self.data.to_csv("all_data.csv")
        print "end cash", self.cash


def fix_row_values(row_values, i, stocks_data):
    for j in range(len(row_values[0])):
        if np.isnan(row_values[0][j]) and i > 0:
            row_values[j] = stocks_data[i-1:i][j]


def main():
    start_time = datetime.datetime(1980, 10, 1)
    end_time = datetime.datetime(2016, 10, 8)

    # all_stocks = finsymbols.get_sp500_symbols()
    all_stocks = finsymbols.get_sp500_symbols()
    all_stocks.append(finsymbols.get_nasdaq_symbols())
    all_stocks.append(finsymbols.get_nyse_symbols())

    symbol_len = 200
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


    # symbols = [all_stocks[10]["symbol"], all_stocks[100]["symbol"]]
    # symbols = ['AAPL', 'IBM']

    initial_wealth = 100000
    portfolio = Portfolio(symbols, initial_wealth)

    portfolio.init_portfolio(start_time, end_time)
    portfolio.calc_wealth()

    print "the end"

if __name__ == "__main__":
    # execute only if run as a script
    main()