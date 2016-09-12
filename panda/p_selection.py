import finsymbols
import numpy as np
from pandas_datareader import data, wb
import datetime

from panda import analyzer


class Portfolio:

    def __init__(self, stocks_data, initial_wealth):
        num_symbols = stocks_data.columns.size
        self.fractions = self.set_fractions(num_symbols)
        self.initial_wealth = initial_wealth
        quantities = (initial_wealth * self.fractions) * np.array(1.0 / np.squeeze(stocks_data[0:1].values))
        self.quantities = np.floor(quantities)
        self.cash = initial_wealth - np.dot(stocks_data[0:1], self.quantities)

        self.wealth = self.quantities * stocks_data[0:]
        self.total_wealth = self.wealth.apply(lambda x : np.sum(x), axis = 1, raw=False)
        self.total_wealth.name = 'total_wealth  '
        analyzer.analyze(self.total_wealth)
        print "end init portfolio cash ", self.cash

    @staticmethod
    def set_fractions(num_symbols):
        return np.squeeze(np.ones((1, num_symbols), float) / num_symbols)

    def calc_wealth(self, stocks_data):
        fees = 0.005
        min_fees = 1
        current_wealth = self.initial_wealth
        len_data = len(stocks_data.index)
        current_quantities = self.quantities
        all_quantities = self.wealth.copy()
        rebalanced_wealth = self.wealth.copy()
        total_rebalanced = self .total_wealth.copy()
        total_rebalanced.name = "total_rebalanced"

        for i in range(len_data):
            row_values = stocks_data[i:i+1].values
            rebalanced_wealth[i: i + 1] = current_quantities * row_values
            current_wealth = np.sum(rebalanced_wealth[i: i + 1].values) + self.cash
            total_rebalanced[i : i+1] = current_wealth
            current_quantities = np.floor((current_wealth * self.fractions) * np.array(1.0 / np.squeeze(row_values)))
            all_quantities[i : i+1] = [current_quantities]
            change = current_wealth - np.sum((current_quantities * row_values))
            self.cash = change

        print 'end rebalance'
        analyzer.analyze(total_rebalanced)

        # self.data = self.data.join(all_quantities)
        # self.data = self.data.join(total_rebalanced)
        # self.data.to_csv("all_data.csv")
        print "end cash", self.cash


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


def main():
    start_time = datetime.datetime(1980, 10, 1)
    end_time = datetime.datetime(2016, 10, 8)

    # all_stocks = finsymbols.get_sp500_symbols()
    all_stocks = finsymbols.get_sp500_symbols()
    all_stocks.append(finsymbols.get_nasdaq_symbols())
    all_stocks.append(finsymbols.get_nyse_symbols())

    symbol_len = 30
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
    portfolio.calc_wealth(stocks_data)

    print "the end"

if __name__ == "__main__":
    # execute only if run as a script
    main()