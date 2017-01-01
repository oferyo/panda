import finsymbols
import numpy as np
import pandas as pd
from pandas_datareader import data, wb
import datetime
from panda import analyzer
from panda.Logger import my_logger
from panda.analyzer import Analyzer

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

    def analyzer_buy_and_hold(self, analyzer):
        analyzer.analyze('buy_and_hold', self.total_wealth)


    @staticmethod
    def set_fractions(num_symbols):
        return np.squeeze(np.ones((1, num_symbols), float) / num_symbols)

    def calc_wealth(self, stocks_data, fees_per_share, min_fees, reb_period, partial_param, analyzer):
        name = 'fees=' + str(fees_per_share) + "RP=" + str(reb_period) + "D=" + str(partial_param)
        len_data = len(stocks_data.index)
        current_quantities = self.quantities
        rebalanced_wealth = self.wealth.copy()
        total_rebalanced = self .total_wealth.copy()
        # total_rebalanced.name = "total_rebalanced"

        for i in range(len_data):
            row_values = stocks_data[i:i+1].values
            rebalanced_wealth[i: i + 1] = current_quantities * row_values
            current_wealth = np.sum(rebalanced_wealth[i: i + 1].values) + self.cash
            total_rebalanced[i: i+1] = current_wealth
            updated_quantities = current_quantities
            if i % reb_period == 0:
                updated_quantities = np.floor((current_wealth * self.fractions) * np.array(1.0 / np.squeeze(row_values)))
                updated_quantities = np.floor(updated_quantities*partial_param + (1.0 - partial_param)*current_quantities)

            change = current_wealth - np.sum((updated_quantities * row_values))
            self.cash = change
            # self.pay_fees_pre_share(current_quantities, updated_quantities, fees_per_share, min_fees)
            self.pay_fees_per_trade(current_quantities, updated_quantities, row_values)
            current_quantities = updated_quantities

        print 'end rebalance'
        print "end cash", self.cash
        analyzer.analyze(name, total_rebalanced)
        # growth_result['g_r_' + name] = growth
        # annual_sharp_result['a_s_' + name] = annual_sharp

    def pay_fees_pre_share(self, current_quantities, updated_quantities, fees_per_share, min_fees):
        num_shares = np.sum(np.abs(current_quantities - updated_quantities))
        if num_shares > 0:
            fees_to_pay = np.max((min_fees, num_shares*fees_per_share))
            if fees_to_pay > 0.00001:
                # print 'paid fees ', fees_to_pay,  'num_shares', num_shares, 'cash', (self.cash - fees_to_pay)
                self.cash -= fees_to_pay

    def pay_fees_per_trade(self, current_quantities, updated_quantities, row_values, fees_pre_trade = (0.6/100.0)):
        num_shares = np.abs(current_quantities - updated_quantities)
        if np.sum(num_shares) > 0:
            sum_trade = num_shares * row_values
            fees_to_pay = np.sum(sum_trade) * fees_pre_trade
            alt_fees = np.max((1.5, np.sum(num_shares * 0.008)))
            if fees_to_pay > 0.00001:
                # if fees_to_pay > alt_fees:
                #     print 'paid fees ', fees_to_pay, 'alt_fees', alt_fees, 'num_shares', num_shares, 'cash', (self.cash - fees_to_pay)
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


def run_round(all_stocks, start_time, end_time, round_number):
    # symbol_len = 20
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

    initial_wealth = 100000
    stocks_data = get_and_clean_data(symbols, start_time, end_time)
    portfolio = Portfolio(stocks_data, initial_wealth)
    analyzer = Analyzer()
    analyzer.init(stocks_data)
    portfolio.analyzer_buy_and_hold(analyzer)

    #realistic fees 0.009 per share + 1.5 min or 0.7% from trade value

    fees = 0.012
    min_fees = 1.5

    logger.info('start no fees D=1')
    portfolio.calc_wealth(stocks_data, 0.0, 0.0, 1.0, 1.0, analyzer)

    logger.info('start with fees D=1')
    portfolio = Portfolio(stocks_data, initial_wealth)
    portfolio.calc_wealth(stocks_data, fees, min_fees, 1.0, 1.0, analyzer)

    # logger.info('start with fees D=0.5')
    # portfolio = Portfolio(stocks_data, initial_wealth)
    # portfolio.calc_wealth(stocks_data, fees, min_fees, 1.0, 0.5, analyzer)


    logger.info('start with fees D=1 RP=30')
    portfolio = Portfolio(stocks_data, initial_wealth)
    portfolio.calc_wealth(stocks_data, fees, min_fees, 30.0, 0.01, analyzer)


    logger.info('start with fees D=0.01')
    portfolio = Portfolio(stocks_data, initial_wealth)
    portfolio.calc_wealth(stocks_data, fees, min_fees, 1.0, 0.01, analyzer)

    name = "ar_" + str(round_number) + "_" + str(datetime.datetime.now().microsecond)  + ".csv"
    analyzer.to_csv(name)
    logger.info("the end")


def main():
    start_time = datetime.datetime(1980, 10, 1)
    end_time = datetime.datetime(2016, 10, 8)

    # all_stocks = finsymbols.get_sp500_symbols()
    all_stocks = finsymbols.get_sp500_symbols()
    all_stocks.append(finsymbols.get_nasdaq_symbols())
    all_stocks.append(finsymbols.get_nyse_symbols())


    for j in range(5):
        print 'start_round\t', j
        logger.info('start round %s', j)
        run_round(all_stocks, start_time, end_time, 1)
        print 'end_round\t', j

    print "the one end"

if __name__ == "__main__":
    # execute only if run as a script
    main()