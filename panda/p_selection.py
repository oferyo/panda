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
        logger.info("end init portfolio cash %d", self.cash)
        # print "end init portfolio cash ", self.cash

    def analyzer_buy_and_hold(self, analyzer):
        analyzer.analyze('buy_and_hold', self.total_wealth)


    @staticmethod
    def set_fractions(num_symbols):
        return np.squeeze(np.ones((1, num_symbols), float) / num_symbols)

    def calc_wealth(self, stocks_data, fees_per_share, min_fees, reb_period, partial_param, analyzer, use_fees_per_share = True, universal_1_params = None):
        name = 'fees=' + str(fees_per_share) + "RP=" + str(reb_period) + "D=" + str(partial_param) + "u_p" + str(universal_1_params)
        len_data = len(stocks_data.index)
        current_quantities = self.quantities
        rebalanced_wealth = self.wealth.copy()
        total_rebalanced = self .total_wealth.copy()
        # total_rebalanced.name = "total_rebalanced"

        for i in range(len_data):
            row_values = stocks_data[i:i+1].values
            rebalanced_wealth[i: i + 1] = current_quantities * row_values
            current_wealth = np.sum(rebalanced_wealth[i: i + 1].values) + self.cash
            if universal_1_params:
                self.update_fractions(universal_1_params, rebalanced_wealth[i: i + 1].values)

            total_rebalanced[i: i+1] = current_wealth
            updated_quantities = current_quantities
            if i % reb_period == 0:
                updated_quantities = np.floor((current_wealth * self.fractions) * np.array(1.0 / np.squeeze(row_values)))
                updated_quantities = np.floor(updated_quantities*partial_param + (1.0 - partial_param)*current_quantities)

            change = current_wealth - np.sum((updated_quantities * row_values))
            self.cash = change
            if use_fees_per_share:
                if fees_per_share > 0:
                    self.pay_fees_per_share(current_quantities, updated_quantities, fees_per_share, min_fees)
            elif fees_per_share > 0:
                self.pay_fees_per_trade(current_quantities, updated_quantities, row_values)
            current_quantities = updated_quantities

        print 'end rebalance'
        logger.info("end cash %s", self.cash)
        analyzer.analyze(name, total_rebalanced)
        # growth_result['g_r_' + name] = growth
        # annual_sharp_result['a_s_' + name] = annual_sharp

    def pay_fees_per_share(self, current_quantities, updated_quantities, fees_per_share, min_fees):
        num_shares = np.sum(np.abs(current_quantities - updated_quantities))
        if num_shares > 0:
            fees_to_pay = np.max((min_fees, num_shares*fees_per_share))
            if fees_to_pay > 0.00001:
                # print 'paid fees ', fees_to_pay,  'num_shares', num_shares, 'cash', (self.cash - fees_to_pay)
                self.cash -= fees_to_pay

    def pay_fees_per_trade(self, current_quantities, updated_quantities, row_values, fees_pre_trade = (5.0/1000.0)):
        num_shares = np.abs(current_quantities - updated_quantities)
        if np.sum(num_shares) > 0:
            sum_trade = num_shares * row_values
            fees_to_pay = np.sum(sum_trade) * fees_pre_trade
            alt_fees = np.max((1.5, np.sum(num_shares * 0.008)))
            if fees_to_pay > 0.00001:
                # if fees_to_pay > alt_fees:
                #     print 'paid fees ', fees_to_pay, 'alt_fees', alt_fees, 'num_shares', num_shares, 'cash', (self.cash - fees_to_pay)
                self.cash -= fees_to_pay

    def update_fractions(self, universal_1_param, symbols_wealth):
        current_wealth = np.sum(symbols_wealth)
        current_fractions = symbols_wealth / current_wealth
        N = len(self.fractions)
        f1 = 1.0 - universal_1_param - (universal_1_param / (N - 1))
        self.fractions = np.squeeze(current_fractions* f1 + (universal_1_param / (N - 1)))


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

    # print "len_data", len_data, "new_len_data", new_len_data, "col_len", orig_col_len, "new_col", new_col_len
    logger.info("len_data %s new_len_data %s orig_col_len %s new_col_len %s", len_data, new_len_data, orig_col_len, new_col_len)
    logger.info("final stocks are %s", stocks_data.columns)
    return stocks_data


def run_round(all_stocks, start_time, end_time, round_number):
    # symbol_len = 20
    symbol_len = 4
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
    logger.info("round symbols are %s", symbols)

    stocks_data = get_and_clean_data(symbols, start_time, end_time)
    portfolio = Portfolio(stocks_data, initial_wealth)
    analyzer = Analyzer()
    analyzer.init(stocks_data)
    portfolio.analyzer_buy_and_hold(analyzer)

    #realistic fees 0.009 per share + 1.5 min or 0.7% from trade value
    #tests was done on 0.007 good to show results this way
    use_fees_per_share = False
    fees = 0.005
    min_fees = 1.5

    logger.info('start no fees D=1')
    portfolio.calc_wealth(stocks_data, 0.0, 0.0, 1.0, 1.0, analyzer, use_fees_per_share)

    logger.info('start universal no fees D=1')
    portfolio.calc_wealth(stocks_data, 0.0, 0.0, 1.0, 1.0, analyzer, use_fees_per_share, 0.7)


    logger.info('start universal no fees D=1')
    portfolio.calc_wealth(stocks_data, 0.0, 0.0, 1.0, 1.0, analyzer, use_fees_per_share, 0.5)

    logger.info('start universal no fees D=1')
    portfolio.calc_wealth(stocks_data, 0.0, 0.0, 1.0, 1.0, analyzer, use_fees_per_share, 0.25)

    logger.info('start universal no fees D=1')
    portfolio.calc_wealth(stocks_data, 0.0, 0.0, 1.0, 1.0, analyzer, use_fees_per_share, 0.1)


    # logger.info('start with fees D=1')
    # portfolio = Portfolio(stocks_data, initial_wealth)
    # portfolio.calc_wealth(stocks_data, fees, min_fees, 1.0, 1.0, analyzer, use_fees_per_share)

    # logger.info('start with fees D=1 RP=30')
    # portfolio = Portfolio(stocks_data, initial_wealth)
    # portfolio.calc_wealth(stocks_data, fees, min_fees, 30.0, 0.01, analyzer, use_fees_per_share)


    # logger.info('start with fees D=0.01')
    # portfolio = Portfolio(stocks_data, initial_wealth)
    # portfolio.calc_wealth(stocks_data, fees, min_fees, 1.0, 0.01, analyzer, use_fees_per_share)

    num_stocks = len(stocks_data.index)
    name = "ar_num_s" + str(num_stocks) + "_" + str(round_number) + "_" + str(datetime.datetime.now().microsecond)  + ".csv"
    analyzer.to_csv(name)
    logger.info("the end")


def main():

    # cum_label = 0
    # cum_non_label = 0
    # # result = [[140308, 2417065], [316270, 1792759], [476341, 1422235], [663469, 1221244], [833018, 999542], [1055281, 841267], [1351301, 711644], [1674877, 558758], [2157544, 392063], [1833046, 140792]]
    # result = [[61261, 1037613], [136272, 766345], [206275, 609507], [285135, 523400], [358020, 428905], [453139, 360710], [578466, 304938], [716058, 240122], [922954, 169132], [783448, 60819]]
    # l1 = map(lambda x : x[0], result)
    # l2 = map(lambda x: x[1], result)
    #
    # total_lables = sum(l1)
    # total_non_labels = sum(l2)
    # print "total_labels", total_lables, " total_non_labels ", total_non_labels
    # max_diff = 0.0
    # for j in result:
    #     cum_label += j[0]
    #     cum_non_label += j[1]
    #     diff = np.abs(100.0 * float(cum_label) / total_lables - 100.0 * float(cum_non_label) / total_non_labels)
    #     print "cum_lables, " , j[0], ",", j[1], ",", cum_label, ",", cum_non_label, ",", 100.0 * float(j[0])/total_lables, ",", 100.0 * float(j[1])/total_lables, ",", 100.0 * float(cum_label) / total_lables, ",", 100.0 * float(cum_non_label) / total_non_labels
    #     if diff > max_diff:
    #         max_diff = diff

    start_time = datetime.datetime(1990, 10, 1)
    end_time = datetime.datetime(2016, 10, 8)

    # all_stocks = finsymbols.get_sp500_symbols()
    all_stocks = finsymbols.get_sp500_symbols()
    all_stocks.append(finsymbols.get_nasdaq_symbols())
    all_stocks.append(finsymbols.get_nyse_symbols())
    logger.info("starting %s ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    num_rounds = 5
    for j in range(num_rounds):
        print 'start_round\t', j
        logger.info('start round %s', j)
        run_round(all_stocks, start_time, end_time, 1)
        print 'end_round\t', j

    print "the one end"

if __name__ == "__main__":
    # execute only if run as a script
    main()