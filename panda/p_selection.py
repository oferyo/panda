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

    def analyzer_buy_and_hold(self, result):
        analyzer.analyze('buy_and_hold', self.total_wealth, result)


    @staticmethod
    def set_fractions(num_symbols):
        return np.squeeze(np.ones((1, num_symbols), float) / num_symbols)

    def calc_wealth(self, stocks_data, fees_per_share, min_fees, reb_period, partial_param, growth_result = None, annual_sharp_result = None):
        name = 're_' + str(fees_per_share) + "_" + str(min_fees) + "_" + str(reb_period) + "_" + str(partial_param)
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
            updated_quantities = np.floor((current_wealth * self.fractions) * np.array(1.0 / np.squeeze(row_values)))
            updated_quantities = np.floor(updated_quantities*partial_param + (1.0 - partial_param)*current_quantities)
            change = current_wealth - np.sum((updated_quantities * row_values))
            self.cash = change
            self.pay_fees(current_quantities, updated_quantities, fees_per_share, min_fees)
            current_quantities = updated_quantities

        print 'end rebalance'
        print "end cash", self.cash
        growth, annual_sharp = analyzer.analyze(name, total_rebalanced)
        growth_result['g_r_' + name] = growth
        annual_sharp_result['a_s_' + name] = annual_sharp

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
    analyzer_result = pd.DataFrame(index=stocks_data.index)
    portfolio = Portfolio(stocks_data, initial_wealth)

    portfolio.analyzer_buy_and_hold(analyzer_result)
    fees = 0.01

    logger.info('start no fees D=1')
    portfolio.calc_wealth(stocks_data, 0.0, 0.0, 1.0, 1.0, analyzer_result)

    logger.info('start with fees D=1')
    portfolio = Portfolio(stocks_data, initial_wealth)
    portfolio.calc_wealth(stocks_data, fees, 1.0, 1.0, 1.0, analyzer_result)

    logger.info('start with fees D=0.5')
    portfolio = Portfolio(stocks_data, initial_wealth)
    portfolio.calc_wealth(stocks_data, fees, 1.0, 1.0, 0.5, analyzer_result)

    logger.info('start with fees D=0.01')
    portfolio = Portfolio(stocks_data, initial_wealth)
    portfolio.calc_wealth(stocks_data, fees, 1.0, 1.0, 0.01, analyzer_result)


    logger.info("the end")


def log_loss(label, pred):
    return -np.mean(label*np.log(pred[:,0]) + (1 - label) * np.log(1.0 - pred[:,0]))


def main():
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


def random_walk_exp():
    n = 10
    rounds = 10000000
    score = 0.0
    actual_rnd = 0
    for j in range(rounds):
        k = random_walk(n)
        if k >= 0:
            actual_rnd+=1
            score+=k

    final = score/actual_rnd
    print final



def random_walk(n):
    score = 0
    pure_pos = True
    for i in range(n):
        score = (score + 1) if np.random.random() >=0.5 else (score-1)
        if score < 0:
            pure_pos = False
            break

    if pure_pos:
        return score
    else:
        return -1

def random_walk_non_neg(n):
    score = 0
    for i in range(n):
        if score == 0:
            score = 1
        else:
            score = (score + 1) if np.random.random() >=0.5 else (score-1)

    return score




def gamadim_check():
    n = 8
    success = 0
    for i in xrange(1000):
        a = np.random.rand(n)
        b = np.random.rand(n)
        h = np.random.rand(1) * (n - 2)
        mo = -1
        po = -1
        for p in itertools.permutations(range(n)):
            m = run(a, b, h, p)
            if (m > mo):
                mo = m
                po = p

        pa = algo(a, b, h)
        ma = run(a, b, h, pa)
        if (mo != ma):
            print (mo)
            print (ma)
            pa = algo(a, b, h)
            ma = run(a, b, h, pa)
            print (po)
            print (pa)
            print (a)
            print (b)
            print (np.argsort(a - b))
            print (np.argsort(a))
            print (np.argsort(b))
            print (h)
        else:
            success+=1

    print "success pcg " , success*100/1000

def algo(a, b, h):
    p = np.argsort(a)
    A = sum(a)
    out = []
    stuck = []
    for i in p:
        if A + b[i] > h:
            A -= a[i]
            out.append(i)
        else:
            # check if all including the failed one can pass
            all_can_pass = True
            B = A - a[i]
            alt_out = []
            check_if_all = list(out)
            check_if_all.append(i)
            while all_can_pass and len(check_if_all) > 0:
                # find the tallest that can pss now
                can_pass = filter(lambda j: B + a[j] + b[j] > h, check_if_all)
                if can_pass:
                    tallest = np.max((a[can_pass]))
                    index_tallest = list(a).index(tallest)
                    check_if_all.remove(index_tallest)
                    B += tallest
                    alt_out.insert(0, index_tallest)
                else:  # if not all can pass append the tallest to the stuck
                    all_can_pass = False
                    stuck.append(i)
            if all_can_pass:
                A -= a[i]
                out = list(alt_out)

    return out + stuck

import itertools

def run(a, b, h, p):
    c = 0
    A = sum(a)
    for i in p:
        if (A + b[i] > h):
            c = c + 1
            A = A - a[i]
        else:
            break;
    return c

if __name__ == "__main__":
    # execute only if run as a script
    main()