import pandas as pd
import numpy as np
from pandas_datareader import data, wb
import datetime


class Portfolio:

    def __init__(self, symbols, initial_wealth, fractions = None):
        self.symbols = symbols
        self.initial_wealth = initial_wealth
        self.fractions = fractions
        if fractions == None:
            self.fractions = np.squeeze(np.ones((1, len(symbols)), float) / len(symbols))


    def init_portfolio(self, start_time, end_time):
        self.stocks_data = stocks_data = data.get_data_yahoo(self.symbols, start=start_time, end=end_time)['Adj Close']
        # clean data
        # quantitis = np.dot((self.initial_wealth * self.fractions), 1.0 / np.squeeze(stocks_data[0:1].values))
        quantitis = (self.initial_wealth * self.fractions) * np.array(1.0 / np.squeeze(stocks_data[0:1].values))
        self.quantitis = np.floor(quantitis)
        self.cash = self.initial_wealth - np.dot(stocks_data[0:1], self.quantitis)

        self.wealth = self.quantitis * stocks_data[0:]
        self.wealth.to_csv("wealth.csv")

        # self.wealth.to_csv("sd.csv")
        self.total_wealth = self.wealth.apply(lambda x : np.sum(x), axis = 1, raw=False)
        self.total_wealth.to_csv("tw.csv")
        print "end"

    def calc_wealth(self):
        fees = 0.005
        min_fees = 1
        current_wealth = self.initial_wealth
        current_quantitis = self.quantitis
        index = 0

        for i in range(len(self.stocks_data.index)):
            row_values = self.stocks_data[i:i+1].values
            self.wealth[i: i + 1] = current_quantitis * row_values
            # current_wealth =  self.wealth[i: i + 1].apply(lambda x : np.sum(x), axis = 1, raw=False)
            current_wealth = np.sum(self.wealth[i: i + 1].values)
            print "cw ", current_wealth
            current_quantitis = np.floor((current_wealth * self.fractions) * np.array(1.0 / np.squeeze(row_values)))
            print "cw ", current_wealth

        print "end"



def main():
    xx = np.random.random((3,3))
    yy= np.max(xx, axis=0) + 1

    symbols = ['AAPL', 'IBM']
    initial_wealth = 100000
    start_time = datetime.datetime(2012, 10, 1)
    end_time = datetime.datetime(2012, 10, 8)
    portfolio = Portfolio(symbols, initial_wealth)
    portfolio.init_portfolio(start_time, end_time)
    portfolio.calc_wealth()

    aapl = pd.io.data.get_data_yahoo(['AAPL', 'IBM'],
                                     start=datetime.datetime(1999, 10, 1),
                                     end=datetime.datetime(2012, 10, 3))

    xx = aapl.head()

    yy = aapl.head(-10)
    print "applhead", xx
    print "appltail", yy
    print  "end"

if __name__ == "__main__":
    # execute only if run as a script
    main()