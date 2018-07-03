import tushare as ts
import logging
import argparse
import datatype

class StockSpider(object):
    def __init__(self, code, start="2008-01-01", end="2018-01-01"):
        self.code = code
        self.start = start
        self.end = end

    def crawl(self):
        stock_frame = ts.get_k_data(code=self.code, start=self.start, end=self.end, retry_count=30)
        for index in stock_frame.index:
            stock_series = stock_frame.loc[index]
            stock_dict = stock_series.to_dict()
            stock = datatype.Stock(stock_dict)
            stock.save_to_file()
        logging.warning("Finish crawling code: {}, items count: {}".format(self.code, stock_frame.shape[0]))


def main(args):
    codes = args.codes
    # codes = ['sh']
    for _code in codes:
        StockSpider(_code, args.start, args.end).crawl()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('codes')
    args = parser.parse_args()
    #main(args)