import tushare as ts
import logging
import argparse
import datatype

class StockSpider(object):
    def __init__(self, code, start="2000-01-01", end="2018-07-01"):
        self._code = code
        self._start = start
        self._end = end

    def crawl(self, code):
        stock_his_data = ts.get_k_data( code = code, start=self._start, end = self._end)
        stock_his_data.to_json('./'+self._code+'.json',orient='records')

def main(args):
    
    all_basics = ts.get_stock_basics()
    for code in all_basics.index:
        StockSpider( code ).crawl( code )


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('codes')
    #args = parser.parse_args()
    args = 'maybe used later'
    main(args)