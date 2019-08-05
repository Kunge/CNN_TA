import tushare as ts
import logging
import argparse
import datatype
import multiprocessing

start="1990-01-01"
end="2019-07-10"

def crawl(*args):
    for code in args:
        stock_his_data = ts.get_k_data( code = code, start=start, end = end)
        stock_his_data.to_json('./'+code+'.json',orient='records')

def main(args):
    
    all_basics = ts.get_stock_basics().index
    all_num = len(all_basics)
    seg_len = 200
    a = [tuple(all_basics[x:x+seg_len]) for x in range(0,all_num,seg_len)]
    with multiprocessing.Pool(processes=10) as pool:
        pool.starmap(crawl,a)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('codes')
    #args = parser.parse_args()
    args = 'maybe used later'
    main(args)