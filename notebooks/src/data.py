from datetime import datetime
import pandas as pd
import logging
import os
from yahooquery import Ticker

def get_details(tickers,remove_ticker_list):
    t = Ticker(tickers,asynchronous=True)
    res_prof=t.summary_profile
    res_detail=t.quotes
    
    dfs=[]
    for x in tickers:
        if x not in remove_ticker_list:
            if isinstance(res_prof[x],dict) and isinstance(res_detail[x],dict):
                try:
                    tmp=pd.DataFrame.from_dict(res_prof[x]|res_detail[x],orient="index").transpose()
                    tmp["ticker"]=x
                    dfs.append(tmp)
                except:
                    continue
    return pd.concat(dfs,ignore_index=True)

def load_or_download(start_date,data_dir,yesterday):
    if os.path.exists(f"{data_dir}/data_{yesterday}.csv"):
        logging.info(f"Loading Yesterday's data from {data_dir}/data_{yesterday}.csv")
        data=pd.read_csv(f"{data_dir}/data_{yesterday}.csv")
        data=data[data.date>=start_date]
    else:
        logging.info(f"Yesterday's data doesn't exist. Downloading...")
        tickers=pd.read_csv(f"{data_dir}/tickers.csv").ticker.values.tolist()
        details=get_details(tickers,remove_ticker_list=["PPCB"])
        details=details[["ticker",
                        "industry",
                        "sector",
                        "country",
                        "fullExchangeName",
                        "quoteType",
                        "fiftyTwoWeekLow",
                        "fiftyTwoWeekHigh",
                        "marketCap",
                        "exchange",
                        "epsTrailingTwelveMonths",
                        "averageDailyVolume3Month",
                        "epsForward",
                        "forwardPE"]]
        details=details[details.quoteType=="EQUITY"]
        details=details[~details.country.isnull()]        
        filtered_details=details[details.marketCap>details.marketCap.quantile(.95)]
        t=Ticker(filtered_details.ticker,asynchronous=True)
        data_init=t.history(start=start_date,adj_ohlc=True,adj_timezone=False)
        data=data_init.reset_index()
        data=data.rename(columns={"symbol":"ticker","close":"Close"})
        data=data.merge(filtered_details.dropna(),on="ticker",how="inner")
        logging.info(f"Storing the data to {data_dir}/data_{yesterday}.csv")
        data.to_csv(f"{data_dir}/data_{yesterday}.csv",index=False)
    return data

def update(conn):
    date=pd.read_sql("SELECT max(date) FROM stock_price",con=conn)["max(date)"].values[0]
    max_date=pd.to_datetime(date)
    diff=(datetime.today()-max_date).days
    start_date=(max_date+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if diff>0:
        tickers=list(pd.read_sql(f"SELECT DISTINCT ticker FROM stock_price",con=conn).ticker.unique())
        logging.info(f"Updating data from {start_date} for the following tickers: {tickers}")
        t=Ticker(tickers,asynchronous=True)
        data_init=t.history(start=start_date,adj_ohlc=True,adj_timezone=False)
        data=data_init.reset_index()
        data=data.rename(columns={"symbol":"ticker","close":"Close"})
        data["date"]=pd.to_datetime(data.date.dt.date)
        data.to_sql("stock_price",con=conn,if_exists="append",index=False)
