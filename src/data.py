from datetime import datetime, timedelta
import pandas as pd
import logging
import os
from yahooquery import Ticker
import numpy as np
from .misc_utils import create_connection,build_calendar,get_train_val_test,shift,build_grid


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


def load_hist_earnings(tickers):
    t=Ticker(tickers,asynchronous=True)
    earnings=t.earning_history.reset_index()
    earnings=earnings.rename(columns={"symbol":"ticker"}).drop(columns=["row","maxAge"])
    earnings["ticker"]=earnings["ticker"].apply(lambda x:str(x) if not isinstance(x,dict) else np.nan)
    earnings["epsActual"]=earnings["epsActual"].apply(lambda x:float(x) if not isinstance(x,dict) else np.nan)
    earnings["epsEstimate"]=earnings["epsEstimate"].apply(lambda x:float(x) if not isinstance(x,dict) else np.nan)
    earnings["epsDifference"]=earnings["epsDifference"].apply(lambda x:float(x) if not isinstance(x,dict) else np.nan)
    earnings["surprisePercent"]=earnings["surprisePercent"].apply(lambda x:float(x) if not isinstance(x,dict) else np.nan)
    earnings["quarter"]=earnings["quarter"].apply(lambda x:pd.to_datetime(x) if not isinstance(x,dict) else np.nan)
    earnings["period"]=earnings["period"].apply(lambda x:str(x) if not isinstance(x,dict) else np.nan)
    return earnings


def load_future_earnings(tickers):
    t=Ticker(tickers,asynchronous=True)
    res=t.earnings
    ticker_list=[]
    quarter_estimate=[]
    quarter_estimate_date=[]
    quarter_estimate_year=[]
    earnings_date_start=[]
    earnings_date_end=[]
    for ticker in res.keys():
        if isinstance(res[ticker],dict):
            ticker_list.append(ticker)
            if "currentQuarterEstimate" in res[ticker]['earningsChart']:
                quarter_estimate.append(res[ticker]['earningsChart']['currentQuarterEstimate'])
            else:
                quarter_estimate.append(np.nan)
                
            if "currentQuarterEstimateDate" in res[ticker]['earningsChart']:
                quarter_estimate_date.append(res[ticker]['earningsChart']['currentQuarterEstimateDate'])
            else:
                quarter_estimate_date.append(np.nan)
                
            if "currentQuarterEstimateYear" in res[ticker]['earningsChart']:
                quarter_estimate_year.append(res[ticker]['earningsChart']['currentQuarterEstimateYear'])
            else:
                quarter_estimate_year.append(np.nan)
        
                
            if len(res[ticker]['earningsChart']['earningsDate'])==2:
                earnings_date_start.append(res[ticker]['earningsChart']['earningsDate'][0])
                earnings_date_end.append(res[ticker]['earningsChart']['earningsDate'][1])
            elif len(res[ticker]['earningsChart']['earningsDate'])==1:
                earnings_date_start.append(res[ticker]['earningsChart']['earningsDate'][0])
                earnings_date_end.append(res[ticker]['earningsChart']['earningsDate'][0])
            elif len(res[ticker]['earningsChart']['earningsDate'])==0:
                earnings_date_start.append(np.nan)
                earnings_date_end.append(np.nan)
            else:
                print(f"Unable to process ticker {ticker}")
    future_earnings=pd.DataFrame(
        zip(ticker_list,quarter_estimate,quarter_estimate_date,quarter_estimate_year,earnings_date_start,earnings_date_end),
        columns=["ticker","epsEstimate","quarter_period","quarter_year","date_start","date_end"]
    )
    return future_earnings

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
        data=data[data.date.dt.hour==0] ## don't load in-market-hours data
        data["date"]=pd.to_datetime(data.date.dt.date)
        data=data[data["date"]>=start_date]
        data.to_sql("stock_price",con=conn,if_exists="append",index=False)

def prepare_data(last_date,db_path,price_column="Close",lookback=365,horizon=7,forecast_steps=1,update_data=False,test_ratio=.1,val_ratio=0):
    conn=create_connection(db_path)
    initial_date=(last_date-timedelta(days=lookback)).strftime("%Y-%m-%d")
    if update_data:
        update(conn=conn)
    data=pd.read_sql(f"SELECT * FROM stock_price WHERE date>='{initial_date} 00:00:00' AND date <= '{last_date.strftime('%Y-%m-%d')} 00:00:00'",con=conn)
    data=data.groupby(["ticker","date"],as_index=False).mean()
    profile=pd.read_sql(f"SELECT ticker,industry FROM profile",con=conn)
    data=data.merge(profile,on="ticker",how="left")
    data=data[["date","ticker","industry","Close"]]
    data["date"]=pd.to_datetime(data["date"])
    calendar=build_calendar(start_date=data.date.min(),end_date=data.date.max(),forecast_horizon=5*horizon,forecast_steps=forecast_steps)
    calendar=get_train_val_test(calendar=calendar,end_date=data.date.max(),horizon=horizon,test_ratio=test_ratio,val_ratio=val_ratio)
    grid=build_grid(data,calendar)
    data=data.merge(grid,on=["date","ticker"],how="right")
    return_cols=[]
    if horizon>1:
        for i in range(1,horizon+1):
            data=shift(data,lag=i,column=price_column)
            return_cols.append(f"log_return_{i}_shift_-{i}")
            data[f"log_return_{i}"]=np.log(data[f"{price_column}_shift_{i}"]/data[price_column])
            data=shift(data,lag=-i,column=f"log_return_{i}")
    return data
