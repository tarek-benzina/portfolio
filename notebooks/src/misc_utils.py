import numpy as np
import pandas as pd
import holidays

def rmse(data,pred_col,actual_col):
    return np.sqrt(np.mean((data[pred_col]-data[actual_col])**2))

def mape(data,pred_col,actual_col):
    return 100*np.sum(np.abs(data[pred_col]-data[actual_col]))/np.sum(np.abs(data[actual_col]))

def add_date_int_index(calendar):
    dates=pd.DataFrame(sorted(calendar[(calendar.working_day==1)].date.unique()),columns=['date'])
    dates["i"]=range(0,len(dates))
    return calendar.merge(dates,on="date",how="left")

def build_calendar(start_date,end_date,forecast_horizon,forecast_steps=1,market="NYSE"):
    end_date=end_date+pd.Timedelta(days=forecast_horizon+forecast_steps)
    dates=pd.date_range(start_date,end_date)
    market_holidays = holidays.financial_holidays(market)
    calendar=pd.DataFrame(dates,columns=["date"])
    calendar["holiday"]=calendar["date"].apply(lambda x:market_holidays.get(x) if market_holidays.get(x) else "no_holiday")
    calendar["dow"]=calendar.date.dt.weekday
    calendar["weekend"]=calendar.dow.apply(lambda x: 1 if x in [5,6] else 0)
    calendar["woy"]=calendar.date.dt.isocalendar().week
    calendar["working_day"]=0
    calendar.loc[(calendar.holiday=="no_holiday")&(calendar.weekend==0),"working_day"]=1
    return add_date_int_index(calendar)

def get_train_val_test(calendar,end_date,horizon,test_ratio=.2,val_ratio=.1):
    calendar["train"]=0
    calendar["test"]=0
    calendar["val"]=0
    size=calendar[(calendar.date<=end_date)].i.max()-horizon
    calendar.loc[(calendar.i<=size*(1-val_ratio-test_ratio)),"train"]=1
    calendar.loc[(calendar.train==0)&(calendar.i<=size*(1-test_ratio)),"val"]=1
    calendar.loc[(calendar.train==0)&(calendar.val==0)&(calendar.i<=size),"test"]=1
    return calendar

def split_train_val_test(data_subset):
    train_df=data_subset[(data_subset["backtesting_mode"]=="train")].copy()
    train_end_date=train_df.date.max()
    test_df=data_subset[(data_subset["backtesting_mode"]=="test")].copy()
    val_df=train_df.sample(frac=.3)
    train_df=train_df.drop(index=val_df.index)
    return {"train":train_df,"val":val_df,"test":test_df,"train_end_date":train_end_date}

def shift(subset,lag=1,column='Close'):
    subset_delayed=subset[["ticker",column,"date","i"]].copy()
    subset_delayed["i"]=subset_delayed.i-lag
    subset_delayed=subset_delayed.rename(columns={column:f"{column}_shift_{lag}","date":f"lag_date_{column}_{lag}"})
    subset=subset.merge(subset_delayed.dropna(subset=["ticker","i"]),on=["i","ticker"],how="left")
    return subset 

def build_grid(data,calendar):
    stock_index=data.groupby(['ticker'],as_index=False).date.count().drop(columns=["date"])
    stock_index["k"]=1
    calendar["k"]=1
    grid=stock_index.merge(calendar,on="k")
    return grid.drop(columns=["k"])

def count_nan(data):
    nan_count={}
    for col in data.columns:
        nan_count[col]=len(data[data[col].isnull()])
    return nan_count