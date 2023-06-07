import tensorflow as tf
import pandas as pd
import numpy as np
import pandas as pd
from notebooks.src.misc_utils import *
from notebooks.src.model_utils import *
from notebooks.src.sequence_utils import *
from notebooks.src.portfolio import *
from yahooquery import Ticker
from datetime import datetime,timedelta
import os
import logging

DATA_DIR="notebooks/"
MODELS_DIR="notebooks/"
HORIZON=7
PRICE_COLUMN="Close"

HISTORY=14
FORECAST_STEPS=1
FUTURE=0
TARGET=f"log_return_{HORIZON}"

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

if __name__=="__main__":
    logging_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    today=datetime.today().strftime("%Y-%m-%d")
    yesterday=(datetime.today()-timedelta(days=1)).strftime("%Y-%m-%d")
    logging.info(f"Today: {today}")
    logging.info(f"Yesterday: {yesterday}")
    if os.path.exists(f"{DATA_DIR}/data_{yesterday}.csv"):
        logging.info(f"Loading Yesterday's data from {DATA_DIR}/data_{yesterday}.csv")
        data=pd.read_csv(f"{DATA_DIR}/data_{yesterday}.csv")
    else:
        tickers=pd.read_csv(f"{DATA_DIR}/tickers.csv").ticker.values.tolist()
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
        data_init=t.history(start="2021-01-01",adj_ohlc=True,adj_timezone=False)
        data=data_init.reset_index()
        data=data.rename(columns={"symbol":"ticker","close":"Close"})
        data=data.merge(filtered_details.dropna(),on="ticker",how="inner")
        data.to_csv(f"{DATA_DIR}/data_{yesterday}.csv",index=False)
    data=data[["date","ticker","industry","fullExchangeName","Close","marketCap","fiftyTwoWeekLow","fiftyTwoWeekHigh","averageDailyVolume3Month","epsTrailingTwelveMonths",
                    "averageDailyVolume3Month",
                    "epsForward",
                    "forwardPE"]]
    data["date"]=pd.to_datetime(data["date"])
    start_date=data.date.min()
    end_date=data.date.max()

    calendar=build_calendar(start_date=start_date,end_date=end_date,forecast_horizon=5*HORIZON,forecast_steps=FORECAST_STEPS)
    calendar=get_train_val_test(calendar=calendar,end_date=end_date,horizon=HORIZON,test_ratio=.0,val_ratio=.0)
    grid=build_grid(data,calendar)
    data=data.merge(grid,on=["date","ticker"],how="right")
    return_cols=[]
    if HORIZON>1:
        for i in range(1,HORIZON+1):
            data=shift(data,lag=i,column=PRICE_COLUMN)
            return_cols.append(f"log_return_{i}_shift_-7")
            data[f"log_return_{i}"]=np.log(data[f"{PRICE_COLUMN}_shift_{i}"]/data[PRICE_COLUMN])
            data=shift(data,lag=-HORIZON,column=f"log_return_{i}")
    norm=Normaliser(target="Close",timeseries_id_column="ticker")
    norm.fit(data[data.train==1])
    data=norm.normalise(data)
    num_features=["Close_scaled"]+return_cols
    initial_available_date=data[~data[TARGET].isnull()].date.min()
    enc=Encoder()
    enc.load(f"{MODELS_DIR}/encoder")
    features=enc.get_encoded_features()
    features=features+num_features
    data=enc.apply(data)
    inference_date=data[(data.weekend==0)&(data.holiday=="no_holiday")&(data.date>data[~data[TARGET].isnull()][f"lag_date_Close_{HORIZON}"].max())].date.min()
    sequencer=TimeseriesSquencer(
        features=features,
        target=TARGET,
        history=HISTORY,
        future=FUTURE,
        steps=FORECAST_STEPS,
        id_column="ticker",
        time_column="date",
        data_end_date=inference_date.strftime("%Y-%m-%d"),
        include_target_feature=False,
        extra_id_columns=[])
    train_dict=sequencer.sequence_timeseries(data[data.train==1].dropna())
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                patience=10,
                                                  restore_best_weights=True,
                                                mode='min') 
    def same_sign(y_true,y_pred):
        return tf.reduce_sum(tf.abs(tf.cast(tf.sign(y_true),dtype=tf.float64)+tf.cast(tf.sign(y_pred),dtype=tf.float64))/2)/tf.reduce_sum(tf.abs(tf.cast(tf.sign(y_true),dtype=tf.float64)))                          
    if os.path.exists(f"{MODELS_DIR}/model-{yesterday}"):                
        model = tf.keras.models.load_model(f"{MODELS_DIR}/model-{yesterday}", custom_objects={'same_sign':same_sign})
    else:
        raise ValueError(f"no model found under {MODELS_DIR}/model-{yesterday}")
    history=model.fit(
        train_dict["X"],
        train_dict["Y"],
        validation_split=.2,
        epochs=1000,
        batch_size=30,
        verbose=1,
        callbacks=[early_stopping])
    model.save(f"model-{today}")
