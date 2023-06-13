import tensorflow as tf
import pandas as pd
import numpy as np
import pandas as pd
from notebooks.src.misc_utils import *
from notebooks.src.model_utils import *
from notebooks.src.sequence_utils import *
from notebooks.src.portfolio import *
from notebooks.src.data import update
from datetime import datetime,timedelta
import logging

DATA_DIR="notebooks/"
MODELS_DIR="notebooks/"
HORIZON=7
PRICE_COLUMN="Close"
DB_PATH="stock_db"
HISTORY=14
FORECAST_STEPS=1
FUTURE=0
TARGET=f"log_return_{HORIZON}"
LOOKBACK=365


if __name__=="__main__":
    logging_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    conn=create_connection(DB_PATH)
    today=datetime.today().strftime("%Y-%m-%d")
    yesterday=(datetime.today()-timedelta(days=1)).strftime("%Y-%m-%d")
    initial_date=(datetime.today()-timedelta(days=LOOKBACK)).strftime("%Y-%m-%d")
    logging.info(f"Today: {today}")
    logging.info(f"Yesterday: {yesterday}")
    logging.info(f"Data Start Date: {initial_date}")
    update(conn=conn) ### UPDATE Price Data
    data=pd.read_sql(f"SELECT * FROM stock_price WHERE date>='{initial_date} 00:00:00'",con=conn)
    data=data.groupby(["ticker","date"],as_index=False).mean()
    profile=pd.read_sql(f"SELECT ticker,industry FROM profile",con=conn)
    data=data.merge(profile,on="ticker",how="left")
    data=data[["date","ticker","industry","Close"]]
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
            return_cols.append(f"log_return_{i}_shift_-{i}")
            data[f"log_return_{i}"]=np.log(data[f"{PRICE_COLUMN}_shift_{i}"]/data[PRICE_COLUMN])
            data=shift(data,lag=-i,column=f"log_return_{i}")
    norm=Normaliser()
    norm.load(f"{MODELS_DIR}/normaliser")
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

    
    model_path=get_latest_model_path(models_dir=MODELS_DIR,prefix="model-")            
    model = tf.keras.models.load_model(model_path, custom_objects={'same_sign':same_sign})

    history=model.fit(
        train_dict["X"],
        train_dict["Y"],
        validation_split=.2,
        epochs=1000,
        batch_size=30,
        verbose=1,
        callbacks=[early_stopping])
    model.save(f"{MODELS_DIR}/model-{today}")
