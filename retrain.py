import tensorflow as tf
from src.misc_utils import *
from src.model_utils import *
from src.sequence_utils import *
from src.portfolio import *
from src.data import prepare_data
from datetime import datetime
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
LOOKBACK=180


if __name__=="__main__":
    logging_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    conn=create_connection(DB_PATH)
    today=datetime.today().strftime("%Y-%m-%d")

    data=prepare_data(last_date=datetime.today(),
                    db_path=DB_PATH,
                    price_column=PRICE_COLUMN,
                    lookback=LOOKBACK,
                    horizon=HORIZON,
                    forecast_steps=FORECAST_STEPS,
                    update_data=True,
                    test_ratio=0,
                    val_ratio=0)
    model_path=get_latest_model_path(models_dir=MODELS_DIR,prefix="model-") 
    logging.info(f"Today: {today}")
    norm=Normaliser()
    norm.load(f"{model_path}/normaliser")
    data=norm.normalise(data)
    num_features=["Close_scaled",
              "log_return_1_shift_-1",
              "log_return_2_shift_-2",
              "log_return_3_shift_-3",
              "log_return_4_shift_-4",
              "log_return_5_shift_-5",
              "log_return_6_shift_-6",
              "log_return_7_shift_-7",
    ]
    
    enc=Encoder()
    enc.load(f"{model_path}/encoder")
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
    norm.save(f"model-{inference_date.strftime('%Y-%m-%d')}/normaliser")
    enc.save(f"model-{inference_date.strftime('%Y-%m-%d')}/encoder")
