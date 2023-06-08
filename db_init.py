import pandas as pd
import numpy as np

import sqlite3
from sqlite3 import Error

from yahooquery import Ticker


DB_NAME="stock_db"
DATA_DIR="/home/notebooks/"
START_DATE="2021-01-01"

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def get_profiles(tickers):
    t = Ticker(tickers,asynchronous=True)
    res_prof=t.summary_profile    
    dfs=[]
    for x in tickers:
        if isinstance(res_prof[x],dict):
            try:
                tmp=pd.DataFrame.from_dict(res_prof[x],orient="index").transpose()
                tmp["ticker"]=x
                dfs.append(tmp)
            except:
                continue
    return pd.concat(dfs,ignore_index=True)

def get_details(tickers):
    t = Ticker(tickers,asynchronous=True)
    res_prof=t.price  
    dfs=[]
    for x in tickers:
        if isinstance(res_prof[x],dict):
            try:
                tmp=pd.DataFrame.from_dict(res_prof[x],orient="index").transpose()
                tmp["ticker"]=x
                dfs.append(tmp)
            except:
                continue
    return pd.concat(dfs,ignore_index=True)




if __name__=="__main__":
    tickers=pd.read_csv(f"{DATA_DIR}/tickers.csv").ticker.values.tolist()
    conn=create_connection(DB_NAME)
    profiles=get_profiles(tickers)
    profiles["companyOfficers"]=profiles["companyOfficers"].astype(str)
    details=get_details(tickers)
    profiles=profiles.merge(details[["ticker","quoteType","exchange","exchangeName","marketState","marketCap"]],on="ticker",how="left")
    profiles["marketCap"]=profiles.marketCap.apply(lambda x:float(x) if not isinstance(x,dict) else np.nan)
    profiles.to_sql(name="profile",con=conn,if_exists="append",index=False)

    modelled_tickers=profiles[(profiles.quoteType=="EQUITY")&(profiles.exchangeName=="NasdaqGS")]
    modelled_tickers[(modelled_tickers.marketCap>modelled_tickers.marketCap.quantile(.8))][["ticker"]].to_sql(name="modelled_tickers",con=conn,if_exists="append",index=False)

    ##download
    tickers=list(pd.read_sql("SELECT * FROM modelled_tickers",con=conn).ticker.values)
    t=Ticker(tickers,asynchronous=True)
    data_init=t.history(start=START_DATE,adj_ohlc=True,adj_timezone=False)
    data=data_init.reset_index()
    data=data.rename(columns={"symbol":"ticker","close":"Close"})
    data["date"]=pd.to_datetime(data.date.dt.date)
    data.to_sql("stock_price",con=conn,if_exists="append",index=False) ### create table with primary keys
