import numpy as np
import pandas as pd



class TimeseriesSquencer:
    def __init__(self,features,target,history,future,steps,id_column,time_column,data_end_date,include_target_feature=False,extra_id_columns=[]):
        self.features=features
        self.target=target
        self.history=history
        self.future=future
        self.steps=steps
        self.id_column=id_column
        self.time_column=time_column
        self.data_end_date=data_end_date
        self.ids=[id_column,time_column]
        self.include_target_feature=include_target_feature
        self.extra_id_columns=extra_id_columns
        if self.include_target_feature:
            self.features=self.features+[self.target]
    def sequence_timeseries(self,df):
        df=df[df[self.time_column]<=self.data_end_date].copy()
        x_train=[]
        y_train=[]
        x_train_ids=[]
        y_train_ids=[]
        for ticker,data in df.groupby(self.id_column):            
            sequences=[(np.arange(i,i+self.history),
                        np.arange(i+self.history+self.future,i+self.history+self.future+self.steps)) 
                       for i in range(0,int(len(data))-self.history-self.future-self.steps+1)]
            data=data.sort_values(self.time_column)
            ticker_data=data[self.features].to_numpy()
            ticker_x=np.stack([ticker_data[seq[0]] for seq in sequences])
            
            data["id_column"]=ticker
            ticker_x_ids=data[["id_column",self.time_column]+self.extra_id_columns].to_numpy()
            id_x=np.stack([ticker_x_ids[seq[0]] for seq in sequences])
            
            ticker_y_ids=data[["id_column",self.time_column]+self.extra_id_columns].to_numpy()
            id_y=np.stack([ticker_y_ids[seq[1]] for seq in sequences])
            
            ticker_target=data[self.target].to_numpy()
            ticker_y=np.stack([ticker_target[seq[1]] for seq in sequences])
            
            x_train.append(ticker_x)
            x_train_ids.append(id_x)
            y_train.append(ticker_y)
            y_train_ids.append(id_y)
            
        
        sequenced_data={}
        sequenced_data={
            "X":np.concatenate(x_train),
            "Y":np.concatenate(y_train),
            "X_ids":np.concatenate(x_train_ids),
            "Y_ids":np.concatenate(y_train_ids),
        }
        return sequenced_data
    
    def sequence_timeseries_for_inference(self,df):
        x_train=[]
        y_train=[]
        x_train_ids=[]
        y_train_ids=[]
        for ticker,data in df.groupby(self.id_column):
            sequences=[(np.arange(i,i+self.history),np.arange(i+self.history,i+self.history+self.future)) for i in range(0,int(len(data[data[self.time_column]<=self.data_end_date]))-self.history+1)]
            if len(data[data[self.time_column]>self.data_end_date]) < self.future:
                raise ValueError(f"""Time series {self.id_column}: {ticker} does not have enough data for inferencing beyond the end date of the data.
                Please make sure to provide data with {self.future} calendar dates after the end date of the timeseries that can be used for inferencing""")
            data=data.sort_values(self.time_column)
            ticker_data=data[self.features+[self.target]].to_numpy()
            ticker_x=np.stack([ticker_data[seq[0]] for seq in sequences])
            
            data["id_column"]=ticker
            ticker_x_ids=data[["id_column",self.time_column]].to_numpy()
            id_x=np.stack([ticker_x_ids[seq[0]] for seq in sequences])
            
            ticker_y_ids=data[["id_column",self.time_column]].to_numpy()
            id_y=np.stack([ticker_y_ids[seq[1]] for seq in sequences])
            
            ticker_target=data[self.target].to_numpy()
            ticker_y=np.stack([ticker_target[seq[1]] for seq in sequences])
            
            x_train.append(ticker_x)
            x_train_ids.append(id_x)
            y_train.append(ticker_y)
            y_train_ids.append(id_y)
            
        
        sequenced_data={}
        sequenced_data={
            "X":np.concatenate(x_train),
            "Y":np.concatenate(y_train),
            "X_ids":np.concatenate(x_train_ids),
            "Y_ids":np.concatenate(y_train_ids),
        }
        return sequenced_data
