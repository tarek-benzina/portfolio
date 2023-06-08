import logging
import numpy as np
import pandas as pd
import os

class Encoder(object):
    def __init__(self):
        self.categoricals=[]
        self.mappings=pd.DataFrame()
    def build(self,categoricals):
        self.categoricals=categoricals
    def load(self,directory):
        self.mappings=pd.read_csv(f"{directory}/encoder_mappings.csv")
        self.categoricals=list(np.load(f"{directory}/encoder_categoricals.npy"))
    def fit(self,train):
        train=train.copy()
        train["_toremove_"]=1
        dims=train.groupby(self.categoricals,as_index=False)["_toremove_"].count().drop(columns=["_toremove_"])
        self.mappings=pd.concat([dims,pd.get_dummies(dims,columns=self.categoricals,drop_first=True)],axis=1)
    def apply(self,data):
        data=data.merge(self.mappings,on=self.categoricals,how="left")
        data[self.get_encoded_features()]=data[self.get_encoded_features()].fillna(0)
        return data
    def get_encoded_features(self):
        return self.mappings.drop(columns=self.categoricals).columns.tolist()
    def save(self,directory="./encoder/"):
        #if os.path.exists(directory):
        #    raise ValueError(f"Directory {directory} exists already!")
        os.makedirs(f"{directory}",exist_ok=True)
        self.mappings.to_csv(f"{directory}/encoder_mappings.csv",index=False)
        np.save(f"{directory}/encoder_categoricals.npy",arr=self.categoricals)

class Normaliser(object):
    
    def __init__(self):
        self.timeseries_id_column=""
        self.target=""
        self.method=""
        self.min_max=pd.DataFrame()

    def build(self,target,timeseries_id_column,method="minmax"):
        self.timeseries_id_column=timeseries_id_column
        self.target=target
        self.method=method
        self.min_max=pd.DataFrame()

    def load(self,directory):
        self.min_max=pd.read_csv(f"{directory}/min_max.csv")
        attr=list(np.load(f"{directory}/normaliser_attr.npy"))
        self.timeseries_id_column=attr[0]
        self.target=attr[1]
        self.method=attr[2]


    def fit(self,train):
        train=train.copy()
        self.min_max=train.groupby(self.timeseries_id_column,as_index=False).agg(min_target=(self.target,"min"),max_target=(self.target,"max"))

    def normalise(self,data):
        data=data.merge(self.min_max,on=self.timeseries_id_column,how="left")
        data[f"{self.target}_scaled"]=(data[self.target]-data["min_target"])/(data["max_target"]-data["min_target"])
        return data.drop(columns=["min_target","max_target"])

    def denormalise(self,data,column):
        data=data.merge(self.min_max,on=self.timeseries_id_column,how="left")
        data[f"{column}_denormalised"]=data["min_target"]+data[column]*(data["max_target"]-data["min_target"])
        return data.drop(columns=["min_target","max_target"])

    def save(self,directory="./normaliser/"):
        #if os.path.exists(directory):
        #    raise ValueError(f"Directory {directory} exists already!")
        os.makedirs(f"{directory}",exist_ok=True)
        self.min_max.to_csv(f"{directory}/min_max.csv",index=False)
        np.save(f"{directory}/normaliser_attr.npy",arr=[self.timeseries_id_column,self.target,self.method])

def get_predictions(model,subset,subset_seq,steps=[0]):
    subset=subset.copy()
    pred=model.predict(subset_seq["X"])
    for step in steps:
        result=pd.DataFrame(subset_seq["Y_ids"][:,step,:],columns=["ticker","date"])
        result[f"pred_step_{step}"]=pred[:,step]
        subset=subset.merge(result,on=["ticker","date"],how="left")
    return subset


def get_latest_model_path(models_dir,prefix="model-"):
    models=[x for x in os.listdir(models_dir) if x.startswith(prefix)]
    if len(models)>0:
        logging.info(f"Latest model is {max(models)}")
        return f"{models_dir}/{max(models)}"
    else:
        raise ValueError(f"No models found with a prefix {prefix} under {models_dir}")
