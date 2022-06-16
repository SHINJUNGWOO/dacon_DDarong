from tabnanny import verbose
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torchvision import transforms 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import string
from datetime import datetime
from sklearn.decomposition import PCA
from pytimekr import pytimekr
from lightgbm import LGBMRegressor

def nmae(true , pred ):
    score = np.mean((np.abs(true-pred))/true)

    return score
def data_to_int(str_data):
    str_data = str_data.split("-")
    val = int(str_data[0])*24*60 + int(str_data[1])*31 + int(str_data[2])
    return val

def find_hol(str_data):
    tmp_data = str_data.split("-")
    for i in pytimekr.holidays(int(tmp_data[0])):
        if str_data == str(i):
            return 1
    else:
        return 0

def finddow(str_data):
    return datetime.strptime(str_data.split()[0],"%Y-%m-%d").weekday()

def feature_engineering(df,train= True):
    if train == False:
        df["sunshine_sum"] = df["sunshine_sum"].fillna(0)
    
    df["precipitation"] = df["precipitation"].fillna(0)

    df = df.dropna(axis=0)

    df["dow"] = df['date'].apply(lambda x: finddow(x))
    df["holiday"] = df['date'].apply(lambda x: find_hol(x))
    df["holiday"] = df[['holiday','dow']].apply(lambda x: 1 if (x[0] == 1) or (x[1] in [5,6]) else 0 ,axis = 1)


    df['dow'] = df['dow'].astype('category') 
    df['holiday'] = df['holiday'].astype('category')
    #df = df.drop(columns ="date")

    if train == True:
        answer_df = df["rental"]
        df = df.drop(columns ="rental")
    else:
        answer_df = df["date"]
    
    #df["date"] = df['date'].apply(lambda x: data_to_int(x))
    return df,answer_df

if __name__ == '__main__':
    #Test Data
    file_path = "C:\D_file\개발연습\Test_dae\\train.csv"
    df = pd.read_csv(file_path)
    test, val = feature_engineering(df)


    train_x, test_x, train_y, test_y = train_test_split(test, val, test_size=0.2, random_state=42)
    

    params = {'learning_rate': 0.01, 
            'max_depth': 16, 
            'objective': 'regression', 
            'metric': 'mse',  
            'num_leaves': 10}

    lgbm_wrapper = LGBMRegressor(n_estimators=2000)

    lgbm_wrapper.fit(train_x,train_y,eval_metric="mse",eval_set = [(test_x,test_y)],verbose =1000)
    predict_test = lgbm_wrapper.predict(test_x)
    predic_test = np.around(predict_test)
    print(nmae(test_y,predict_test))

    file_path = "C:\D_file\개발연습\Test_dae\\test.csv"
    df = pd.read_csv(file_path)

    test, label = feature_engineering(df,train=False)
    predict_test = lgbm_wrapper.predict(test)
    predict_test = np.around(predict_test).astype(dtype=int)
    predict_test = pd.DataFrame(predict_test,columns=["rental"])
    answer = pd.concat([label,predict_test],axis = 1)
    answer = answer.set_index("date")
    answer.to_csv("answer.csv", mode='w')