import numpy as np
import pandas as pd
from pycox import models
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import torch

from pycox.datasets import gbsg
from pycox.evaluation import EvalSurv
from models.keras_nn import hazardNN

np.random.seed(1234)
_ = torch.manual_seed(123)

num_nodes_list = [64,128,256]
lr_list = [0.1,0.01,0.001,0.0001]
batch_size_list = [64,128,256]
nn_config_list = []
for lr in lr_list:
    for batch_size in batch_size_list:
        for num_nodes in num_nodes_list:
            nn_config = {
                "hidden_layers_nodes": num_nodes,
                "learning_rate":lr,
                "activation": 'relu', 
                "optimizer": 'adam',
                "batch_size": batch_size,
                "patience": 10
            }
            nn_config_list.append(nn_config)


kf = KFold(n_splits=5)
data = gbsg.read_df()
data = data.sample(frac=1).reset_index(drop=True)

ci = []
bs = []
nbll = []
fold_n = 0

for train_index, test_index in kf.split(data):

    fold_n += 1
    print(fold_n)

    df_train = data.iloc[train_index]
    df_test = data.iloc[test_index]
    df_val = df_train.sample(frac=0.5)
    df_train = df_train.drop(df_val.index)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # expand the training data frame
    sorted_train = sorted(np.unique(df_train['duration']))
    N = len(df_train)
    id_list = []
    start_list = []
    end_list = []
    fail_list = []

    for i in range(N):
        d = df_train['duration'][i]
        e = df_train['event'][i]
        start = 0
        for time in sorted_train:
            if time > d: break
            id_list.append(i)
            start_list.append(start)
            end_list.append(time)
            start = time
            if time < d: fail_list.append(0)
            else: fail_list.append(e)

    df = pd.DataFrame({'id':id_list,'fail':fail_list,'start':start_list,'end':end_list})
    df_train['id'] = df_train.index
    df_train = pd.merge(df_train,df)
    df_train = df_train.drop(columns=['duration','event'])

    # expand the validation data frame
    sorted_val = sorted(np.unique(df_val['duration']))
    N = len(df_val)
    id_list = []
    start_list = []
    end_list = []
    fail_list = []

    for i in range(N):
        d = df_val['duration'][i]
        e = df_val['event'][i]
        start = 0
        for time in sorted_val:
            if time > d: break
            id_list.append(i)
            start_list.append(start)
            end_list.append(time)
            start = time
            if time < d: fail_list.append(0)
            else: fail_list.append(e)

    df = pd.DataFrame({'id':id_list,'fail':fail_list,'start':start_list,'end':end_list})
    df_val['id'] = df_val.index
    df_val = pd.merge(df_val,df)
    df_val = df_val.drop(columns=['duration','event'])

    # expand the test data frame by training partition
    durations_test = np.array(df_test['duration'])
    events_test = np.array(df_test['event'])

    N = len(df_test)
    id_list = []
    start_list = []
    end_list = []
    fail_list = []

    for i in range(N):
        start = 0
        for time in sorted_train:
            id_list.append(i)
            start_list.append(start)
            end_list.append(time)
            start = time

    df = pd.DataFrame({'id':id_list,'start':start_list,'end':end_list})
    df_test['id'] = df_test.index
    df_test_1 = pd.merge(df_test,df)
    df_test_1 = df_test_1.drop(columns=['duration','event'])

    # expand the test data frame by validation partition
    id_list = []
    start_list = []
    end_list = []
    fail_list = []

    for i in range(N):
        start = 0
        for time in sorted_val:
            id_list.append(i)
            start_list.append(start)
            end_list.append(time)
            start = time

    df = pd.DataFrame({'id':id_list,'start':start_list,'end':end_list})
    df_test_2 = pd.merge(df_test,df)
    df_test_2 = df_test_2.drop(columns=['duration','event'])

    min_loss = np.inf
    # grid search best model
    for nn_config in nn_config_list:

        print(nn_config)
        # model 1
        model_1_tmp = hazardNN(nn_config)
        x_mean_1_tmp, x_std_1_tmp, loss_1 = model_1_tmp.train(
            x_train = df_train.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,6,9,10]]),
            fail_train = df_train.groupby('id').apply(lambda x: x.values[:,8]),
            x_valid = df_val.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,6,9,10]]),
            fail_valid = df_val.groupby('id').apply(lambda x: x.values[:,8])
            )

        # model 2
        model_2_tmp = hazardNN(nn_config)
        x_mean_2_tmp, x_std_2_tmp, loss_2 = model_2_tmp.train(
            x_train = df_val.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,6,9,10]]),
            fail_train = df_val.groupby('id').apply(lambda x: x.values[:,8]),
            x_valid = df_train.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,6,9,10]]),
            fail_valid = df_train.groupby('id').apply(lambda x: x.values[:,8])
            )

        loss = (loss_1+loss_2)/2
        print("loss",loss)
        if loss<min_loss:
            min_loss = loss
            model_1, model_2 = model_1_tmp, model_2_tmp
            x_mean_1, x_std_1 = x_mean_1_tmp, x_std_1_tmp
            x_mean_2, x_std_2 = x_mean_2_tmp, x_std_2_tmp


    x_test_1 = df_test_1.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,6,8,9]])
    x_test_1 = tf.ragged.constant(x_test_1).to_tensor()
    dt_test_1 = tf.expand_dims(x_test_1[:,:,-1]-x_test_1[:,:,-2],-1)  
    x_test_1 = (x_test_1-x_mean_1)/x_std_1

    df_test_1['pred_survival'] = np.exp(model_1.model.predict([x_test_1,dt_test_1]).flatten())
    df_test_1['pred_survival'] = df_test_1['pred_survival']*(df_test_1['end']-df_test_1['start'])
    df_test_1['pred_survival'] = df_test_1.groupby(['id'])['pred_survival'].cumsum()
    df_test_1['pred_survival'] = np.exp(-df_test_1['pred_survival'])

    surv_1 = df_test_1.pivot_table(index='end',columns='id',values='pred_survival')

    
    x_test_2 = df_test_2.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,6,8,9]])
    x_test_2 = tf.ragged.constant(x_test_2).to_tensor()
    dt_test_2 = tf.expand_dims(x_test_2[:,:,-1]-x_test_2[:,:,-2],-1)  
    x_test_2 = (x_test_2-x_mean_2)/x_std_2

    df_test_2['pred_survival'] = np.exp(model_2.model.predict([x_test_2,dt_test_2]).flatten())
    df_test_2['pred_survival'] = df_test_2['pred_survival']*(df_test_2['end']-df_test_2['start'])
    df_test_2['pred_survival'] = df_test_2.groupby(['id'])['pred_survival'].cumsum()
    df_test_2['pred_survival'] = np.exp(-df_test_2['pred_survival'])

    surv_2 = df_test_2.pivot_table(index='end',columns='id',values='pred_survival')


    for i in range(len(surv_2)):
        try:
            surv_2.iloc[i,:] = surv_1.loc[surv_1.index<=surv_2.index[i],:].iloc[-1,:]
        except IndexError:
            surv_2.iloc[i,:] = 1

    surv = pd.concat([surv_1,surv_2])

    surv_2 = df_test_2.pivot_table(index='end',columns='id',values='pred_survival')

    for i in range(len(surv_1)):
        try:
            surv_1.iloc[i,:] = surv_2.loc[surv_2.index<=surv_1.index[i],:].iloc[-1,:]
        except IndexError:
            surv_1.iloc[i,:] = 1

    surv = (pd.concat([surv_1,surv_2])+surv)/2
    surv = surv.sort_index()
    surv = surv.drop_duplicates()

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    ci.append(ev.concordance_td())
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    _ = ev.brier_score(time_grid).plot()
    bs.append(ev.integrated_brier_score(time_grid))
    nbll.append(ev.integrated_nbll(time_grid))
    print(ci,bs,nbll)

print(np.mean(ci),np.mean(bs),np.mean(nbll))